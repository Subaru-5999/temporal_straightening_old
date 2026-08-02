import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: OPTIONAL initialization for the first sub-planner call, shape
                (B, horizon, action_dim), normalized -- i.e. the same convention
                `GDPlanner.init_actions` returns. None (the default, and what every normal run
                passes) reproduces the original behavior exactly: the sub-planner initializes
                from scratch per `sample_type`.

                This used to be documented as "actions is NOT used" and was silently dropped,
                which made `plan.py`'s `debug_dset_init` flag a NO-OP: every config in this repo
                plans through MPCPlanner (open-loop is just `max_iter=1`), so the ground-truth
                initialization never reached the optimizer. Verified by the symptom -- runs with
                and without `debug_dset_init=true` returned bit-identical success arrays.

                Forwarding it makes the flag work, which is what enables the decisive
                objective-vs-optimizer diagnostic: `plan.py` sets it to `gt_actions`, the dataset
                action sequence whose env rollout DEFINED this goal, so it reaches the goal by
                construction. Combined with `opt_steps=0` (evaluate the initialization, no
                optimization) and `opt_steps=100` (optimize from it), this measures whether
                minimizing the latent objective preserves or destroys a known-good plan.
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()

        cur_obs_0 = obs_0
        memo_actions = actions
        while not np.all(self.is_success) and self.iter < self.max_iter:
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            actions, _ = self.sub_planner.plan(
                obs_0=cur_obs_0,
                obs_g=obs_g,
                actions=memo_actions,
                step=self.iter,
            )  # (b, t, act_dim)
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(
                obs_0=init_obs_0,
                state_0=init_state_0,
            )
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
                filename=f"plan{self.iter}",
                save_video=True,
            )
            new_successes = successes & ~self.is_success  # Identify new successes
            self.is_success = (
                self.is_success | successes
            )  # Update overall success status
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            # Pair the sub-planner's PER-TASK objective values with the per-task outcome, once,
            # on the first MPC iteration (for open-loop, max_iter=1, that is the whole run).
            # This is the strongest test of the noise-floor mechanism available without training:
            # one checkpoint, one protocol, 50 paired observations, no cross-checkpoint confound.
            # Written to the run's output dir (plan.py has already chdir'd there).
            probe = getattr(self.sub_planner, "last_probe_per_task", None)
            if probe is not None and self.iter == 0:
                try:
                    import json as _json
                    with open("probe_per_task.json", "w") as _f:
                        _json.dump({
                            "obj_init": probe["obj_init"],
                            "obj_final": probe["obj_final"],
                            "success": [bool(x) for x in successes],
                            "state_dist": [float(x) for x in logs.get("state_dist", [])]
                            if hasattr(logs.get("state_dist", []), "__iter__") else [],
                        }, _f)
                    print("[probe] wrote probe_per_task.json", flush=True)
                except Exception as _e:      # diagnostics must never break an eval
                    print(f"[probe] could not write probe_per_task.json: {_e}", flush=True)
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            # Free cached GPU memory between iterations. The executed horizon grows
            # each iter, so memory climbs; on a MIG slice, memory pressure triggers
            # torch's NVML free-memory query which asserts. Clearing avoids that.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len
