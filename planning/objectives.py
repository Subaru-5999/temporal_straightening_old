import numpy as np
import torch
import torch.nn as nn


def corridor_penalty(z_obs_pred, z_obs_tgt, rho=0.0, eps=1e-8):
    """LATENT GEODESIC CORRIDOR: how far the PREDICTED latent path strays SIDEWAYS off the
    straight segment from the start latent to the goal latent. Returns (B,) -- per-eval, so it
    drops straight into the existing per-eval objective.

    THE PROBLEM IT FIXES. Open-loop planning minimises the TERMINAL cost only
    (`objective_fn_last`: `z_obs_pred["visual"][:, -1:]` vs the goal). The intermediate
    predicted latents are therefore COMPLETELY UNCONSTRAINED: any action sequence whose final
    predicted latent lands near the goal scores equally well, however absurd the path in
    between. 100 Adam steps through a differentiable world model is a strong optimiser, and a
    strong optimiser against an unconstrained interior is exactly how you find the model's
    adversarial minima -- action sequences the model CONFIDENTLY BELIEVES reach the goal but
    which do not survive contact with the real environment. The measured baseline is consistent
    with this: the predictor is very accurate on DATASET actions (validation skill vs
    persistence 0.04-0.09 at k=1..4) yet open-loop PushT still fails ~25% of the time, and MPC
    -- which re-observes and thus cannot be fooled for more than one step -- beats it by ~14
    points. That 14-point open-loop/MPC gap is the signature of a plan whose interior is wrong.

    THE PRIOR, AND WHY IT IS THE PAPER'S OWN. Straightening is trained so that real latent
    trajectories are locally straight, and the paper's PCA/heatmap analyses argue Euclidean
    distance then approximates geodesic distance. If both hold, the latent path a REAL agent
    takes from z_0 to z_g is close to the straight segment between them. So a predicted path
    that bows far off that segment is off the model's own training manifold -- it is
    extrapolation, and extrapolation is where the model is unreliable. Penalising it is a
    TRUST REGION expressed in the geometry the paper spent its whole training budget building.

    WHY NOT JUST USE `mode=all`. Three regimes, and the difference is the point:
      mode=last : endpoint constrained, interior free            -> exploitable
      mode=all  : EVERY frame pulled toward the GOAL             -> demands premature arrival,
                  i.e. it prices in a *wrong* prior (be at the goal at t=1)
      corridor  : every frame pulled toward the LINE, FREE ALONG IT
    The corridor removes exactly the degeneracy that `mode=last` leaves, and adds no opinion
    about WHEN progress happens. That distinction matters here because latent speed is not
    uniform in practice, so any objective that fixes the progress schedule (`all`, or
    equally-spaced waypoints) fights the dynamics. Decomposing the deviation into a component
    ALONG the segment and a component PERPENDICULAR to it, and charging only the perpendicular
    part, makes the term invariant to the progress rate by construction.

    MATH. With F the flattened visual-latent dimension, u = z_g - z_0, and d_k = z_k - z_0:
        along_k = (<d_k, u> / ||u||^2) * u          (component on the segment)
        perp_k  = d_k - along_k                     (sideways deviation)
        dev_k   = ||perp_k|| / ||u||                (SCALE-FREE: normalised by segment length)
        penalty = mean_k [ max(0, dev_k - rho) ]^2
    `dev_k` is dimensionless and invariant to any global rescaling of the latent, so the
    coefficient means the same thing across checkpoints -- the lesson from the rollout failure,
    where a raw latent-space magnitude was comparable across neither runs nor scales.
    `rho` is a DEAD ZONE: real trajectories are not perfectly straight either, so set rho to a
    high quantile of the deviation REAL dataset segments exhibit (measure it, do not tune it on
    success) and the term charges nothing until a plan leaves the envelope of real behaviour.
    rho=0 makes it a plain quadratic pull toward the segment.

    COST: zero. `wm.rollout` already returns every intermediate latent; `objective_fn_all`
    already consumes them. No extra forward pass, no retraining -- this is evaluated on the
    checkpoints that already exist.

    Frames: index 0 of `z_obs_pred` is the ENCODED REAL start observation (see
    `VWorldModel.rollout`: it concatenates the encoding of obs_0 before any prediction), and the
    last index is the terminal prediction the goal term already handles. So the corridor is
    charged on the strict interior, indices 1..T-2. z_0 is detached: it is a constant with
    respect to the actions being optimised.
    """
    zp = z_obs_pred["visual"]
    b, t = zp.shape[0], zp.shape[1]
    if t < 3:
        return zp.new_zeros(b)          # no interior frames -> nothing to constrain
    z0 = zp[:, 0:1].reshape(b, 1, -1).detach()
    zg = z_obs_tgt["visual"].reshape(b, 1, -1)
    zk = zp[:, 1 : t - 1].reshape(b, t - 2, -1)

    u = zg - z0                                             # (b, 1, F)
    u_sq = u.pow(2).sum(-1, keepdim=True).clamp_min(eps)     # ||z_g - z_0||^2
    d = zk - z0                                             # (b, K, F)
    proj = (d * u).sum(-1, keepdim=True) / u_sq              # scalar coeff along the segment
    perp = d - proj * u
    dev = (perp.pow(2).sum(-1) / u_sq.squeeze(-1)).clamp_min(0).sqrt()   # (b, K), scale-free
    if rho and rho > 0:
        dev = (dev - rho).clamp_min(0.0)                     # dead zone at real-data straightness
    return dev.pow(2).mean(dim=1)


def create_objective_fn(alpha, base, mode="last", corridor_beta=0.0, corridor_rho=0.0):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
        corridor_beta: weight of the LATENT GEODESIC CORRIDOR term (see `corridor_penalty`).
            0.0 -> the returned objective is bit-identical to the paper's. Added to EVERY mode
            rather than introducing new mode names, so `mode=last` + beta gives the open-loop
            variant and `mode=staged` + beta the PushT MPC variant, with the paper's own mode
            semantics untouched.
        corridor_rho: dead-zone radius on the normalised sideways deviation; measure it from
            real dataset segments, do not tune it against success.
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")
    corridor_beta = float(corridor_beta or 0.0)
    corridor_rho = float(corridor_rho or 0.0)

    def objective_fn_last(z_obs_pred, z_obs_tgt, step=None):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt, step=None, coeffs=None, base=base):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        if coeffs is None:
            coeffs = np.array([base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32)
            coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        else:
            coeffs = coeffs.to(z_obs_pred["visual"].device)

        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_staged(z_obs_pred, z_obs_tgt, step=None):
        if step is None:
            return objective_fn_all(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)
        # stage 1: optimize only terminal match
        if step < z_obs_pred["visual"].shape[1] - 1:
            return objective_fn_last(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt)
        # stage 2: use the full-horizon weighted objective
        else:
            return objective_fn_all(z_obs_pred=z_obs_pred, z_obs_tgt=z_obs_tgt, coeffs=None)

    if mode == "last":
        base_fn = objective_fn_last
    elif mode == "all":
        base_fn = objective_fn_all
    elif mode == "staged":
        base_fn = objective_fn_staged
    else:
        raise NotImplementedError

    if corridor_beta == 0.0:
        return base_fn      # paper-faithful path: same function object, nothing wrapped

    def objective_fn_corridor(z_obs_pred, z_obs_tgt, step=None, **kwargs):
        goal_cost = base_fn(z_obs_pred, z_obs_tgt, step=step, **kwargs)
        pen = corridor_penalty(z_obs_pred, z_obs_tgt, rho=corridor_rho)
        return goal_cost + corridor_beta * pen

    return objective_fn_corridor
