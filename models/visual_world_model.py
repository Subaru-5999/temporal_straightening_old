import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
import logging
from torchvision import transforms
from einops import rearrange, repeat

log = logging.getLogger(__name__)

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        straighten=False,
        stop_grad=True,
        vcreg=False,
        vcreg_std_coeff=0,
        vcreg_cov_coeff=0,
        vcreg_apply_to="enc",
        straighten_scales=None,
        straighten_scale_weights=None,
        straighten_goal_weight=0.0,
        straighten_lambdas=None,
        straighten_speed_lambda=0.0,
        rollout_steps=1,
        rollout_gamma=0.9,
        rollout_batch_frac=1.0,
        rollout_checkpoint=True,
        iso_lambda=0.0,
        iso_eps=0.1,
        iso_checkpoint=True,
        **kwargs,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used
        self.straighten = False
        self.straighten_scale = 0.0
        self.curvature_mode = None
        self.stop_grad = bool(stop_grad)
        self.vcreg = bool(vcreg)
        self.std_coeff = float(vcreg_std_coeff)
        self.cov_coeff = float(vcreg_cov_coeff)
        if vcreg_apply_to != "enc":
            raise ValueError(
                f"Only encoder VCReg is supported, got vcreg_apply_to='{vcreg_apply_to}'."
            )

        if isinstance(straighten, str):
            if straighten.startswith("aggcos"):
                suffix = straighten.replace("aggcos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "aggcos"
            elif straighten.startswith("cos"):
                suffix = straighten.replace("cos", "")
                self.straighten_scale = float(suffix) if suffix else 1.0
                self.curvature_mode = "cos"

        # ---- Multi-scale (hierarchical) straightening config ----------------------
        # Backward compatible: scales == [1] reproduces the paper's single-scale
        # (consecutive-frame) curvature loss exactly. Set e.g. [1, 4] to also
        # straighten at coarser temporal scales. Curvature at scale s needs >= 2s+1
        # latent frames in the window (enforced by the dataloader window length).
        if not straighten_scales:
            self.straighten_scales = [1]
        else:
            self.straighten_scales = [int(s) for s in straighten_scales]

        # Per-scale coefficients. Two mutually-exclusive ways to specify them:
        #   (A) straighten_lambdas: ABSOLUTE lambda per scale, e.g. [0.1, 0.2]. PREFERRED for
        #       ablations. loss += sum_s lambda_s * L_curv^(s). straighten_lambdas takes
        #       precedence and folds the global scale into 1.0, so the effective coefficient
        #       equals the value you pass. Setting a lambda to 0 fully DISABLES that scale
        #       (its loss term AND, via the dataloader, its window requirement) -> e.g.
        #       lambdas=[0.1, 0] on scales=[1,4] is bit-identical to the paper's single-scale.
        #   (B) straighten_scale_weights (legacy): multipliers on the global straighten_scale
        #       parsed from the `straighten` string (aggcos1e-1 -> 0.1). Effective coefficient
        #       is straighten_scale * w_s. Kept so earlier runs reproduce exactly.
        if straighten_lambdas is not None:
            assert len(straighten_lambdas) == len(self.straighten_scales), (
                "straighten_lambdas must have the same length as straighten_scales"
            )
            self.straighten_scale = 1.0
            self.straighten_scale_weights = [float(l) for l in straighten_lambdas]
        elif straighten_scale_weights is None:
            self.straighten_scale_weights = [1.0] * len(self.straighten_scales)
        else:
            assert len(straighten_scale_weights) == len(self.straighten_scales), (
                "straighten_scale_weights must have the same length as straighten_scales"
            )
            self.straighten_scale_weights = [float(w) for w in straighten_scale_weights]

        # weight mu of the optional directional (goal-aligned) term; 0 = disabled
        self.straighten_goal_weight = float(straighten_goal_weight)

        # ---- Arc-length (constant-speed) consistency; 0.0 == paper exactly ----------------
        # The paper's cosine curvature is SCALE-INVARIANT in each argument, so it constrains
        # only the DIRECTION of the latent velocity and puts no pressure whatsoever on its
        # MAGNITUDE. But the appendix proposition that justifies the cosine proxy
        # (Assumption "Constant velocity and smooth actions") ASSUMES ||v_t||_2 = c for all t.
        # This term supplies the missing half of that assumption. See
        # `_scale_speed_consistency` for the derivation and why this is NOT the rejected
        # smoothness loss.
        self.straighten_speed_lambda = float(straighten_speed_lambda)

        # ---- Multi-step rollout-consistency loss (NOVEL; rollout_steps=1 == paper exactly) ----
        # The predictor is TRAINED one step ahead (teacher forcing: every input is a real encoded
        # frame) but USED autoregressively for H steps at planning time, consuming its own output.
        # Nothing in the one-step objective constrains how the predictor TRANSFORMS inherited
        # error, so error can be amplified each step (e_{k+1} ~ L*e_k + eps). This term adds
        #   sum_{k=2..K} gamma^(k-1) * || rollout_k(z) - sg(z_{num_hist+k-1}) ||^2
        # so gradients flow through the COMPOSITION f(f(...)) and the model is penalised for
        # amplifying its own error. K=1 adds nothing -> bit-identical to the paper.
        self.rollout_steps = max(1, int(rollout_steps) if rollout_steps else 1)
        self.rollout_gamma = float(rollout_gamma)
        # Minimum window: k-step targets need real frames up to index num_hist + K - 1.
        self.rollout_min_frames = self.num_hist + self.rollout_steps
        # MEMORY LEVER. The K chained predict() calls have COMPOSED autograd graphs (step k's
        # graph contains steps 1..k-1), and each call retains ~(b, heads, T*p, T*p) attention
        # maps per layer. Computing the rollout term on a random SUB-BATCH is an unbiased
        # estimate of the full-batch mean (same expectation, slightly noisier), and cuts this
        # term's memory proportionally. 1.0 = use the whole batch (default).
        self.rollout_batch_frac = float(rollout_batch_frac)
        if not (0.0 < self.rollout_batch_frac <= 1.0):
            raise ValueError(
                f"rollout_batch_frac must be in (0, 1], got {self.rollout_batch_frac}"
            )
        # PRIMARY memory lever: gradient (activation) checkpointing on the rollout's predict()
        # calls. Without it each call retains, per ViT layer, the (b, heads, num_hist*p,
        # num_hist*p) attention score tensor AND its softmax output -- for depth=6, heads=16,
        # b=32, num_hist*p=3*196=588 that is ~8.5 GB per predict() in fp32, so K=4 extra calls
        # alone need ~34 GB and OOM a 45 GB MIG slice. With checkpointing each rollout step
        # stores only its input/output and RECOMPUTES the interior during backward, so peak
        # memory is ~one predict() instead of K, at roughly +30% step time. Mathematically a
        # no-op: identical gradients (PyTorch preserves RNG + autocast state).
        self.rollout_checkpoint = bool(rollout_checkpoint)
        if self.rollout_steps > 1:
            log.info(
                "Multi-step rollout consistency ENABLED: K=%s gamma=%s batch_frac=%s "
                "grad_checkpoint=%s (needs window>=%s frames)",
                self.rollout_steps,
                self.rollout_gamma,
                self.rollout_batch_frac,
                self.rollout_checkpoint,
                self.rollout_min_frames,
            )
        else:
            log.info("Multi-step rollout consistency disabled (K=1, paper-exact)")

        # ---- Action-isometry conditioning term (NOVEL; iso_lambda=0 == paper exactly) --------
        # Targets kappa(B) for B = dz_{t+1}/da_t -- the HORIZON-INDEPENDENT prefactor in the
        # paper's own conditioning bound, which straightening cannot touch. Deliberately
        # SCALE-FREE so it cannot be satisfied by shrinking the latent (the failure mode that
        # sank the rollout-consistency term). Needs NO window widening, so iterations/epoch and
        # the straightening window stay bit-identical to the paper.
        self.iso_lambda = float(iso_lambda)
        self.iso_eps = float(iso_eps)
        self.iso_checkpoint = bool(iso_checkpoint)
        if self.iso_lambda > 0:
            log.info(
                "Action-isometry conditioning ENABLED: lambda=%s eps=%s grad_checkpoint=%s "
                "(2 extra predictor calls/step, no window change)",
                self.iso_lambda, self.iso_eps, self.iso_checkpoint,
            )
        else:
            log.info("Action-isometry conditioning disabled (lambda=0, paper-exact)")

        # Effective ABSOLUTE lambda per scale (= straighten_scale * weight); for logging/inspection.
        self.straighten_effective_lambdas = [
            self.straighten_scale * w for w in self.straighten_scale_weights
        ]
        # A scale is ACTIVE only if its coefficient > 0. lambda_s = 0 disables scale s entirely:
        # no loss contribution here, and no window widening (see train.py) -> switching a scale
        # off reverts both the loss and the data pipeline to the remaining active scales.
        self.straighten_active_scales = [
            s for s, w in zip(self.straighten_scales, self.straighten_scale_weights) if w > 0
        ]

        # Straightening runs only if a mode is set, the global scale > 0, and >= 1 scale is
        # active. (All lambdas == 0 => straightening off => pure prediction loss.)
        self.straighten = (
            self.curvature_mode is not None
            and self.straighten_scale > 0
            and len(self.straighten_active_scales) > 0
        )

        # Training window must hold >= 2s+1 frames for the largest ACTIVE scale only.
        if self.straighten_active_scales:
            self.straighten_min_frames = 2 * max(self.straighten_active_scales) + 1
        else:
            self.straighten_min_frames = self.num_hist + self.num_pred

        log.info("num_action_repeat: %s", self.num_action_repeat)
        log.info("num_proprio_repeat: %s", self.num_proprio_repeat)
        log.info("proprio encoder: %s", proprio_encoder)
        log.info("action encoder: %s", action_encoder)
        log.info("proprio_dim: %s, after repeat: %s", proprio_dim, self.proprio_dim)
        log.info("action_dim: %s, after repeat: %s", action_dim, self.action_dim)
        log.info("emb_dim: %s", self.emb_dim)
        if self.straighten:
            log.info(
                "Straightening enabled: mode=%s, scale=%s",
                self.curvature_mode,
                self.straighten_scale,
            )
            log.info(
                "Multi-scale straightening: scales=%s effective_lambdas=%s active_scales=%s "
                "goal_weight=%s (needs window>=%s frames)",
                self.straighten_scales,
                self.straighten_effective_lambdas,
                self.straighten_active_scales,
                self.straighten_goal_weight,
                self.straighten_min_frames,
            )
        else:
            log.info("Straightening disabled")
        log.info("Stop-grad enabled: %s", self.stop_grad)
        log.info(
            "VCReg enabled: %s, apply_to=enc, std_coeff=%s, cov_coeff=%s",
            self.vcreg,
            self.std_coeff,
            self.cov_coeff,
        )

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        log.info("Model emb_dim: %s", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
        }
        return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def visual_only(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-2, :]
        drop = self.proprio_dim + self.action_dim
        return z[..., :-drop] if drop > 0 else z

    def visual_prop(self, z):
        if self.concat_dim == 0:
            return z[:, :, :-1, :]
        return z[..., :-self.action_dim]

    def vcreg_std_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1 - std_x))

    def vcreg_cov_loss(self, z: torch.Tensor) -> torch.Tensor:
        x = z.reshape(-1, z.shape[-1])
        _, d = x.shape
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum() / d
        return cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _cos_curvature(self, v1, v2, eps=1e-6, step_thresh=1e-6):
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=eps)
        loss = 1.0 - cos
        if step_thresh > 0:
            step1 = v1.norm(dim=-1)
            step2 = v2.norm(dim=-1)
            mask = (step1 > step_thresh) & (step2 > step_thresh)
            loss = loss[mask]
        return loss.mean()

    def _scale_velocity_curvature(self, z, s):
        """Scale-s straightening term  L_curv^(s) = 1 - mean_t cos(v_t^(s), v_{t+s}^(s)),
        where the scale-s velocity is v_t^(s) = z_{t+s} - z_t (paper Eq. 3-4/6 with a
        step gap of s). Returns None if the window is too short (needs >= 2s+1 frames).
        For s == 1 this is identical to the paper's consecutive-frame curvature.
        z: (b, T, ..., d); cosine is taken over the last dim."""
        T = z.shape[1]
        if T < 2 * s + 1:
            return None
        va = z[:, s:] - z[:, :-s]      # v_t^(s) = z_{t+s} - z_t   (length T - s)
        v1 = va[:, :-s]                # v_t^(s)
        v2 = va[:, s:]                 # v_{t+s}^(s)
        return self._cos_curvature(v1, v2)

    def _scale_speed_consistency(self, z, s, eps=1e-6, step_thresh=1e-6):
        """Arc-length consistency at scale s:  L_speed^(s) = mean_t [ r_t + 1/r_t - 2 ],
        with  r_t = ||v_{t+s}^(s)|| / ||v_t^(s)||.  Zero iff consecutive latent steps have
        EQUAL length; strictly positive otherwise.

        WHY THIS TERM EXISTS (it is derived, not guessed).
        The appendix proposition that licenses "high cosine similarity => A is close to I"
        rests on Assumption `as:app_cos_const`, ||v_t||_2 = c for ALL t. That assumption is
        used exactly once, to turn a norm into a cosine:
            ||v_{t+1} - v_t||^2 = ||v_t||^2 + ||v_{t+1}||^2 - 2||v_t|| ||v_{t+1}|| C_t
                                = 2 c^2 (1 - C_t)          <-- only if ||v_{t+1}|| = ||v_t|| = c
        Nothing in the training loss enforces it: cosine similarity is invariant to the length
        of each velocity, so the deployed regularizer never sees ||v_t||. Dropping the
        assumption and redoing the same algebra with r_t = ||v_{t+1}||/||v_t|| gives the exact,
        assumption-free identity
            ||v_{t+1} - v_t||^2 / (||v_t|| ||v_{t+1}||)  =  (r_t + 1/r_t - 2) + 2 (1 - C_t)
        and hence the honest version of the paper's own directional bound
            ||(A - I) v_hat_t||  <=  sqrt( (r_t - 1)^2 + 2 r_t (1 - C_t) ) + sigma_max(B) * Da / ||v_t||
        (set r_t = 1 to recover the paper's Eq. app_dir_point_const exactly).

        CONSEQUENCE: driving the cosine term to its optimum, C_t -> 1, does NOT drive the
        bound to zero. A residual of |r_t - 1| survives, and eps = ||A - I|| enters the
        conditioning bound as ((1+eps)/(1-eps))^(2(H-1)), i.e. amplified exponentially in the
        planning horizon. Speed mismatch is therefore an un-attacked floor on exactly the
        quantity straightening is trying to shrink. This term attacks that floor, and the
        identity above says the two terms are the two halves of ONE quantity: the
        geometric-mean-normalised velocity difference. Their relative weight is thus FIXED by
        the algebra (1 : 2), not by hand tuning -- with the paper's lambda_curv = 0.1, the
        matched speed coefficient is lambda_speed = 0.05.

        WHY THIS IS NOT THE SMOOTHNESS LOSS THE PAPER REJECTED (App. `app:smooth_tc`).
        That loss is  E||z_{t+1} - z_t||^2: it penalises the MAGNITUDE of motion, so it is
        minimised by collapsing distinct states onto each other, which is what the paper
        observed. This term is a function of the RATIO only, so under a global rescaling
        z -> a*z every r_t is unchanged and the loss is EXACTLY constant: its gradient has no
        radial component, and shrinking the latent buys nothing. That property is the specific
        lesson from the rollout-consistency failure (encoder temporal contrast collapsed ~21%,
        planning fell 19-27 points) and from the scale-shortcut in the isometry term.
        The r + 1/r - 2 form (rather than (r-1)^2) is used because it is symmetric under
        r -> 1/r, so speeding up and slowing down are penalised identically; (r-1)^2 charges
        r=2 four times as much as r=1/2 and would bias the model toward deceleration, i.e.
        back toward the collapse direction.

        COST: zero extra forward passes and NO window widening -- it reuses the very same
        velocities the curvature term already forms, so iterations/epoch, the data window and
        the straightening window stay bit-identical to the baseline. This is the only extension
        so far in which the added loss term is the ONLY difference from the paper run.

        Returns None if the window is too short (needs >= 2s+1 frames) or every step was
        masked out as numerically stationary.
        """
        T = z.shape[1]
        if T < 2 * s + 1:
            return None
        # fp32: under bf16 autocast a ratio of two O(1) norms keeps only ~3 significant
        # digits, and this term is a small difference around r = 1.
        va = (z[:, s:] - z[:, :-s]).float()
        n1 = va[:, :-s].norm(dim=-1)        # ||v_t^(s)||
        n2 = va[:, s:].norm(dim=-1)         # ||v_{t+s}^(s)||
        mask = (n1 > step_thresh) & (n2 > step_thresh)
        r = n2.clamp_min(eps) / n1.clamp_min(eps)
        term = r + 1.0 / r - 2.0            # = (sqrt(r) - 1/sqrt(r))^2 >= 0, zero iff r == 1
        term = term[mask]
        if term.numel() == 0:
            return None
        return term.mean()

    def _scale_goal_alignment(self, z, s, z_goal):
        """Optional directional term  1 - mean_t cos(v_t^(s), z_goal - z_t): encourage each
        scale-s velocity to point toward the goal. At training time we have no true goal,
        so z_goal is the window's last latent (a pseudo-goal). Returns None if too short."""
        T = z.shape[1]
        if T < s + 1:
            return None
        va = z[:, s:] - z[:, :-s]                 # v_t^(s)              (length T - s)
        to_goal = z_goal.unsqueeze(1) - z[:, :-s]  # (z_goal - z_t)       (length T - s)
        cos = F.cosine_similarity(va, to_goal, dim=-1, eps=1e-6)
        return (1.0 - cos).mean()

    def _inject_action(self, z_frame, act_raw):
        """Replace the ACTION channels of a predicted latent with the encoded REAL action.

        The action is an INPUT, not something to predict, so during a rollout each newly
        predicted frame must have its action slot overwritten with the true next action --
        exactly what `replace_actions_from_z` does inside `rollout()`. This variant is
        NON-IN-PLACE (built with torch.cat) so it is always safe for autograd during training.
        z_frame: (b, 1, num_patches, dim);  act_raw: (b, 1, raw_action_dim)
        """
        if self.concat_dim != 1:
            raise NotImplementedError(
                "Multi-step rollout loss currently supports concat_dim=1 only "
                f"(got concat_dim={self.concat_dim})."
            )
        act_emb = self.encode_act(act_raw)                                  # (b, 1, a)
        act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_frame.shape[2])
        act_rep = act_tiled.repeat(1, 1, 1, self.num_action_repeat)          # (b,1,p,action_dim)
        return torch.cat([z_frame[..., : -self.action_dim], act_rep], dim=-1)

    def _predict_maybe_ckpt(self, hist, use_ckpt=None):
        """`predict(hist)` with optional activation checkpointing (training + grad only).

        Same output and same gradients as `self.predict(hist)`; trades ~30% extra compute for
        an O(calls) -> O(1) reduction in retained attention activations. Disabled automatically
        when grad is off (eval/val), where there is nothing to retain anyway.
        use_ckpt=None defaults to the rollout flag (back-compatible); callers with their own
        knob (e.g. the action-isometry term) pass it explicitly.
        """
        flag = self.rollout_checkpoint if use_ckpt is None else bool(use_ckpt)
        if flag and self.training and torch.is_grad_enabled():
            return ckpt.checkpoint(self.predict, hist, use_reentrant=False)
        return self.predict(hist)

    def _orthonormal_action_probes(self, b, d_a, device, dtype):
        """Two per-sample RANDOM ORTHONORMAL directions in raw-action space, (b, 1, d_a) each.

        Orthonormal (not merely random) so the isotropy target is simple: for B^T B = c*I the two
        directions must give EQUAL squared response and ZERO cross term. Gram-Schmidt on a
        d_a-vector is negligible cost (d_a = 10 for PushT: frameskip 5 x a 2-D action).
        """
        d1 = torch.randn(b, 1, d_a, device=device, dtype=dtype)
        d1 = d1 / d1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        d2 = torch.randn(b, 1, d_a, device=device, dtype=dtype)
        d2 = d2 - (d2 * d1).sum(-1, keepdim=True) * d1          # remove the d1 component
        d2 = d2 / d2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return d1, d2

    def action_isometry_loss(self, z, act, z_pred_first=None):
        """SCALE-FREE conditioning penalty on the action->latent Jacobian B = dz_{t+1}/da_t.

        MOTIVATION (from the paper's OWN bound, appendix Eq. app_kappaA / main Eq. kappaA).
        With A = dz_{t+1}/dz_t, B = dz_{t+1}/da_t and W_K the finite-horizon controllability
        Gramian, the conditioning of the planning Hessian obeys

            kappa_eff(H) = kappa(W_K) <= kappa(B)^2 * kappa(A)^(2(H-1))
                        <= kappa(B)^2 * ((1+eps)/(1-eps))^(2(H-1)),   eps = ||A - I||_2

        Straightening shrinks eps, i.e. it attacks the SECOND factor only. `kappa(B)^2` is a
        prefactor that straightening cannot touch and that is HORIZON-INDEPENDENT -- it hurts
        planning at H=5 exactly as much as at H=50. Once eps is small, kappa_eff is floored at
        kappa(B)^2 and further pressure on A buys nothing (which is what the multi-scale ties
        looked like). This term attacks that floor directly by pushing B towards an isometry,
        kappa(B) -> 1, so equal-size action perturbations move the latent by equal amounts
        whatever direction they point in.

        SCALE-FREE BY CONSTRUCTION -- this is the lesson from the rollout-consistency failure.
        That term was satisfiable by SHRINKING the latent, and the model took the shortcut
        (encoder temporal contrast collapsed ~21%, planning success fell 19-27 points). So this
        loss constrains only the RATIO of B's singular values, never their magnitude:

            G     = B^T B                       (d_a x d_a, symmetric PSD)
            c     = trace(G)/d_a                (mean squared response; DETACHED)
            L_iso = || G/c - I ||_F^2

        B -> s*B leaves L_iso exactly unchanged, so collapse is not a solution.

        ESTIMATOR (no explicit Jacobian). With orthonormal probes d1, d2 and
        B*d = [f(z, a + eta*d) - f(z, a)] / eta,

            g11 = <Bd1,Bd1>,  g22 = <Bd2,Bd2>,  g12 = <Bd1,Bd2>
            c   = (g11 + g22)/2                                      (detached)
            L   = [ (g11 - g22)^2 / 2  +  2*g12^2 ] / c^2

        i.e. penalise unequal response along the two directions plus any cross-coupling. Note
        `eta` CANCELS: every g scales as 1/eta^2, so numerator and c^2 both scale as 1/eta^4.
        Only the finite-difference nonlinearity depends on eta, not the loss scale.

        COST: exactly TWO extra PREDICTOR calls (f(z,a) is reused from the base forward). The
        predictor is tiny here (emb_dim 28, ~0.7M params) and -- unlike the rollout loss -- the
        calls are INDEPENDENT, not composed, so no chained autograd graph.

        WINDOW: uses only frames 0..num_hist-1 and the action at num_hist-1. NO window widening,
        so the data pipeline, iterations/epoch and the straightening window stay bit-identical to
        the paper. This makes it the first extension we can compare to the baseline with the loss
        term as the ONLY difference.

        Returns (loss_or_None, logs). `iso_response_c` in the logs is the mean squared latent
        response to a unit action perturbation: the COLLAPSE GUARD. If it falls sharply the model
        is shrinking B, which is the rollout failure mode; watch it every run.
        """
        if self.iso_lambda <= 0:
            return None, {}
        nh = self.num_hist
        ad = self.action_dim
        d_a = act.shape[-1]                     # RAW action dim (10 for PushT: frameskip 5 x 2-D)

        hist = z[:, :nh]
        a_t = act[:, nh - 1 : nh]               # the action that drives the predicted frame

        base = z_pred_first if z_pred_first is not None else self._predict_maybe_ckpt(
            hist, use_ckpt=self.iso_checkpoint
        )
        # Finite differences in fp32: under bf16 autocast a difference of two O(1) tensors keeps
        # only ~3 significant digits, which would swamp the O(eta) signal.
        f0 = base[:, -1:, ..., : -ad].float()

        d1, d2 = self._orthonormal_action_probes(z.shape[0], d_a, z.device, a_t.dtype)
        responses = []
        for d in (d1, d2):
            frame_p = self._inject_action(hist[:, -1:], a_t + self.iso_eps * d)
            hist_p = torch.cat([hist[:, :-1], frame_p], dim=1)
            f_p = self._predict_maybe_ckpt(hist_p, use_ckpt=self.iso_checkpoint)
            responses.append((f_p[:, -1:, ..., : -ad].float() - f0).flatten(1))

        Bd1, Bd2 = responses
        g11 = (Bd1 * Bd1).sum(-1)
        g22 = (Bd2 * Bd2).sum(-1)
        g12 = (Bd1 * Bd2).sum(-1)
        # c must stay DIFFERENTIABLE. With c detached the forward value is still invariant to
        # B -> a*B, but the gradient is NOT: writing N for the numerator, N is homogeneous of
        # degree 4 in B, so B . grad(N/c^2) = 4N/c^2 = 4L > 0 whenever L > 0. Descent then has a
        # strictly inward radial component and the term IS satisfiable by shrinking B -- exactly
        # the collapse shortcut the docstring above claims it forbids. Keeping c in the graph
        # makes L homogeneous of degree 0, so by Euler's identity B . grad(L) = 0 exactly and the
        # radial direction carries no gradient at all. (The earlier unit test compared only
        # forward VALUES under rescaling, which is why it passed.)
        c = (0.5 * (g11 + g22)).clamp_min(1e-12)
        loss = (0.5 * (g11 - g22).pow(2) + 2.0 * g12.pow(2)) / c.pow(2)
        loss = loss.mean()
        logs = {
            "iso_loss_raw": loss.detach(),
            # collapse guard: mean squared response, converted back to a per-unit-perturbation
            # figure so it is comparable across runs and across iso_eps settings.
            "iso_response_c": (c.detach() / (self.iso_eps ** 2)).mean(),
        }
        return loss, logs


    def rollout_consistency_loss(self, z, act, z_pred_first=None):
        """Multi-step rollout-consistency loss.

        Mirrors the PLANNING-time procedure (`rollout`): start from the real history
        z[:, :num_hist], predict, take the newest predicted frame, inject the real action,
        slide the window, repeat. The k-th prediction is compared against the real latent.

        k=1 is ALREADY covered by the standard one-step `z_loss` in forward(), so only
        k>=2 contribute to the loss; k=1 is still measured and returned for diagnostics.

        `z_pred_first` is the base forward's `predict(z[:, :num_hist])`. The rollout's k=1 step
        has EXACTLY that input, so passing it in removes one fully duplicated predictor call
        (compute + memory) with no change to the result.

        Returns (loss_or_None, logs) where logs holds per-k errors (detached) so the
        error-vs-k curve -- i.e. whether error is amplified -- is directly observable.
        """
        T = z.shape[1]
        nh = self.num_hist
        # Cap K by what the window actually holds: predicting frame nh+k-1 needs T > nh+k-1.
        max_k = min(self.rollout_steps, T - nh)
        logs = {}
        if max_k < 1:
            return None, logs

        # Optional sub-batch for memory (unbiased: a random subset's mean estimates the
        # full-batch mean). Must subsample z, act AND the reused z_pred_first TOGETHER so
        # frames, actions and the cached k=1 prediction stay row-aligned.
        if self.rollout_batch_frac < 1.0:
            b_full = z.shape[0]
            b_sub = max(1, int(round(b_full * self.rollout_batch_frac)))
            if b_sub < b_full:
                idx = torch.randperm(b_full, device=z.device)[:b_sub]
                z = z[idx]
                act = act[idx]
                if z_pred_first is not None:
                    z_pred_first = z_pred_first[idx]

        hist = z[:, :nh]                       # real frames 0 .. nh-1 (teacher-forced start)
        total = None
        for k in range(1, max_k + 1):
            if k == 1 and z_pred_first is not None:
                pred = z_pred_first            # reuse the base forward's identical call
            else:
                pred = self._predict_maybe_ckpt(hist)   # (b, nh, p, d)
            z_next = pred[:, -1:, ...]         # prediction of frame index j
            j = nh + k - 1                     # absolute index of the predicted frame

            tgt = z[:, j : j + 1]
            tgt = tgt.detach() if self.stop_grad else tgt
            # Compare visual+proprio only (exclude action channels), matching z_loss.
            err = self.emb_criterion(
                z_next[..., : -self.action_dim], tgt[..., : -self.action_dim]
            )
            logs[f"rollout_err_k{k}"] = err.detach()
            if k >= 2:                         # k=1 is already in z_loss; don't double-count
                w = self.rollout_gamma ** (k - 1)
                total = (w * err) if total is None else (total + w * err)

            if k < max_k:                      # prepare the next step of the rollout
                z_next = self._inject_action(z_next, act[:, j : j + 1])
                hist = torch.cat([hist[:, 1:], z_next], dim=1)   # slide, keep exactly nh frames
        return total, logs

    def total_curvature(self, features, mode="cos"):
        if features.shape[1] < 3:
            raise ValueError(f"Features must have at least 3 frames for curvature calculation, got {features.shape[1]}")

        # Build the per-frame representation z on which curvature is measured.
        if mode == "aggcos":
            if not hasattr(self.encoder, "agg"):
                raise ValueError("curvature mode 'aggcos' requires encoder.agg().")
            b, t, p, d = features.shape
            tokens = features.reshape(b * t, p, d)
            z = self.encoder.agg(tokens).reshape(b, t, -1)   # (b, T, dim) global feature
        elif mode == "cos":
            z = features                                     # (b, T, p, d) per-patch
        else:
            raise ValueError(f"Unknown curvature mode '{mode}'. Use 'cos' or 'aggcos'.")

        # ---- Multi-scale curvature: weighted sum over scales (scales==[1] == paper) ----
        #   L_multi = sum_s  w_s * L_curv^(s)
        total = None
        used = []
        # Per-scale RAW curvature values (before the lambda weighting), stashed for logging.
        # Diagnostic for the redundancy question: if a coarse scale's L_curv^(s) is already
        # ~0 while L_curv^(1) is clearly positive, that scale is implied by the fine scale
        # (telescoping: v^(s) is a sum of s consecutive fine velocities) and contributes
        # almost no gradient regardless of its lambda.
        self._last_scale_curvatures = {}
        speed_total = None      # arc-length term, summed over active scales (see below)
        for s, w in zip(self.straighten_scales, self.straighten_scale_weights):
            if w == 0:
                continue  # lambda=0 -> scale disabled (no loss term, no window requirement)
            c = self._scale_velocity_curvature(z, s)
            if c is None:
                continue  # this scale doesn't fit the current window; skip it
            used.append(s)
            self._last_scale_curvatures[f"curv_s{s}"] = c.detach()
            total = (w * c) if total is None else (total + w * c)

            # ---- Arc-length (constant-speed) consistency at the same scale ----------------
            # ALWAYS measured (free: same velocities, no extra forward pass) so that even
            # paper-faithful baseline runs log the speed dispersion. Accumulated SEPARATELY and
            # added in forward() with an ABSOLUTE lambda -- folding it into `total` here would
            # silently multiply it by the legacy global scale (aggcos1e-1 -> 0.1).
            sp = self._scale_speed_consistency(z, s)
            if sp is not None:
                self._last_scale_curvatures[f"speed_s{s}"] = sp.detach()
                speed_total = sp if speed_total is None else (speed_total + sp)
        self._last_speed_term = speed_total
        if total is None:
            raise ValueError(
                f"No straightening scale fit the window (T={features.shape[1]}, "
                f"scales={self.straighten_scales}); each scale s needs T >= 2s+1 frames."
            )

        # ---- Optional directional (goal) term; pseudo-goal = last latent in the window ----
        if self.straighten_goal_weight > 0:
            z_goal = z[:, -1]
            g_total = None
            for s in self.straighten_scales:
                g = self._scale_goal_alignment(z, s, z_goal)
                if g is None:
                    continue
                g_total = g if g_total is None else (g_total + g)
            if g_total is not None:
                total = total + self.straighten_goal_weight * g_total

        return total

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        decoder_enabled = self.decoder is not None and self.train_decoder
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        # Cap the prediction target to num_hist frames. When the window T is longer than
        # num_hist+num_pred (needed for multi-scale straightening) this keeps the
        # prediction loss unchanged; when T == num_hist+num_pred it is identical to the
        # original z[:, num_pred:] slice. The full-length z (all T frames) is still used
        # for the straightening term below.
        z_tgt = z[:, self.num_pred : self.num_pred + self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred : self.num_pred + self.num_hist, ...]  # (b, num_hist, 3, ...)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if decoder_enabled:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                    recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
            z_tgt_for_loss = z_tgt.detach() if self.stop_grad else z_tgt
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt_for_loss[:, :, :-2, :])
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt_for_loss[:, :, -2, :])
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt_for_loss[:, :, :-1, :])
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt_for_loss[:, :, :, :-(self.proprio_dim + self.action_dim)]
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim], 
                    z_tgt_for_loss[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim]
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim], 
                    z_tgt_for_loss[:, :, :, :-self.action_dim]
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss

            if self.vcreg:
                z_vic_in = self.visual_prop(z)
                z_std_loss = self.vcreg_std_loss(z_vic_in)
                z_cov_loss = self.vcreg_cov_loss(z_vic_in)
                z_reg_loss = z_std_loss * self.std_coeff + z_cov_loss * self.cov_coeff
                loss_components["z_vicreg_std_loss"] = z_std_loss
                loss_components["z_vicreg_cov_loss"] = z_cov_loss
                loss_components["z_vcreg_loss_scaled"] = z_reg_loss
                loss = loss + z_reg_loss

            if self.straighten and self.straighten_scale > 0:
                feats = self.visual_only(z)
                curvature_loss = self.total_curvature(feats, mode=self.curvature_mode)
                loss = loss + curvature_loss * self.straighten_scale
                loss_components["curvature_loss_used_for_training"] = curvature_loss
                # ---- Arc-length (constant-speed) consistency; lambda=0 -> no-op (paper) ----
                # ABSOLUTE coefficient, applied outside the legacy global scale. The derivation
                # in _scale_speed_consistency fixes the paper-matched value at HALF the
                # curvature lambda (0.1 -> 0.05): both terms are halves of the single quantity
                # ||v_{t+1}-v_t||^2 / (||v_t|| ||v_{t+1}||) = (r+1/r-2) + 2(1-C).
                _speed = getattr(self, "_last_speed_term", None)
                if self.straighten_speed_lambda > 0 and _speed is not None:
                    _speed_scaled = self.straighten_speed_lambda * _speed.to(loss.dtype)
                    loss = loss + _speed_scaled
                    loss_components["speed_loss_scaled"] = _speed_scaled.detach()
                # Per-scale raw curvatures (logging only; detached, does not affect the loss).
                for _k, _v in getattr(self, "_last_scale_curvatures", {}).items():
                    loss_components[_k] = _v

            # ---- Multi-step rollout consistency (k>=2). rollout_steps=1 -> no-op (paper). ----
            # Always computed when rollout_steps>1 so the per-k error curve is logged even if
            # the added loss is None (e.g. window too short); k=1 error is diagnostic only.
            if self.rollout_steps > 1:
                # Hand over the already-computed one-step prediction: the rollout's k=1 step
                # has the identical input z[:, :num_hist], so this skips a duplicate call.
                roll_loss, roll_logs = self.rollout_consistency_loss(
                    z, act, z_pred_first=z_pred
                )
                for _k, _v in roll_logs.items():
                    loss_components[_k] = _v
                if roll_loss is not None:
                    loss = loss + roll_loss
                    loss_components["rollout_loss_used_for_training"] = roll_loss

            # ---- Action-isometry conditioning (k(B) -> 1). iso_lambda=0 -> no-op (paper). ----
            # Reuses the base one-step prediction as the finite-difference reference point, so
            # the term costs exactly two extra predictor calls.
            iso_loss, iso_logs = self.action_isometry_loss(z, act, z_pred_first=z_pred)
            for _k, _v in iso_logs.items():
                loss_components[_k] = _v
            if iso_loss is not None:
                loss = loss + self.iso_lambda * iso_loss
                loss_components["iso_loss_scaled"] = (self.iso_lambda * iso_loss).detach()
        else:
            visual_pred = None
            z_pred = None

        if decoder_enabled:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z