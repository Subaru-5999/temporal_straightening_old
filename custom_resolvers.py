import hydra
from omegaconf import OmegaConf

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')

def replace_substring(value: str, old: str, new: str) -> str:
    return str(value).replace(str(old), str(new))


def _to_list(v):
    """Normalize an OmegaConf value (ListConfig / None / 'null' / list) to a python list or None."""
    if v is None:
        return None
    if isinstance(v, str):
        return None if v.strip().lower() in ("none", "null", "") else [v]
    try:
        return list(v)
    except TypeError:
        return None


def _fmt_num(x):
    # 0.1 -> "0.1", 0.0 -> "0", 1.0 -> "1", 0.2 -> "0.2"  (filesystem-safe, compact)
    return ("%g" % float(x))


def _scalar_or_none(v):
    """Normalize an OmegaConf scalar (None / 'null' / 'none' / number) to a value or None."""
    if v is None:
        return None
    if isinstance(v, str):
        return None if v.strip().lower() in ("none", "null", "") else v
    return v


def straighten_tag(scales=None, lambdas=None, weights=None) -> str:
    """Build a filesystem-safe suffix that encodes the multi-scale straightening settings,
    so every distinct setting gets its OWN checkpoint folder (no collisions, self-documenting,
    and directly reusable as the Hugging Face sub-path).

    Returns '' (empty) for the paper single-scale default -- scales null or [1] with no custom
    lambdas/weights -- so paper-faithful runs keep their exact original folder name.

    Examples:
      scales=[1,4], lambdas=[0.1,0.2] -> "_ms1-4_lam0.1-0.2"
      scales=[1,4], lambdas=[0.1,0]   -> "_ms1-4_lam0.1-0"     (coarse disabled)
      scales=[1,4], weights=[1,2]     -> "_ms1-4_w1-2"         (legacy weights)
    """
    scales = _to_list(scales)
    lambdas = _to_list(lambdas)
    weights = _to_list(weights)
    if not scales:
        scales = None

    is_default_single = (scales is None) or ([int(s) for s in scales] == [1])
    if is_default_single and lambdas is None and weights is None:
        return ""

    parts = []
    if scales is not None:
        parts.append("ms" + "-".join(str(int(s)) for s in scales))
    if lambdas is not None:
        parts.append("lam" + "-".join(_fmt_num(x) for x in lambdas))
    elif weights is not None:
        parts.append("w" + "-".join(_fmt_num(x) for x in weights))
    return ("_" + "_".join(parts)) if parts else ""


def rollout_tag(rollout_steps=1, rollout_gamma=0.9) -> str:
    """Suffix encoding the multi-step rollout-consistency setting, so those runs get their OWN
    checkpoint folder. Returns '' for rollout_steps<=1 (the paper default), keeping
    paper-faithful run names byte-identical.

    Example: (rollout_steps=4, rollout_gamma=0.9) -> "_roll4g0.9"
    """
    k = _scalar_or_none(rollout_steps)
    k = int(float(k)) if k is not None else 1
    if k <= 1:
        return ""
    g = _scalar_or_none(rollout_gamma)
    g = float(g) if g is not None else 0.9
    return f"_roll{k}g{_fmt_num(g)}"


def iso_tag(iso_lambda=0.0) -> str:
    """Suffix encoding the action-isometry conditioning setting, so those runs get their OWN
    checkpoint folder. Returns '' for iso_lambda<=0 (the paper default), keeping paper-faithful
    run names byte-identical.

    Example: iso_lambda=0.01 -> "_iso0.01"
    """
    l = _scalar_or_none(iso_lambda)
    l = float(l) if l is not None else 0.0
    if l <= 0:
        return ""
    return f"_iso{_fmt_num(l)}"


def arc_tag(straighten_speed_lambda=0.0) -> str:
    """Suffix encoding the arc-length (constant-speed) consistency setting, so those runs get
    their OWN checkpoint folder. Returns '' for lambda<=0 (the paper default), keeping
    paper-faithful run names byte-identical.

    Example: straighten_speed_lambda=0.05 -> "_arc0.05"
    """
    l = _scalar_or_none(straighten_speed_lambda)
    l = float(l) if l is not None else 0.0
    if l <= 0:
        return ""
    return f"_arc{_fmt_num(l)}"


def initmode_tag(debug_dset_init=False) -> str:
    """Suffix marking a run whose planner was initialised at the DATASET's ground-truth actions
    (`debug_dset_init=true`). Returns '' otherwise, so normal runs keep byte-identical paths.

    Necessary because the plan output dir encodes `init${planner.sub_planner.sample_type}`, which
    stays 'zero' regardless of this flag -- so a gt-initialised run and a zero-initialised run
    with the same opt_steps landed in the SAME folder and APPENDED to the same logs.json,
    silently interleaving two different experiments.

    gt_actions is privileged information: this tag exists so diagnostic runs can never be
    mistaken for, or mixed into, reportable numbers.
    """
    v = _scalar_or_none(debug_dset_init)
    if isinstance(v, str):
        v = v.strip().lower() in ("true", "1", "yes")
    return "_gtinit" if bool(v) else ""


def corridor_tag(corridor_beta=0.0, corridor_rho=0.0) -> str:
    """Suffix encoding the latent-geodesic-corridor planning objective, so corridor evals land in
    their OWN plan_outputs folder and can never be mixed with the paper-faithful numbers.
    Returns '' for beta<=0 (the paper default), keeping baseline output paths byte-identical.

    Example: (beta=0.5, rho=0.2) -> "_cor0.5r0.2";  (beta=0.5, rho=0) -> "_cor0.5"
    """
    b = _scalar_or_none(corridor_beta)
    b = float(b) if b is not None else 0.0
    if b <= 0:
        return ""
    r = _scalar_or_none(corridor_rho)
    r = float(r) if r is not None else 0.0
    return f"_cor{_fmt_num(b)}" + (f"r{_fmt_num(r)}" if r > 0 else "")


def run_variant_tag(epochs, seed=0) -> str:
    """Suffix encoding training length (and non-zero training seed) so runs at different
    epoch counts / seeds land in their OWN folders (planning outputs inherit it via model_name).

    Convention (kept consistent with base_cell / aggregate_trainseeds):
      - always append _ep<N>  (so 2-epoch and 3-epoch runs never collide),
      - append _seed<N> only when seed != 0  (seed 0 == the base cell, no suffix, so the
        Table-1 master table and paper-target lookup keep matching the canonical cell names).

    Examples: (epochs=2, seed=0) -> "_ep2";  (epochs=3, seed=1) -> "_ep3_seed1".
    """
    tag = f"_ep{int(epochs)}"
    if int(seed) != 0:
        tag += f"_seed{int(seed)}"
    return tag


# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)
OmegaConf.register_new_resolver("replace_substring", replace_substring)
OmegaConf.register_new_resolver("straighten_tag", straighten_tag)
OmegaConf.register_new_resolver("rollout_tag", rollout_tag)
OmegaConf.register_new_resolver("iso_tag", iso_tag)
OmegaConf.register_new_resolver("arc_tag", arc_tag)
OmegaConf.register_new_resolver("corridor_tag", corridor_tag)
OmegaConf.register_new_resolver("initmode_tag", initmode_tag)
OmegaConf.register_new_resolver("run_variant_tag", run_variant_tag)

if __name__ == "__main__":
    register_resolvers()

