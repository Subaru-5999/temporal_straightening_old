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
OmegaConf.register_new_resolver("run_variant_tag", run_variant_tag)

if __name__ == "__main__":
    register_resolvers()

