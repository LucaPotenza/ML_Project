import numpy as np

# ============================================================================
# REPRODUCIBILITY / GLOBAL UTILS
# ============================================================================

def set_seed(seed: int = 0):
    """Make runs reproducible (as much as possible)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic mode can slow down training, but helps debugging
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)