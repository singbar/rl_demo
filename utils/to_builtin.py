import numpy as mp

def to_builtin(x):
    """Convert NumPy types to plain Python types for Temporal serialization."""
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.tolist()
    elif isinstance(x, np.generic):
        return x.item()
    else:
        return x