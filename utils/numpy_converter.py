import numpy as np


def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    elif isinstance(obj, np.generic):
        return obj.item()  # Handles float32, int64, etc.
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj