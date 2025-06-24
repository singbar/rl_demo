# This function converts NumPy types into native Python types to make them serializable by Temporal.
# Temporal (and many other systems) cannot serialize NumPy objects like ndarray or float32 directly.
# This utility ensures the data passed to workflows/activities is JSON-serializable and compatible.

# Imports
import numpy as np

def to_builtin(x):
    """
    Recursively convert NumPy types to native Python types:
    - np.ndarray → list or scalar
    - np.generic (e.g., np.float32, np.int64) → Python float or int
    - all other types are returned unchanged
    """

    # Case 1: Convert NumPy arrays
    if isinstance(x, np.ndarray):
        if x.size == 1:
            # If it's a single-element array (e.g., np.array([42.0])), convert to scalar
            return x.item()
        # Otherwise, convert entire array to a list of lists or flat list
        return x.tolist()

    # Case 2: Convert NumPy scalar values (like np.float32, np.int64) to Python equivalents
    elif isinstance(x, np.generic):
        return x.item()  # Converts to float, int, etc.

    # Case 3: Already a built-in Python type — return unchanged
    else:
        return x
