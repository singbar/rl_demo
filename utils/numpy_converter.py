# Import NumPy for handling arrays and numerical types
import numpy as np

# This recursive function converts NumPy-specific types (like float32, int64, or ndarray)
# into built-in Python types (like float, int, or list). This is useful when you need to 
# serialize data (e.g., to JSON) or pass it to systems that expect native Python types.
def convert_numpy(obj):
    # If the input is a dictionary (e.g., {"key": np.float32(5.0)}),
    # recursively convert each key-value pair
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    
    # If the input is a list (e.g., [np.array([1, 2, 3])]),
    # recursively convert each element in the list
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    
    # If the input is a tuple (e.g., (np.float64(1.0), np.int64(2))),
    # recursively convert each element and return a new tuple
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    
    # If the input is a NumPy scalar (like np.float32, np.int64, etc.),
    # call .item() to convert it into a native Python scalar (float or int)
    elif isinstance(obj, np.generic):
        return obj.item()
    
    # If the input is a full NumPy array (np.ndarray),
    # call .tolist() to convert it into a nested list of Python numbers
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # If the input is already a built-in type (int, float, str, etc.),
    # return it unchanged
    else:
        return obj
