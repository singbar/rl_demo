#This function converts NumPy types to plain Python types for Temporal serialization

#Imports
import numpy as np

def to_builtin(x):

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return x.item()
        return x.tolist()
    elif isinstance(x, np.generic):
        return x.item()
    else:
        return x