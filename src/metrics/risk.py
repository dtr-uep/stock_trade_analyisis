import numpy as np

# QLike
def qlike(r_true, var_pred):
    return np.mean(((r_true**2 + 1e-12) / var_pred) - np.log((r_true**2 + 1e-12) /var_pred) - 1) 