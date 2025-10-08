import numpy as np

def f0_rmse(f0_pred, f0_ref):
    f0_pred = np.asarray(f0_pred)
    f0_ref  = np.asarray(f0_ref)
    mask = (f0_ref > 1e-6) & (f0_pred > 1e-6)
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((1200*np.log2(f0_pred[mask]/f0_ref[mask]))**2))
