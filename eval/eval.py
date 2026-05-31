"""
Simple evaluation utilities for SemEval subtasks.

Provides:
- task1_correlation(...) : within-person and between-person Pearson r for subtask1
  (modes: "within", "between", "all", "composite")
- task2_correlation(...) : between-person Pearson r for subtasks 2a/2b (or direct pearson if user_ids is None)

Notes
- Inputs may be Python lists or numpy arrays.
- Pearson r is computed using scipy.stats. If an input is constant (no variance) r will be returned as float("nan").
- Functions return dicts; keys/payload vary by function and requested mode.
"""
from typing import Sequence, Optional, Dict, Tuple, Any, List
import numpy as np
from scipy.stats import pearsonr

def _pearson(x: Sequence[float], y: Sequence[float]) -> Tuple[float, Optional[float]]:
    """
    Compute Pearson r. If inputs have <2 non-NaN samples or are constant,
    returns (nan, None).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.size < 2:
        return float("nan"), None
    # pearsonr raises ValueError on constant input; catch and return nan
    r, p = pearsonr(x_arr, y_arr)
    return float(r), float(p)


def _mae(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Compute Mean Absolute Error.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    return float(np.nanmean(np.abs(x_arr - y_arr)))

def task1_correlation(
    user_ids: Sequence[Any],
    text_ids: Sequence[Any],
    predictions: Sequence[float],
    labels: Sequence[float],
) -> Dict[str, Any]:
    """
    Compute three Pearson correlation metrics for Subtask 1 (always all are returned).

    Metrics returned (flat dict):
      - r_within: average of per-user Pearson r (nan if not computable)
      - p_within: harmonic-mean combined p-value of per-user tests (None if no p-values)
      - r_between: Pearson r computed across users after averaging predictions and labels per user
      - p_between: p-value for r_between (None if not computable)
      - r_composite: composite correlation via Fisher's z (nan if not computable)
      - mae_within: average of per-user MAE
      - mae_between: MAE computed across users after averaging predictions and labels per user
      - mae_composite: composite MAE via Fisher's z
    """
    # convert to arrays
    user_arr = np.asarray(user_ids)
    preds = np.asarray(predictions, dtype=float)
    labs = np.asarray(labels, dtype=float)

    # --- within-person: per-user pearson, then average (ignores users with <2 samples) ---
    unique_users = np.unique(user_arr)
    per_user_r: Dict[str, float] = {}
    per_user_p: Dict[str, Optional[float]] = {}
    r_vals: List[float] = []
    p_vals: List[float] = []

    for u in unique_users:
        mask = user_arr == u
        if np.sum(mask) < 2:
            continue
        if np.var(labs[mask]) == 0.0:
            continue
        if np.var(preds[mask]) == 0.0:
            per_user_r[str(u)] = 0.0
            per_user_p[str(u)] = 1e-10
            r_vals.append(0.0)
            p_vals.append(1e-10)
            continue
        
        r, p = _pearson(preds[mask], labs[mask])
        per_user_r[str(u)] = r
        per_user_p[str(u)] = p
        r_vals.append(r)
        p_vals.append(float(p))

    r_within = float(np.mean(r_vals))
    p_within = float(len(p_vals) / sum(1.0 / max(pv, 1e-10) for pv in p_vals))

    # --- between-person: average per-user then pearson across users ---
    user_means_pred: List[float] = []
    user_means_lab: List[float] = []
    for u in unique_users:
        mask = user_arr == u
        pred_mean = np.nanmean(preds[mask])
        lab_mean = np.nanmean(labs[mask])
        user_means_pred.append(pred_mean)
        user_means_lab.append(lab_mean)

    n_between = len(user_means_pred)
    
    r_between, p_between = _pearson(user_means_pred, user_means_lab)
    
    ### Compute the same for MAE
    mae_within = []
    for u in unique_users:
        mask = user_arr == u
        user_mae = _mae(preds[mask], labs[mask])
        mae_within.append(user_mae)

    mae_within = float(np.nanmean(mae_within))
    mae_between = _mae(user_means_pred, user_means_lab)

    # --- composite via Fisher's z (atanh average -> tanh) ---
    def valid_for_arctanh(x: float) -> bool:
        return not np.isnan(x) and abs(x) < 1.0

    z_w = np.arctanh(r_within)
    z_b = np.arctanh(r_between)
    z_avg = 0.5 * (z_w + z_b)
    r_composite = float(np.tanh(z_avg))
    
    mae_composite = 0.5 * (np.arctanh(mae_within) + np.arctanh(mae_between))
    mae_composite = float(np.tanh(mae_composite))

    # assemble flat dict and return
    return {
        "r_within": r_within,
        "p_within": p_within,
        "r_between": r_between,
        "p_between": p_between,
        "r_composite": r_composite,
        "mae_within": mae_within,
        "mae_between": mae_between,
        "mae_composite": mae_composite
    }


def task2_correlation(
    user_ids: Optional[Sequence[Any]],
    predictions: Sequence[float],
    labels: Sequence[float],
) -> Dict[str, Any]:
    """
    Compute Pearson correlation for Subtask 2a/2b.

    TODO: user_ids might be used for additional sub-metrics.
    
    Returns a dict: {"r": float, "p": float, "mae": float (optional)}
    """

    r, p = _pearson(predictions, labels)
    mae = _mae(predictions, labels)

    return {"r": r, "p": p, "mae": mae}
