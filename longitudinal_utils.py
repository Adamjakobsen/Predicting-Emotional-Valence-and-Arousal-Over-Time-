from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

TARGETS = ["valence", "arousal"]
REQUIRED_COLUMNS = [
    "user_id", "text_id", "text", "timestamp", "collection_phase",
    "is_words", "valence", "arousal",
]

# ---------------------------------------------------------------------------
# Basic preparation
# ---------------------------------------------------------------------------
_CONTRACTION_PATTERNS = [
    (re.compile(r"\s+'\s*([a-zA-Z]{1,3})\b"), r"'\1"),
    (re.compile(r"\bn\s*'\s*t\b"), r"n't"),
]
_PUNCT_AFTER = re.compile(r"\s+([,.;:!?\)\]])")
_PUNCT_BEFORE = re.compile(r"([\(\[])\s+")
_LITERAL_NEWLINE = re.compile(r"\\\s*n")
_MULTI_SPACE = re.compile(r"\s{2,}")


def detokenize(text: str) -> str:
    """Undo the spaced-punctuation tokenization used in the provided CSV files."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = _LITERAL_NEWLINE.sub(" ", text)
    for pattern, replacement in _CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    text = _PUNCT_AFTER.sub(r"\1", text)
    text = _PUNCT_BEFORE.sub(r"\1", text)
    return _MULTI_SPACE.sub(" ", text).strip()


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Fail loudly when the expected task schema is not present."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Validate and add clean text plus parsed timestamp columns."""
    require_columns(df, REQUIRED_COLUMNS, name)
    out = df.copy()
    out["text_clean"] = out["text"].map(detokenize)
    out["timestamp_parsed"] = pd.to_datetime(out["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    out["is_words_int"] = out["is_words"].astype(bool).astype(int)
    out["collection_phase"] = out["collection_phase"].astype(int)
    out["_row_id"] = np.arange(len(out), dtype=int)
    return out


def chronological_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical ordering used for all history construction."""
    return df.sort_values(["user_id", "timestamp_parsed", "text_id", "_row_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation split and official-style metrics
# ---------------------------------------------------------------------------
def build_val_split(
    df: pd.DataFrame,
    seed: int = 13,
    unseen_frac: float = 0.20,
    within_user_holdout: float = 0.25,
) -> pd.Series:
    """Create a validation split with both unseen users and within-user holdouts."""
    rng = np.random.default_rng(seed)
    users = df["user_id"].unique().copy()
    rng.shuffle(users)
    n_unseen = int(round(len(users) * unseen_frac))
    unseen_users = set(users[:n_unseen])

    is_val = pd.Series(False, index=df.index)
    is_val |= df["user_id"].isin(unseen_users)

    has_timestamps = "timestamp_parsed" in df.columns

    for user_id, sub in df.groupby("user_id"):
        if user_id in unseen_users or len(sub) < 4:
            continue
        n_hold = max(2, int(round(len(sub) * within_user_holdout)))
        if has_timestamps:
            # Hold out the chronologically last n_hold entries per user. This
            # mirrors the real test-time scenario where all test entries are
            # genuinely later than every training entry, so longitudinal history
            # features are always built from truly prior observations.
            chosen = sub.sort_values("timestamp_parsed").index[-n_hold:].tolist()
        else:
            essays = sub.index[~sub["is_words"].astype(bool)].tolist()
            words = sub.index[sub["is_words"].astype(bool)].tolist()
            rng.shuffle(essays)
            rng.shuffle(words)
            words_ratio = len(words) / len(sub)
            n_words = int(round(n_hold * words_ratio))
            n_essays = n_hold - n_words
            chosen = essays[:n_essays] + words[:n_words]
        is_val.loc[chosen] = True
    return is_val


def _one_dim_metrics(user_ids: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique_users = np.unique(user_ids)
    within_rs = []
    for user_id in unique_users:
        mask = user_ids == user_id
        if mask.sum() < 2:
            continue
        if np.var(labels[mask]) <= 1e-12:
            continue
        if np.var(predictions[mask]) <= 1e-12:
            within_rs.append(0.0)
            continue
        within_rs.append(float(pearsonr(predictions[mask], labels[mask]).statistic))

    r_within = float(np.mean(within_rs)) if within_rs else float("nan")
    pred_means = np.array([predictions[user_ids == u].mean() for u in unique_users], dtype=float)
    label_means = np.array([labels[user_ids == u].mean() for u in unique_users], dtype=float)

    if np.var(pred_means) <= 1e-12 or np.var(label_means) <= 1e-12:
        r_between = float("nan")
    else:
        r_between = float(pearsonr(pred_means, label_means).statistic)

    with np.errstate(invalid="ignore"):
        z = 0.5 * (
            np.arctanh(np.clip(r_within, -0.999999, 0.999999))
            + np.arctanh(np.clip(r_between, -0.999999, 0.999999))
        )
        r_composite = float(np.tanh(z))

    mae_within = float(np.mean(np.abs(predictions - labels)))
    mae_between = float(np.mean(np.abs(pred_means - label_means)))
    return {
        "r_within": r_within,
        "r_between": r_between,
        "r_composite": r_composite,
        "mae_within": mae_within,
        "mae_between": mae_between,
    }


def evaluate(df: pd.DataFrame, pred_valence: np.ndarray, pred_arousal: np.ndarray) -> dict[str, float]:
    require_columns(df, ["user_id", "valence", "arousal"], "evaluation frame")
    user_ids = df["user_id"].to_numpy()
    preds = {"valence": np.asarray(pred_valence, dtype=float), "arousal": np.asarray(pred_arousal, dtype=float)}
    out: dict[str, float] = {}
    for target in TARGETS:
        m = _one_dim_metrics(user_ids, preds[target], df[target].to_numpy(dtype=float))
        for key, value in m.items():
            out[f"{target}_{key}"] = value
    out["mean_r_composite"] = 0.5 * (out["valence_r_composite"] + out["arousal_r_composite"])
    return out


def evaluate_seen_unseen(
    df: pd.DataFrame,
    pred_valence: np.ndarray,
    pred_arousal: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute metrics broken down by seen/unseen user status.

    Requires an ``is_seen_user`` boolean column in df (present in the test CSV).
    Always includes an ``"all"`` key; adds ``"seen"`` / ``"unseen"`` when the
    column exists. Useful for understanding where the longitudinal layer helps.
    """
    pred_valence = np.asarray(pred_valence, dtype=float)
    pred_arousal = np.asarray(pred_arousal, dtype=float)
    result: dict[str, dict[str, float]] = {"all": evaluate(df, pred_valence, pred_arousal)}
    if "is_seen_user" not in df.columns:
        return result
    seen_mask = df["is_seen_user"].astype(bool).to_numpy()
    for label, mask in [("seen", seen_mask), ("unseen", ~seen_mask)]:
        if mask.sum() >= 2:
            result[label] = evaluate(
                df[mask].reset_index(drop=True),
                pred_valence[mask],
                pred_arousal[mask],
            )
    return result


def metrics_table(metrics: dict[str, float]) -> pd.DataFrame:
    rows = []
    for dim in TARGETS:
        rows.append({
            "dimension": dim,
            "r_within": metrics[f"{dim}_r_within"],
            "r_between": metrics[f"{dim}_r_between"],
            "r_composite": metrics[f"{dim}_r_composite"],
            "mae_within": metrics[f"{dim}_mae_within"],
            "mae_between": metrics[f"{dim}_mae_between"],
        })
    rows.append({
        "dimension": "mean",
        "r_within": np.nan,
        "r_between": np.nan,
        "r_composite": metrics["mean_r_composite"],
        "mae_within": np.nan,
        "mae_between": np.nan,
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Longitudinal history features
# ---------------------------------------------------------------------------
def _empty_history_features(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_has_history": 0.0,
        f"{prefix}_n_prior": 0.0,
        f"{prefix}_days_since_prev": 0.0,
        f"{prefix}_days_since_first": 0.0,
        f"{prefix}_phase_gap": 0.0,
        f"{prefix}_prev_valence": 0.0,
        f"{prefix}_prev_arousal": 0.0,
        f"{prefix}_prev_delta_valence": 0.0,
        f"{prefix}_prev_delta_arousal": 0.0,
        f"{prefix}_mean_valence_all": 0.0,
        f"{prefix}_mean_arousal_all": 0.0,
        f"{prefix}_std_valence_all": 0.0,
        f"{prefix}_std_arousal_all": 0.0,
        f"{prefix}_mean_valence_last2": 0.0,
        f"{prefix}_mean_arousal_last2": 0.0,
        f"{prefix}_mean_valence_last3": 0.0,
        f"{prefix}_mean_arousal_last3": 0.0,
        f"{prefix}_mean_valence_last5": 0.0,
        f"{prefix}_mean_arousal_last5": 0.0,
        f"{prefix}_trend_valence_last5": 0.0,
        f"{prefix}_trend_arousal_last5": 0.0,
        f"{prefix}_has_trend": 0.0,
        f"{prefix}_n_words_prior": 0.0,
        f"{prefix}_n_essays_prior": 0.0,
    }


def _window_mean(values: np.ndarray, size: int) -> float:
    return float(values[-size:].mean())


def _last5_slope(times: list[pd.Timestamp], values: np.ndarray) -> float:
    take = min(5, len(values))
    t = np.array([(ts - times[-take]).total_seconds() / 86400.0 for ts in times[-take:]], dtype=float)
    y = values[-take:].astype(float)
    if np.var(t) <= 1e-12:
        return 0.0
    return float(np.polyfit(t, y, deg=1)[0])


def summarize_history(
    history: list[dict[str, object]],
    current_timestamp: pd.Timestamp,
    current_phase: int,
    prefix: str = "hist",
) -> dict[str, float]:
    """Summarize prior labelled or predicted states for one user."""
    if len(history) == 0:
        return _empty_history_features(prefix)

    val = np.array([float(h["valence"]) for h in history], dtype=float)
    aro = np.array([float(h["arousal"]) for h in history], dtype=float)
    times = [h["timestamp"] for h in history]
    phases = np.array([int(h["collection_phase"]) for h in history], dtype=int)
    is_words = np.array([int(h["is_words_int"]) for h in history], dtype=int)

    features = {
        f"{prefix}_has_history": 1.0,
        f"{prefix}_n_prior": float(len(history)),
        f"{prefix}_days_since_prev": float((current_timestamp - times[-1]).total_seconds() / 86400.0),
        f"{prefix}_days_since_first": float((current_timestamp - times[0]).total_seconds() / 86400.0),
        f"{prefix}_phase_gap": float(current_phase - phases[-1]),
        f"{prefix}_prev_valence": float(val[-1]),
        f"{prefix}_prev_arousal": float(aro[-1]),
        f"{prefix}_prev_delta_valence": float(val[-1] - val[-2]) if len(val) >= 2 else 0.0,
        f"{prefix}_prev_delta_arousal": float(aro[-1] - aro[-2]) if len(aro) >= 2 else 0.0,
        f"{prefix}_mean_valence_all": float(val.mean()),
        f"{prefix}_mean_arousal_all": float(aro.mean()),
        f"{prefix}_std_valence_all": float(val.std(ddof=0)),
        f"{prefix}_std_arousal_all": float(aro.std(ddof=0)),
        f"{prefix}_mean_valence_last2": _window_mean(val, 2),
        f"{prefix}_mean_arousal_last2": _window_mean(aro, 2),
        f"{prefix}_mean_valence_last3": _window_mean(val, 3),
        f"{prefix}_mean_arousal_last3": _window_mean(aro, 3),
        f"{prefix}_mean_valence_last5": _window_mean(val, 5),
        f"{prefix}_mean_arousal_last5": _window_mean(aro, 5),
        f"{prefix}_trend_valence_last5": _last5_slope(times, val) if len(history) >= 2 else 0.0,
        f"{prefix}_trend_arousal_last5": _last5_slope(times, aro) if len(history) >= 2 else 0.0,
        f"{prefix}_has_trend": 1.0 if len(history) >= 2 else 0.0,
        f"{prefix}_n_words_prior": float(is_words.sum()),
        f"{prefix}_n_essays_prior": float((1 - is_words).sum()),
    }
    return features


def current_time_features(row: pd.Series, global_start: pd.Timestamp) -> dict[str, float]:
    ts = row["timestamp_parsed"]
    hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    day_of_week = ts.dayofweek
    return {
        "is_words": float(row["is_words_int"]),
        "collection_phase": float(row["collection_phase"]),
        "hour_sin": math.sin(2 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2 * math.pi * hour / 24.0),
        "dow_sin": math.sin(2 * math.pi * day_of_week / 7.0),
        "dow_cos": math.cos(2 * math.pi * day_of_week / 7.0),
        "days_from_global_start": float((ts - global_start).total_seconds() / 86400.0),
    }


def append_history(history_by_user: dict[int, list[dict[str, object]]], row: pd.Series, valence: float, arousal: float) -> None:
    user_id = int(row["user_id"])
    history_by_user.setdefault(user_id, []).append({
        "timestamp": row["timestamp_parsed"],
        "collection_phase": int(row["collection_phase"]),
        "is_words_int": int(row["is_words_int"]),
        "valence": float(valence),
        "arousal": float(arousal),
    })


def make_one_feature_row(
    row: pd.Series,
    text_pred: np.ndarray,
    history_by_user: dict[int, list[dict[str, object]]],
    global_start: pd.Timestamp,
) -> dict[str, float]:
    features = current_time_features(row, global_start)
    features.update({
        "text_pred_valence": float(text_pred[0]),
        "text_pred_arousal": float(text_pred[1]),
        "text_pred_valence_x_has_history": float(text_pred[0])
            * (1.0 if len(history_by_user.get(int(row["user_id"]), [])) > 0 else 0.0),
        "text_pred_arousal_x_has_history": float(text_pred[1])
            * (1.0 if len(history_by_user.get(int(row["user_id"]), [])) > 0 else 0.0),
    })
    features.update(summarize_history(
        history_by_user.get(int(row["user_id"]), []),
        current_timestamp=row["timestamp_parsed"],
        current_phase=int(row["collection_phase"]),
    ))
    return features


def build_training_longitudinal_features(
    df: pd.DataFrame,
    text_predictions: np.ndarray,
    global_start: pd.Timestamp,
) -> pd.DataFrame:
    """Build features for labelled rows using only earlier rows from the same user.

    Exposure-bias note: history values here come from gold labels (train targets).
    At test time ``recursive_predict_targets`` instead feeds the model's own prior
    predictions as history — a train/test mismatch known as exposure bias.  The
    effect appears small empirically (test r_composite is higher than val despite
    this), but an oracle upper-bound can be measured by passing gold test labels
    as history instead of recursive predictions.
    """
    if len(df) != len(text_predictions):
        raise ValueError("df and text_predictions must have equal length")

    work = df.copy().reset_index(drop=True)
    work["text_pred_valence_input"] = text_predictions[:, 0]
    work["text_pred_arousal_input"] = text_predictions[:, 1]
    ordered = chronological_sort(work)

    history_by_user: dict[int, list[dict[str, object]]] = {}
    rows: list[dict[str, float]] = []
    row_ids: list[int] = []

    for _, row in ordered.iterrows():
        text_pred = np.array([row["text_pred_valence_input"], row["text_pred_arousal_input"]], dtype=float)
        rows.append(make_one_feature_row(row, text_pred, history_by_user, global_start))
        row_ids.append(int(row["_row_id"]))
        append_history(history_by_user, row, float(row["valence"]), float(row["arousal"]))

    features = pd.DataFrame(rows)
    features["_row_id"] = row_ids
    return features.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)


def recursive_predict_targets(
    known_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_text_predictions: np.ndarray,
    model,
    feature_columns: list[str],
    global_start: pd.Timestamp,
) -> np.ndarray:
    """
    Predict target rows chronologically.

    Known rows append gold states. Target rows append their own model predictions,
    so later target rows from the same user can use earlier predicted states.
    """
    if len(target_df) != len(target_text_predictions):
        raise ValueError("target_df and target_text_predictions must have equal length")

    known = known_df.copy().reset_index(drop=True)
    known["_event_type"] = "known"
    known["_target_pos"] = -1
    known["text_pred_valence_input"] = 0.0
    known["text_pred_arousal_input"] = 0.0

    target = target_df.copy().reset_index(drop=True)
    target["_event_type"] = "target"
    target["_target_pos"] = np.arange(len(target), dtype=int)
    target["text_pred_valence_input"] = target_text_predictions[:, 0]
    target["text_pred_arousal_input"] = target_text_predictions[:, 1]

    events = pd.concat([known, target], ignore_index=True, sort=False)
    events["_event_rank"] = (events["_event_type"] == "target").astype(int)
    events = events.sort_values([
        "user_id", "timestamp_parsed", "text_id", "_event_rank", "_row_id"
    ]).reset_index(drop=True)

    history_by_user: dict[int, list[dict[str, object]]] = {}
    predictions = np.zeros((len(target), 2), dtype=float)

    target_count = 0
    for _, row in events.iterrows():
        if row["_event_type"] == "known":
            append_history(history_by_user, row, float(row["valence"]), float(row["arousal"]))
        else:
            text_pred = np.array([row["text_pred_valence_input"], row["text_pred_arousal_input"]], dtype=float)
            feature_row = make_one_feature_row(row, text_pred, history_by_user, global_start)
            pred = np.asarray(model.predict_one(feature_row), dtype=float).reshape(2)
            target_pos = int(row["_target_pos"])
            predictions[target_pos] = pred
            append_history(history_by_user, row, float(pred[0]), float(pred[1]))
            target_count += 1
            if target_count % 500 == 0:
                print(f"recursive predicted {target_count}/{len(target_df)}", flush=True)

    return predictions


@dataclass
class ClosedFormRidge:
    """Small dense ridge regressor used for the longitudinal calibration layer."""
    feature_columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        x_arr = x[self.feature_columns].to_numpy(dtype=float)
        x_scaled = (x_arr - self.mean) / self.scale
        x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        return x_design @ self.weights

    def predict_one(self, features: dict[str, float]) -> np.ndarray:
        x_arr = np.array([features[col] for col in self.feature_columns], dtype=float)
        x_scaled = (x_arr - self.mean) / self.scale
        x_design = np.concatenate([np.array([1.0]), x_scaled])
        return x_design @ self.weights

    def feature_weights_df(self) -> pd.DataFrame:
        """Return a DataFrame of ridge weights sorted by absolute magnitude.

        Weights are in standardised-feature space (comparable across features).
        Row 0 of ``self.weights`` is the intercept; feature rows start at 1.
        """
        rows = []
        for i, col in enumerate(self.feature_columns):
            w_val = float(self.weights[i + 1, 0])
            w_aro = float(self.weights[i + 1, 1])
            rows.append({
                "feature": col,
                "weight_valence": w_val,
                "weight_arousal": w_aro,
                "abs_max": max(abs(w_val), abs(w_aro)),
            })
        return pd.DataFrame(rows).sort_values("abs_max", ascending=False).reset_index(drop=True)


def fit_longitudinal_model(x: pd.DataFrame, y: np.ndarray, alpha: float) -> ClosedFormRidge:
    """Fit a transparent linear calibration model by solving the ridge equations."""
    feature_columns = x.columns.tolist()
    x_arr = x.to_numpy(dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mean = x_arr.mean(axis=0)
    scale = x_arr.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    x_scaled = (x_arr - mean) / scale
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(x_design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y_arr)
    return ClosedFormRidge(
        feature_columns=feature_columns,
        mean=mean,
        scale=scale,
        weights=weights,
    )


def write_submission(path: str, df: pd.DataFrame, predictions: np.ndarray) -> None:
    out = pd.DataFrame({
        "user_id": df["user_id"].to_numpy(),
        "text_id": df["text_id"].to_numpy(),
        "pred_valence": predictions[:, 0],
        "pred_arousal": predictions[:, 1],
    })
    out.to_csv(path, index=False)
