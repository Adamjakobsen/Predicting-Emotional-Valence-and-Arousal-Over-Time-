from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from longitudinal_utils import (
    TARGETS,
    build_training_longitudinal_features,
    build_val_split,
    evaluate,
    fit_longitudinal_model,
    prepare_frame,
    recursive_predict_targets,
)

ARTIFACTS   = Path("artifacts/paper_results/deberta_base")
TRAIN_CSV   = Path("datasets/train.csv")
TEST_CSV    = Path("datasets/test.csv")
ALPHA       = 5.0

# Cumulative feature groups — each entry adds to all previous ones.
FEATURE_GROUPS = [
    (
        "text predictions",
        ["text_pred_valence", "text_pred_arousal"],
    ),
    (
        "+ dataset metadata",
        ["collection_phase", "is_words",
         "days_from_global_start", "hour_sin", "hour_cos", "dow_sin", "dow_cos"],
    ),
    (
        "+ history presence",
        # How much history exists and when it was written.
        ["hist_has_history", "hist_n_prior",
         "hist_days_since_first", "hist_days_since_prev", "hist_phase_gap",
         "hist_n_words_prior", "hist_n_essays_prior"],
    ),
    (
        "+ most-recent targets",
        # The single previous observation 
        ["hist_prev_valence", "hist_prev_arousal",
         "hist_prev_delta_valence", "hist_prev_delta_arousal"],
    ),
    (
        "+ long-run statistics",
        # Full-history mean and std 
        ["hist_mean_valence_all", "hist_mean_arousal_all",
         "hist_std_valence_all", "hist_std_arousal_all"],
    ),
    (
        "+ windowed recency",
        # Short-window means (last 2, 3, 5 entries).
        ["hist_mean_valence_last2", "hist_mean_arousal_last2",
         "hist_mean_valence_last3", "hist_mean_arousal_last3",
         "hist_mean_valence_last5", "hist_mean_arousal_last5"],
    ),
    (
        "+ trends & interactions",
        # Linear trends over last 5 entries; text×history interaction terms.
        ["hist_trend_valence_last5", "hist_trend_arousal_last5", "hist_has_trend",
         "text_pred_valence_x_has_history", "text_pred_arousal_x_has_history"],
    ),
]


def fit_and_eval(train_df, test_df, train_preds, test_preds, global_start, feature_set):
    x = build_training_longitudinal_features(train_df, train_preds, global_start)
    cols = [c for c in x.columns if c in feature_set]
    model = fit_longitudinal_model(
        x[cols], train_df[TARGETS].to_numpy(float), alpha=ALPHA
    )
    preds = recursive_predict_targets(
        known_df=train_df,
        target_df=test_df,
        target_text_predictions=test_preds,
        model=model,
        feature_columns=cols,
        global_start=global_start,
    )
    return preds, len(cols)


def main():
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw  = pd.read_csv(TEST_CSV)
    train_df  = prepare_frame(train_raw, "train")
    test_df   = prepare_frame(test_raw,  "test")
    global_start = train_df["timestamp_parsed"].min()

    d          = np.load(ARTIFACTS / "text_predictions.npz")
    train_oof  = d["train_oof"]
    test_preds = d["test_pred"]

    print(f"\n{'Step':<30} {'n':>4}  {'r_comp':>7}  {'r_val':>7}  {'r_aro':>7}  {'Δr_comp':>8}")
    print("-" * 75)

    active_features: set[str] = set()
    prev_score = None

    for label, new_features in FEATURE_GROUPS:
        active_features.update(new_features)
        preds, n_cols = fit_and_eval(
            train_df, test_df, train_oof, test_preds, global_start, active_features
        )
        m = evaluate(test_df, preds[:, 0], preds[:, 1])
        r_comp = m["mean_r_composite"]
        r_val  = m["valence_r_composite"]
        r_aro  = m["arousal_r_composite"]
        delta  = f"{r_comp - prev_score:+.4f}" if prev_score is not None else "    —"
        print(f"{label:<30} {n_cols:>4}  {r_comp:>7.4f}  {r_val:>7.4f}  {r_aro:>7.4f}  {delta:>8}")
        prev_score = r_comp


if __name__ == "__main__":
    main()
