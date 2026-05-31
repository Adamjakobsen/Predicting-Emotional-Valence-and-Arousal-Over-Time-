#!/usr/bin/env python3
# coding: utf-8
"""
Run the complete Subtask-1 paper pipeline with the longitudinal layer as default.

For each requested text backbone, this script produces:
  * validation text-only metrics;
  * validation longitudinal ablations;
  * test text-only metrics;
  * test longitudinal ablations;
  * one default longitudinal submission CSV.

Backbones:
  * deberta_base      -- DeBERTa-v3-base M6-style regression;
  * deberta_large     -- DeBERTa-v3-large M6-style regression;
  * gemma_lora        -- Gemma LoRA regression.

The default reported model is always the longitudinal calibrator. Text-only rows
are included only as ablations. No rule-based prediction path is implemented.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

from longitudinal_utils import (
    TARGETS,
    build_training_longitudinal_features,
    build_val_split,
    evaluate,
    evaluate_seen_unseen,
    fit_longitudinal_model,
    metrics_table,
    prepare_frame,
    recursive_predict_targets,
    write_submission,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s : %(levelname)s : %(message)s")
log = logging.getLogger("main")

MODEL_PRESETS = {
    "deberta_base": {
        "kind": "transformer",
        "model_family": "deberta",
        "backbone": "microsoft/deberta-v3-base",
        "epochs_arg": "deberta_epochs",
        "batch_arg": "deberta_batch_size",
        "lr_arg": "deberta_lr",
        "max_len_arg": "deberta_max_len",
        "pooling": "cls",
        "grad_accum": 1,
    },
    "deberta_large": {
        "kind": "transformer",
        "model_family": "deberta",
        "backbone": "microsoft/deberta-v3-large",
        "epochs_arg": "large_epochs",
        "batch_arg": "large_batch_size",
        "lr_arg": "deberta_lr",
        "max_len_arg": "large_max_len",
        "pooling": "cls",
        # grad_accum=4 matches base's effective batch size of 16 (4 × 4 = 16).
        # Without this the large model sees 4× fewer gradient steps per epoch,
        # which was the likely cause of its underperformance vs. base.
        "grad_accum": 4,
    },
    # --- Ablation presets (run with --models deberta_base_ccc / deberta_base_mean_pool) ---
    "deberta_base_ccc": {
        "kind": "transformer",
        "model_family": "deberta",
        "backbone": "microsoft/deberta-v3-base",
        "epochs_arg": "deberta_epochs",
        "batch_arg": "deberta_batch_size",
        "lr_arg": "deberta_lr",
        "max_len_arg": "deberta_max_len",
        "pooling": "cls",
        "grad_accum": 1,
        "loss": "ccc",
    },
    "deberta_base_mean_pool": {
        "kind": "transformer",
        "model_family": "deberta",
        "backbone": "microsoft/deberta-v3-base",
        "epochs_arg": "deberta_epochs",
        "batch_arg": "deberta_batch_size",
        "lr_arg": "deberta_lr",
        "max_len_arg": "deberta_max_len",
        "pooling": "mean",
        "grad_accum": 1,
        "loss": "mse",
    },
    "gemma_lora": {
        "kind": "transformer",
        "model_family": "gemma_lora",
        "backbone": "google/gemma-4-E4B-it",
        "epochs_arg": "gemma_epochs",
        "batch_arg": "gemma_batch_size",
        "lr_arg": "gemma_lr",
        "max_len_arg": "gemma_max_len",
        "pooling": "mean",
        "grad_accum_arg": "gemma_grad_accum",
    },
}

ABLATION_VARIANTS = [
    "longitudinal_default",
    "longitudinal_no_history",
    "longitudinal_no_text",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, default=Path("train.csv"))
    p.add_argument("--test", type=Path, default=Path("test.csv"))
    p.add_argument("--artifacts_dir", type=Path, default=Path("artifacts/paper_results"))
    p.add_argument("--cache_dir", type=Path, default=Path("artifacts/cache"))
    p.add_argument("--worker", type=Path, default=Path(__file__).with_name("transformer_worker.py"))
    p.add_argument(
        "--models",
        nargs="+",
        default=["deberta_base", "deberta_large", "gemma_lora"],
        choices=sorted(MODEL_PRESETS),
        help="Models to run. Default runs all paper models. Ablation presets: deberta_base_ccc, deberta_base_mean_pool.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 17, 23])
    p.add_argument("--n_splits", type=int, default=3)
    p.add_argument("--val_seed", type=int, default=13)
    p.add_argument("--val_unseen_frac", type=float, default=0.20)
    p.add_argument("--val_within_frac", type=float, default=0.25)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    # Longitudinal layer.
    p.add_argument("--long_alpha", type=float, default=5.0)

    # DeBERTa settings.
    p.add_argument("--deberta_epochs", type=int, default=4)
    p.add_argument("--deberta_batch_size", type=int, default=16)
    p.add_argument("--deberta_lr", type=float, default=2e-5)
    p.add_argument("--deberta_max_len", type=int, default=192)

    # DeBERTa-large settings.
    p.add_argument("--large_epochs", type=int, default=4)
    p.add_argument("--large_batch_size", type=int, default=4)
    p.add_argument("--large_max_len", type=int, default=192)

    # Gemma LoRA settings.
    p.add_argument("--gemma_epochs", type=int, default=4)
    p.add_argument("--gemma_batch_size", type=int, default=4)
    p.add_argument("--gemma_grad_accum", type=int, default=4)
    p.add_argument("--gemma_lr", type=float, default=1e-4)
    p.add_argument("--gemma_max_len", type=int, default=192)
    p.add_argument("--gemma_lora_r", type=int, default=16)
    p.add_argument("--gemma_lora_alpha", type=int, default=32)
    p.add_argument("--gemma_lora_dropout", type=float, default=0.05)
    p.add_argument("--gemma_lora_targets", choices=["attn", "attn+mlp"], default="attn")
    p.add_argument("--gemma_text_transform", choices=["none", "gemma_chat"], default="none")
    p.add_argument("--gemma_head_warmup_epochs", type=int, default=0)
    return p.parse_args()


def save_frame(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "user_id", "text_id", "text", "text_clean", "is_words", "valence", "arousal",
        "timestamp", "collection_phase",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(path, index=False)


def metric_row(model_name: str, split: str, variant: str, metrics: dict[str, float]) -> dict[str, object]:
    row: dict[str, object] = {"model": model_name, "split": split, "variant": variant}
    for key, value in metrics.items():
        if isinstance(value, (float, int, np.floating, np.integer)):
            row[key] = float(value)
    return row


def selected_columns(all_columns: list[str], variant: str) -> list[str]:
    if variant == "longitudinal_default":
        return list(all_columns)
    if variant == "longitudinal_no_history":
        return [c for c in all_columns if not c.startswith("hist_")]
    if variant == "longitudinal_no_text":
        return [c for c in all_columns if not c.startswith("text_pred_")]
    raise ValueError(f"unknown longitudinal variant: {variant}")


def fit_predict_longitudinal(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    train_text_predictions: np.ndarray,
    target_text_predictions: np.ndarray,
    global_start: pd.Timestamp,
    alpha: float,
    variant: str,
):
    x_all = build_training_longitudinal_features(train_df, train_text_predictions, global_start)
    cols = selected_columns(x_all.columns.tolist(), variant)
    model = fit_longitudinal_model(x_all[cols], train_df[TARGETS].to_numpy(dtype=float), alpha=alpha)
    preds = recursive_predict_targets(
        known_df=train_df,
        target_df=target_df,
        target_text_predictions=target_text_predictions,
        model=model,
        feature_columns=cols,
        global_start=global_start,
    )
    return preds, cols, model


def worker_command(
    args: argparse.Namespace,
    preset: dict[str, object],
    train_csv: Path,
    predict_csv: Path,
    out_npz: Path,
    metrics_json: Path,
    label: str,
    seed: int,
) -> list[str]:
    grad_accum = int(preset.get("grad_accum", getattr(args, str(preset.get("grad_accum_arg")), 1)))
    loss = str(preset.get("loss", "mse"))
    cmd = [
        sys.executable, str(args.worker),
        "--train", str(train_csv),
        "--predict", str(predict_csv),
        "--out", str(out_npz),
        "--metrics_out", str(metrics_json),
        "--model_family", str(preset["model_family"]),
        "--backbone", str(preset["backbone"]),
        "--label", label,
        "--seed", str(seed),
        "--device", args.device,
        "--cache_dir", str(args.cache_dir),
        "--epochs", str(getattr(args, str(preset["epochs_arg"]))),
        "--batch_size", str(getattr(args, str(preset["batch_arg"]))),
        "--grad_accum", str(grad_accum),
        "--lr", str(getattr(args, str(preset["lr_arg"]))),
        "--max_len", str(getattr(args, str(preset["max_len_arg"]))),
        "--pooling", str(preset["pooling"]),
        "--loss", loss,
    ]
    if preset["model_family"] == "gemma_lora":
        cmd += [
            "--lora_r", str(args.gemma_lora_r),
            "--lora_alpha", str(args.gemma_lora_alpha),
            "--lora_dropout", str(args.gemma_lora_dropout),
            "--lora_targets", args.gemma_lora_targets,
            "--text_transform", args.gemma_text_transform,
            "--head_warmup_epochs", str(args.gemma_head_warmup_epochs),
        ]
    return cmd


def run_worker(cmd: list[str]) -> None:
    log.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_prediction_npz(path: Path) -> np.ndarray:
    data = np.load(path)
    return np.column_stack([data["pred_valence"], data["pred_arousal"]]).astype(float)


def transformer_predict_once(
    args: argparse.Namespace,
    preset: dict[str, object],
    model_name: str,
    model_dir: Path,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    run_label: str,
    seed: int,
) -> np.ndarray:
    run_dir = model_dir / "worker_inputs" / run_label / f"seed{seed}"
    train_csv = run_dir / "train.csv"
    pred_csv = run_dir / "predict.csv"
    save_frame(train_df, train_csv)
    save_frame(predict_df, pred_csv)
    out_npz = model_dir / "worker_outputs" / f"{run_label}_seed{seed}.npz"
    metrics_json = model_dir / "worker_outputs" / f"{run_label}_seed{seed}.metrics.json"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    cmd = worker_command(
        args=args,
        preset=preset,
        train_csv=train_csv,
        predict_csv=pred_csv,
        out_npz=out_npz,
        metrics_json=metrics_json,
        label=f"{model_name}_{run_label}_seed{seed}",
        seed=seed,
    )
    run_worker(cmd)
    return load_prediction_npz(out_npz)


def transformer_ensemble_predictions(
    args: argparse.Namespace,
    preset: dict[str, object],
    model_name: str,
    model_dir: Path,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    run_label: str,
) -> np.ndarray:
    preds = []
    for seed in args.seeds:
        preds.append(transformer_predict_once(args, preset, model_name, model_dir, train_df, predict_df, run_label, seed))
    return np.mean(preds, axis=0)


def run_transformer_predictions(
    model_name: str,
    preset: dict[str, object],
    model_dir: Path,
    dev_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    log.info("[%s] validation OOF transformer predictions", model_name)
    splitter = GroupKFold(n_splits=args.n_splits)
    dev_oof = np.zeros((len(dev_df), 2), dtype=float)
    groups = dev_df["user_id"].to_numpy()
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(dev_df, dev_df[TARGETS], groups), start=1):
        fold_train = dev_df.iloc[fit_idx].reset_index(drop=True)
        fold_hold = dev_df.iloc[hold_idx].reset_index(drop=True)
        dev_oof[hold_idx] = transformer_ensemble_predictions(
            args, preset, model_name, model_dir, fold_train, fold_hold, f"validation_oof_fold{fold}"
        )
    val_pred = transformer_ensemble_predictions(args, preset, model_name, model_dir, dev_df, val_df, "validation_to_val")

    log.info("[%s] final OOF transformer predictions", model_name)
    train_oof = np.zeros((len(train_df), 2), dtype=float)
    groups = train_df["user_id"].to_numpy()
    for fold, (fit_idx, hold_idx) in enumerate(splitter.split(train_df, train_df[TARGETS], groups), start=1):
        fold_train = train_df.iloc[fit_idx].reset_index(drop=True)
        fold_hold = train_df.iloc[hold_idx].reset_index(drop=True)
        train_oof[hold_idx] = transformer_ensemble_predictions(
            args, preset, model_name, model_dir, fold_train, fold_hold, f"final_oof_fold{fold}"
        )
    test_pred = transformer_ensemble_predictions(args, preset, model_name, model_dir, train_df, test_df, "final_to_test")

    np.savez_compressed(
        model_dir / "text_predictions.npz",
        dev_oof=dev_oof,
        val_pred=val_pred,
        train_oof=train_oof,
        test_pred=test_pred,
    )
    return {"dev_oof": dev_oof, "val_pred": val_pred, "train_oof": train_oof, "test_pred": test_pred}


def run_one_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    is_val: pd.Series,
    global_start: pd.Timestamp,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    preset = MODEL_PRESETS[model_name]
    model_dir = args.artifacts_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    dev_df = train_df.loc[~is_val].reset_index(drop=True).copy()
    val_df = train_df.loc[is_val].reset_index(drop=True).copy()
    log.info("[%s] dev=%d val=%d train=%d test=%d", model_name, len(dev_df), len(val_df), len(train_df), len(test_df))

    pred_pack = run_transformer_predictions(model_name, preset, model_dir, dev_df, val_df, train_df, test_df, args)

    rows: list[dict[str, object]] = []

    # Text-only ablation rows.
    val_text_metrics = evaluate(val_df, pred_pack["val_pred"][:, 0], pred_pack["val_pred"][:, 1])
    test_text_metrics = evaluate(test_df, pred_pack["test_pred"][:, 0], pred_pack["test_pred"][:, 1])
    rows.append(metric_row(model_name, "validation", "text_only", val_text_metrics))
    rows.append(metric_row(model_name, "test", "text_only", test_text_metrics))

    metrics_payload: dict[str, object] = {
        "model": model_name,
        "validation": {"text_only": val_text_metrics},
        "test": {"text_only": test_text_metrics},
        "feature_columns": {},
        "feature_weights": {},
    }

    # Text-only seen/unseen breakdown (test only; val comes from training data, no is_seen_user).
    test_text_seen_unseen = evaluate_seen_unseen(test_df, pred_pack["test_pred"][:, 0], pred_pack["test_pred"][:, 1])
    metrics_payload["test_by_user_status"] = {"text_only": test_text_seen_unseen}

    # Longitudinal default + ablations.
    for variant in ABLATION_VARIANTS:
        log.info("[%s] validation %s", model_name, variant)
        val_long, val_cols, _ = fit_predict_longitudinal(
            train_df=dev_df,
            target_df=val_df,
            train_text_predictions=pred_pack["dev_oof"],
            target_text_predictions=pred_pack["val_pred"],
            global_start=global_start,
            alpha=args.long_alpha,
            variant=variant,
        )
        val_metrics = evaluate(val_df, val_long[:, 0], val_long[:, 1])
        rows.append(metric_row(model_name, "validation", variant, val_metrics))

        log.info("[%s] test %s", model_name, variant)
        test_long, test_cols, test_model = fit_predict_longitudinal(
            train_df=train_df,
            target_df=test_df,
            train_text_predictions=pred_pack["train_oof"],
            target_text_predictions=pred_pack["test_pred"],
            global_start=global_start,
            alpha=args.long_alpha,
            variant=variant,
        )
        test_metrics = evaluate(test_df, test_long[:, 0], test_long[:, 1])
        rows.append(metric_row(model_name, "test", variant, test_metrics))

        # Seen/unseen breakdown for this variant.
        test_seen_unseen = evaluate_seen_unseen(test_df, test_long[:, 0], test_long[:, 1])
        metrics_payload["test_by_user_status"][variant] = test_seen_unseen

        # Ridge feature weights (standardised-space, comparable across features).
        weights_df = test_model.feature_weights_df()
        metrics_payload.setdefault("feature_weights", {})[variant] = weights_df.to_dict(orient="records")

        metrics_payload["validation"][variant] = val_metrics
        metrics_payload["test"][variant] = test_metrics
        metrics_payload["feature_columns"][variant] = test_cols

        pred_path = model_dir / f"submission_{variant}.csv"
        write_submission(str(pred_path), test_df, test_long)
        np.savez_compressed(model_dir / f"predictions_{variant}.npz", pred_valence=test_long[:, 0], pred_arousal=test_long[:, 1])
        log.info("[%s] wrote %s", model_name, pred_path)

        if variant == "longitudinal_default":
            log.info("[%s] top-10 feature weights:\n%s", model_name,
                     weights_df.head(10).round(4).to_string(index=False))

    (model_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    log.info("[%s] text-only test metrics:\n%s", model_name, metrics_table(test_text_metrics).round(4).to_string(index=False))
    log.info("[%s] default longitudinal test metrics:\n%s", model_name, metrics_table(metrics_payload["test"]["longitudinal_default"]).round(4).to_string(index=False))
    return rows


def main() -> None:
    args = parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.worker.exists():
        raise FileNotFoundError(f"worker script not found: {args.worker}")

    log.info("loading data")
    train_df = prepare_frame(pd.read_csv(args.train), "train")
    test_df = prepare_frame(pd.read_csv(args.test), "test")
    global_start = min(train_df["timestamp_parsed"].min(), test_df["timestamp_parsed"].min())
    is_val = build_val_split(
        train_df,
        seed=args.val_seed,
        unseen_frac=args.val_unseen_frac,
        within_user_holdout=args.val_within_frac,
    )

    config_payload = vars(args).copy()
    config_payload["train"] = str(args.train)
    config_payload["test"] = str(args.test)
    config_payload["artifacts_dir"] = str(args.artifacts_dir)
    config_payload["cache_dir"] = str(args.cache_dir)
    config_payload["worker"] = str(args.worker)
    (args.artifacts_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    all_rows: list[dict[str, object]] = []
    for model_name in args.models:
        all_rows.extend(run_one_model(model_name, train_df, test_df, is_val, global_start, args))

    summary = pd.DataFrame(all_rows)
    summary_path = args.artifacts_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    log.info("wrote %s", summary_path)

    compact_cols = ["model", "split", "variant", "valence_r_composite", "arousal_r_composite", "mean_r_composite"]
    available = [c for c in compact_cols if c in summary.columns]
    log.info("compact summary:\n%s", summary[available].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
