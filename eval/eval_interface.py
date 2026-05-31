#!/usr/bin/env python3
"""
Simple evaluation interface that ties together format_checker.py and eval.py.

Usage (CLI):
  python eval_interface.py --task subtask1 --submission sub.csv --labels gold.csv [--assets-dir assets] [--dimension valence|arousal] [--json out.json]

Functions are importable. Example API usage:

    from eval_interface import evaluate_submission
    result = evaluate_submission(
        "subtask1", "path/to/sub.csv", "path/to/gold.csv", assets_dir="assets", dimension="valence", return_dict=True
    )

When called with return_dict=True the function returns a structured dict with status/metrics/warnings instead of exiting or printing.
"""
from typing import Dict, Any, List, Tuple, Optional
import argparse
import csv
import json
import os
import sys

import format_checker
import eval as eval_mod
from constants import (
    TEMPLATES_DEFAULT, 
    KEY_COLUMNS_DEFAULT, 
    PRED_COLS_DEFAULT, 
    LABEL_COLS_DEFAULT
)

def _read_csv_map(path: str, key_cols: List[str]) -> Tuple[Dict[Tuple[str, ...], Dict[str, str]], List[str]]:
    """
    Read CSV and return a map from key tuple -> row dict, and list of duplicate key strings.
    Key columns are taken in order and converted to str.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    m: Dict[Tuple[str, ...], Dict[str, str]] = {}
    duplicates: List[str] = []
    for r in rows:
        key = tuple((r.get(c, "").strip() for c in key_cols))
        if key in m:
            duplicates.append(",".join(key)
            )
        else:
            m[key] = r
    return m, duplicates


def _safe_float(s: Optional[str]) -> float:
    """Convert string to float, return np.nan on empty / invalid."""
    if s is None:
        return float("nan")
    s = s.strip()
    if s == "" or s.lower() in {"none", "null"}:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def evaluate_submission(
    task: str,
    submission_path: str,
    labels_path: str,
    assets_dir: str = "assets",
    dimension: str = "valence",
    return_dict: bool = False,
    json_out: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run format check, then compute evaluation metric for one dimension (valence or arousal).

    Returns a dict with keys:
      - status: 'ok' or 'format_fail' or 'duplicate_error' or 'label_missing' etc.
      - task, dimension
      - warnings: list[str]
      - metrics: output from eval.py functions under 'metrics'
    """
    warnings: List[str] = []

    # 1) run format checker
    if task != "subtask1":
        rc = format_checker.run_format_check(task, submission_path, assets_dir=assets_dir)
        if rc != 0:
            msg = f"Format check failed (exit code {rc})."
            if return_dict:
                return {"status": "format_fail", "message": msg, "task": task}
            print(msg, file=sys.stderr)
            sys.exit(rc)

    # 2) load template keys and key columns
    template_name = TEMPLATES_DEFAULT.get(task)
    template_path = os.path.join(assets_dir, template_name)
    key_cols = KEY_COLUMNS_DEFAULT[task]
    template_keys = format_checker.load_template_keys(template_path, task)

    # 3) load submission and labels
    sub_map, sub_duplicates = _read_csv_map(submission_path, key_cols)
    if sub_duplicates:
        msg = f"ERROR: duplicate keys in submission: {len(sub_duplicates)} examples: {sub_duplicates[:5]}"
        if return_dict:
            return {"status": "duplicate_error", "message": msg, "task": task}
        print(msg, file=sys.stderr)
        sys.exit(2)

    labels_map, label_duplicates = _read_csv_map(labels_path, key_cols)
    if label_duplicates:
        # duplicates in labels file — treat as error
        msg = f"ERROR: duplicate keys in labels file: {len(label_duplicates)} examples: {label_duplicates[:5]}"
        if return_dict:
            return {"status": "label_duplicate_error", "message": msg, "task": task}
        print(msg, file=sys.stderr)
        sys.exit(2)

    # 4) build arrays in template key order
    # use defaults from constants module
    pred_col = PRED_COLS_DEFAULT[task][dimension]
    label_col = LABEL_COLS_DEFAULT[task][dimension]

    preds: List[float] = []
    labs: List[float] = []
    users: List[str] = []
    texts: List[str] = []

    missing_label_keys: List[Tuple[str, ...]] = []
    missing_submission_keys: List[Tuple[str, ...]] = []

    for key in sorted(template_keys):
        # key is tuple of strings
        sub_row = sub_map.get(key)
        lab_row = labels_map.get(key)
        if sub_row is None:
            if task == "subtask1":
                continue
            missing_submission_keys.append(key)
            continue
        if lab_row is None:
            if task == "subtask1":
                continue
            missing_label_keys.append(key)
            continue
        # read values
        pred_val = _safe_float(sub_row.get(pred_col))
        lab_val = _safe_float(lab_row.get(label_col))
            
        preds.append(pred_val)
        labs.append(lab_val)
        # store user/text for eval functions
        # fill based on key columns
        if task == "subtask1":
            users.append(key[0])
            texts.append(key[1])
        else:
            users.append(key[0])
            texts.append("")

    if missing_submission_keys:
        msg = f"ERROR: submission missing {len(missing_submission_keys)} rows from template. Example: {missing_submission_keys[:5]}"
        if return_dict:
            return {"status": "submission_missing_rows", "message": msg}
        print(msg, file=sys.stderr)
        sys.exit(2)

    if missing_label_keys:
        msg = f"ERROR: labels file missing {len(missing_label_keys)} rows required for evaluation. Example: {missing_label_keys[:5]}"
        if return_dict:
            return {"status": "label_missing_rows", "message": msg}
        print(msg, file=sys.stderr)
        sys.exit(2)

    # 5) compute metrics
    metrics: Dict[str, Any] = {}
    if task == "subtask1":
        metrics = eval_mod.task1_correlation(users, texts, preds, labs)
    else:
        # for task2 use between-person metric via task2_correlation (user_ids provided)
        metrics = eval_mod.task2_correlation(users, preds, labs)

    out: Dict[str, Any] = {"status": "ok", "task": task, "dimension": dimension, "metrics": metrics, "warnings": warnings}

    # 6) output
    if json_out:
        with open(json_out, "w", encoding="utf-8") as fo:
            json.dump(out, fo, indent=2)



    # pretty print
    print(f"Evaluation results for {task} ({dimension}):")

    # Helper for p-value formatting (used by both blocks)
    def fmt_p(val):
        if val is None: return "N/A"
        if val < 0.0001: return f"{val:.2e}"
        return f"{val:.4f}"

    if task == "subtask1":
        # Subtask 1: Group specific between/within/composite keys
        groups = [("r_between", "p_between"), ("r_within", "p_within"), ("r_composite", None)]
        processed = set()

        for r_key, p_key in groups:
            if r_key in metrics:
                r_val = metrics[r_key]
                p_str = ""
                if p_key and p_key in metrics:
                    p_val = metrics[p_key]
                    p_str = f" ({fmt_p(p_val)})"
                    processed.add(p_key)
                print(f"  {r_key:<14} {r_val:.3f}{p_str}")
                processed.add(r_key)
        
        # Print remaining keys for subtask1 (like MAE)
        for k in sorted(metrics.keys()):
            if k not in processed:
                v = metrics[k]
                print(f"  {k:<14} {v:.3f}")

    else:
        # Subtask 2 / Generic: Group 'r' and 'p' if they exist
        processed = set()
        
        # Check specifically for standard 'r'
        if "r" in metrics:
            r_val = metrics["r"]
            p_str = ""
            if "p" in metrics:
                p_val = metrics["p"]
                p_str = f" ({fmt_p(p_val)})"
                processed.add("p")
            
            print(f"  {'r':<14} {r_val:.3f}{p_str}")
            processed.add("r")

        # Print remaining keys (like 'mae', 'n', etc.)
        for k in sorted(metrics.keys()):
            if k not in processed:
                v = metrics[k]
                if isinstance(v, (int, float)):
                    print(f"  {k:<14} {v:.3f}")
                else:
                    print(f"  {k:<14} {v}")

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(" -", w)
    if return_dict:
        return out
    return out 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluation interface using format checker + eval functions")
    p.add_argument("--task", required=True, choices=["subtask1", "subtask2a", "subtask2b"]) 
    p.add_argument("--submission", required=True, help="path to submission CSV")
    p.add_argument("--labels", required=True, help="path to gold labels CSV")
    p.add_argument("--assets-dir", default="assets", help="assets directory containing templates for submission files (user_ids and text_ids will be updated in the template during the evaluation phase.")
    p.add_argument("--dimension", choices=["valence", "arousal", "pred_state_change_valence", "pred_state_change_arousal"], default="valence")
    p.add_argument("--json", help="optional path to write JSON output")
    return p.parse_args()


def _cli_main() -> None:
    args = parse_args()
    res = evaluate_submission(
        args.task, args.submission, args.labels, assets_dir=args.assets_dir, dimension=args.dimension, return_dict=False, json_out=args.json
    )
    # exit code: 0 if status ok, 2 otherwise
    if res.get("status") != "ok":
        sys.exit(2)


if __name__ == "__main__":
    _cli_main()
