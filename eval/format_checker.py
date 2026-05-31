#!/usr/bin/env python3
"""
Simple format checker for SemEval submission files (subtask1, subtask2a, subtask2b).

Checks:
- required columns exist in submission (warn on extra columns)
- all keys (user_id[, text_id]) present in template are present in submission
- prediction cells are non-null (not "", "NULL", "None", etc.)
- reports duplicate keys in submission

Usage (CLI):
    python format_checker.py --task subtask1 --submission path/to/sub.csv [--assets-dir assets]

API usage (importable):
    from format_checker import run_format_check
    rc = run_format_check("subtask1", "path/to/sub.csv", assets_dir="assets")
    # rc == 0 -> passed (warnings allowed)
    # rc == 2 -> failed

This script uses only Python stdlib modules: csv, argparse, os, sys, typing.
"""
import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple, Set, Optional
from constants import (
    TEMPLATES_DEFAULT,
    REQUIRED_COLUMNS_DEFAULT,
    KEY_COLUMNS_DEFAULT,
    NULL_VALUES_DEFAULT,
    DEFAULT_ASSETS_DIR,
)


def read_csv_dicts(path: str) -> List[Dict[str, str]]:
    """Read a CSV file and return list of rows as dicts (csv.DictReader)."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [row for row in reader]
    return rows


def norm(s: Optional[str]) -> str:
    """Normalize string for comparison: strip whitespace, handle None."""
    return s.strip() if s is not None else ""


def is_null(s: Optional[str], null_values: Set[str]) -> bool:
    """Determine whether a cell value should be considered 'null'."""
    return norm(s) in null_values


def check_columns(sub_cols: List[str], required: List[str]) -> Tuple[bool, List[str], List[str]]:
    """
    Compare submission columns with required columns.

    Returns:
        ok: True if no required columns missing
        missing: list of missing required columns
        extra: list of extra columns present in submission
    """
    sub_set = set(sub_cols)
    req_set = set(required)
    missing = [c for c in required if c not in sub_set]
    extra = [c for c in sub_cols if c not in req_set]
    ok = not missing
    return ok, missing, extra


def build_key(row: Dict[str, str], task: str) -> Tuple[str, ...]:
    """
    Build grouping key for a row based on task:
      - subtask1 -> (user_id, text_id)
      - subtask2a/2b -> (user_id,)
    """
    if task == "subtask1":
        return (norm(row.get("user_id", "")), norm(row.get("text_id", "")))
    return (norm(row.get("user_id", "")),)


def load_template_keys(template_path: str, task: str) -> Set[Tuple[str, ...]]:
    """Load keys from template CSV (canonical set of expected rows)."""
    rows = read_csv_dicts(template_path)
    keys = set()
    for r in rows:
        keys.add(build_key(r, task))
    return keys


def load_submission_keys_and_nulls(
    submission_path: str,
    task: str,
    required: List[str],
    null_values: Set[str],
) -> Tuple[Set[Tuple[str, ...]], List[str], List[Tuple[Tuple[str, ...], str]]]:
    """
    Read submission rows and return:
      - set of keys present
      - list of duplicate key string representations
      - list of (key, pred_field) tuples where prediction cells are null/missing
    """
    rows = read_csv_dicts(submission_path)
    keys: Set[Tuple[str, ...]] = set()
    duplicate_keys: List[str] = []
    null_entries: List[Tuple[Tuple[str, ...], str]] = []
    seen: Set[Tuple[str, ...]] = set()
    for r in rows:
        key = build_key(r, task)
        if key in seen:
            duplicate_keys.append(",".join(key))
        seen.add(key)
        keys.add(key)
        # check prediction fields for null
        pred_fields = [f for f in required if f.startswith("pred")]
        for pf in pred_fields:
            if pf not in r or is_null(r.get(pf, ""), null_values):
                null_entries.append((key, pf))
    return keys, duplicate_keys, null_entries


def run_format_check(
    task: str,
    submission_path: str,
    assets_dir: str = DEFAULT_ASSETS_DIR,
    templates: Dict[str, str] = TEMPLATES_DEFAULT,
    required_columns: Dict[str, List[str]] = REQUIRED_COLUMNS_DEFAULT,
    null_values: Set[str] = NULL_VALUES_DEFAULT,
) -> int:
    """
    Run the format check and print results.

    Returns:
        exit_code: 0 if passed (warnings allowed), 2 if failed.
    """
    # Step 1: Verify files exist
    if not os.path.isfile(submission_path):
        print(f"ERROR: submission file not found: {submission_path}", file=sys.stderr)
        return 2

    template_name = templates.get(task)
    if not template_name:
        print(f"ERROR: unknown task {task}", file=sys.stderr)
        return 2
    template_path = os.path.join(assets_dir, template_name)
    if not os.path.isfile(template_path):
        print(f"ERROR: template not found: {template_path}", file=sys.stderr)
        return 2

    required = required_columns[task]

    # Step 2: Read header of submission and check columns
    with open(submission_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            print("ERROR: empty submission file", file=sys.stderr)
            return 2
    header = [h.strip() for h in header]

    # Ensure key columns required for grouping are present regardless of REQUIRED_COLUMNS_DEFAULT
    key_cols = KEY_COLUMNS_DEFAULT.get(task, [])
    missing_key_cols = [c for c in key_cols if c not in header]
    if missing_key_cols:
        print(
            "FAIL: submission is missing key column(s) required for this task:",
            ", ".join(missing_key_cols),
            file=sys.stderr,
        )
        return 2

    # track whether any warnings were emitted so final message can be clean
    warnings_emitted = False

    cols_ok, missing_cols, extra_cols = check_columns(header, required)
    if missing_cols:
        print("FAIL: missing required columns:", ", ".join(missing_cols), file=sys.stderr)
        return 2
    if extra_cols:
        # extra columns are allowed but should be reported
        print("WARN: extra columns detected (these are allowed):", ", ".join(extra_cols))
        warnings_emitted = True

    # Step 3: Load keys from template and submission, check null prediction cells
    template_keys = load_template_keys(template_path, task)
    sub_keys, duplicates, null_entries = load_submission_keys_and_nulls(
        submission_path, task, required, null_values
    )

    # Only treat null prediction cells as failures when they occur for rows
    # that are required by the template. Nulls in extra submission rows are ignored.
    null_entries_in_template = [(k, pf) for (k, pf) in null_entries if k in template_keys]
    ignored_nulls = len(null_entries) - len(null_entries_in_template)
    if ignored_nulls:
        print(f"WARN: {ignored_nulls} null prediction cells found in rows not in template; these are ignored.")
        warnings_emitted = True

    missing_rows = template_keys - sub_keys
    extra_rows = sub_keys - template_keys

    failure = False

    # Step 4: Report issues
    if missing_rows:
        print(f"FAIL: submission is missing {len(missing_rows)} rows present in template. Example(s):")
        for k in sorted(missing_rows)[:5]:
            print("  -", ",".join(k))
        failure = True

    if extra_rows:
        print(f"WARN: submission contains {len(extra_rows)} rows not in template (extra). Example(s):")
        for k in sorted(extra_rows)[:5]:
            print("  -", ",".join(k))
        warnings_emitted = True

    if duplicates:
        print(f"WARN: duplicate rows (same key) in submission: {len(duplicates)} examples: {', '.join(duplicates[:5])}")
        warnings_emitted = True

    if null_entries_in_template:
        print(f"FAIL: found {len(null_entries_in_template)} prediction cells that are null or missing.")
        for k, pf in null_entries_in_template[:8]:
            print("  -", ",".join(k), "missing:", pf)
        failure = True

    # Step 5: Final status and exit code
    if failure:
        print("FORMAT CHECK: DID NOT PASS", file=sys.stderr)
        return 2

    # If no warnings were emitted, keep the success message clean
    if warnings_emitted:
        print("FORMAT CHECK: PASSED (warnings were printed).")
    else:
        print("FORMAT CHECK: PASSED.")
    return 0


def parse_cli_args() -> argparse.Namespace:
    """
    Parse command-line arguments and return the argparse Namespace.
    """
    p = argparse.ArgumentParser(description="Format checker for SemEval tasks 1/2")
    p.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR, help="directory containing templates")
    p.add_argument("--submission", required=True, help="path to submission csv")
    p.add_argument("--task", required=True, choices=["subtask1", "subtask2a", "subtask2b"])
    return p.parse_args()


def _cli_main() -> None:
    """CLI wrapper that parses args and delegates to run_format_check()."""
    args = parse_cli_args()
    rc = run_format_check(args.task, args.submission, assets_dir=args.assets_dir)
    sys.exit(rc)


if __name__ == "__main__":
    _cli_main()
