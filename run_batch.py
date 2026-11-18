"""
Classify every line in data/sample_po_lines.csv, save results, and score against the answer key.

Usage:
    python run_batch.py              # uses Haiku by default
    python run_batch.py --model haiku
    python run_batch.py --model sonnet
    python run_batch.py --force      # overwrite saved results even if many rows errored

Exit code is 0 on success and 1 if the run errored out (results not saved, or nothing scored).
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from classifier import classify_line, CATEGORIES
from config import (
    HAIKU_MODEL, SONNET_MODEL,
    HAIKU_INPUT_PRICE_PER_M, HAIKU_OUTPUT_PRICE_PER_M,
    SONNET_INPUT_PRICE_PER_M, SONNET_OUTPUT_PRICE_PER_M,
)

# Resolve paths relative to this file so the script works from any directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# If more than this share of rows errored, don't overwrite the saved results
# (an API outage or bad key must not replace good results with error rows).
MAX_ERROR_RATE = 0.10

MODEL_MAP = {
    "haiku":  (HAIKU_MODEL,  HAIKU_INPUT_PRICE_PER_M,  HAIKU_OUTPUT_PRICE_PER_M),
    "sonnet": (SONNET_MODEL, SONNET_INPUT_PRICE_PER_M, SONNET_OUTPUT_PRICE_PER_M),
}


def _short(value, n: int) -> str:
    """First n characters of a cell as text; blank/NaN cells become ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)[:n]


def run_batch(model_key: str = "haiku", force: bool = False) -> int:
    """Run the batch. Returns a process exit code (0 = ok, 1 = problem)."""
    model_id, price_in, price_out = MODEL_MAP[model_key]
    model_slug = model_id.replace("/", "-")

    po = pd.read_csv(DATA_DIR / "sample_po_lines.csv")
    answers = pd.read_csv(DATA_DIR / "answer_key.csv")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{model_slug}.csv"
    shown_path = f"results/{out_path.name}"

    rows = []
    total = len(po)
    for i, row in po.iterrows():
        n = i + 1
        print(f"  [{n:>3}/{total}] {row['line_id']}  {_short(row['supplier'], 35)}", end="\r", flush=True)
        result = classify_line(
            supplier=row["supplier"],
            description=row["description"],
            amount=row["amount_usd"],
            model=model_id,
        )
        total_tokens = result["input_tokens"] + result["output_tokens"]
        rows.append({
            "line_id":            row["line_id"],
            "predicted_category": result["category"],
            "confidence":         result["confidence"],
            "reason":             result["reason"],
            "input_tokens":       result["input_tokens"],
            "output_tokens":      result["output_tokens"],
            "tokens":             total_tokens,
            "latency_seconds":    result["latency_seconds"],
        })

    print()  # clear the progress line

    results_df = pd.DataFrame(rows)
    error_rows = results_df[results_df["predicted_category"] == "error"]
    n_errors = len(error_rows)
    exit_code = 0

    # --- Save (never replace good saved results with a mostly-failed run) ---
    if total and n_errors / total > MAX_ERROR_RATE and not force:
        print(
            f"NOT SAVED: {n_errors}/{total} rows errored (more than {MAX_ERROR_RATE:.0%}). "
            f"The existing {shown_path} was left untouched. "
            "Fix the problem and re-run, or pass --force to overwrite anyway."
        )
        exit_code = 1
    else:
        tmp_path = out_path.with_name(out_path.name + ".tmp")
        results_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, out_path)  # atomic: no half-written results file
        print(f"Results saved → {shown_path}")

    # --- Cost ---
    total_in  = results_df["input_tokens"].sum()
    total_out = results_df["output_tokens"].sum()
    cost = (total_in * price_in + total_out * price_out) / 1_000_000
    print(f"\nTokens:  {total_in:,} in  /  {total_out:,} out")
    print(f"Cost:    ${cost:.4f}  ({model_id})")

    # --- Accuracy ---
    answer_cols = ["line_id", "correct_category"] + (["note"] if "note" in answers.columns else [])
    with_supplier = results_df.merge(po[["line_id", "supplier"]], on="line_id", how="left")
    merged = with_supplier.merge(answers[answer_cols], on="line_id", how="inner")

    unscored_ids = sorted(set(results_df["line_id"]) - set(merged["line_id"]))
    if unscored_ids:
        shown = ", ".join(map(str, unscored_ids[:5])) + (" …" if len(unscored_ids) > 5 else "")
        print(f"\nWARNING: {len(unscored_ids)} line(s) have no answer-key entry and were not scored: {shown}")

    scored = merged[merged["predicted_category"] != "error"].copy()
    scored["correct"] = scored["predicted_category"] == scored["correct_category"]

    n_correct = scored["correct"].sum()
    n_scored  = len(scored)
    overall_acc = n_correct / n_scored if n_scored else 0.0
    if n_scored == 0:
        exit_code = 1

    print(f"\n{'─'*60}")
    print(f"ACCURACY  {n_correct}/{n_scored} = {overall_acc:.1%}", end="")
    if n_errors:
        print(f"  ({n_errors} API error row(s) excluded)", end="")
    print()

    # Per-category accuracy
    print(f"\n{'Category':<24}  {'Correct':>7}  {'Total':>5}  {'Acc':>6}")
    print(f"{'─'*24}  {'─'*7}  {'─'*5}  {'─'*6}")
    for cat in CATEGORIES:
        cat_rows = scored[scored["correct_category"] == cat]
        n_cat = len(cat_rows)
        n_cat_ok = cat_rows["correct"].sum()
        acc = n_cat_ok / n_cat if n_cat else 0.0
        flag = "  ✗" if acc < 1.0 and n_cat else ""
        print(f"{cat:<24}  {n_cat_ok:>7}  {n_cat:>5}  {acc:>5.0%}{flag}")

    # Misclassified lines
    wrong = scored[~scored["correct"]].copy()
    if wrong.empty:
        if n_scored:
            print("\nAll lines classified correctly.")
    else:
        print(f"\n{'─'*60}")
        print(f"MISCLASSIFIED  ({len(wrong)} lines)")
        print(f"{'─'*60}")
        for _, r in wrong.iterrows():
            note = r["note"] if "note" in r.index and pd.notna(r["note"]) else ""
            note_str = f"  note: {note}" if note else ""
            print(f"\n{r['line_id']}  {_short(r['supplier'], 60)}")
            print(f"  predicted : {r['predicted_category']}  (conf {r['confidence']:.2f})")
            print(f"  correct   : {r['correct_category']}{note_str}")
            print(f"  reason    : {r['reason']}")

    if not error_rows.empty:
        print(f"\n{'─'*60}")
        print("API ERRORS")
        for _, r in error_rows.iterrows():
            print(f"  {r['line_id']}  {r['reason']}")

    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["haiku", "sonnet"], default="haiku",
        help="Model to use (default: haiku)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite saved results even if more than 10%% of rows errored"
    )
    args = parser.parse_args()
    sys.exit(run_batch(args.model, args.force))
