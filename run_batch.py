"""
Classify every line in data/sample_po_lines.csv, save results, and score against the answer key.

Usage:
    python run_batch.py              # uses Haiku by default
    python run_batch.py --model haiku
    python run_batch.py --model sonnet
"""

import argparse
import os
import sys
import pandas as pd

from classifier import classify_line, CATEGORIES
from config import (
    HAIKU_MODEL, SONNET_MODEL,
    HAIKU_INPUT_PRICE_PER_M, HAIKU_OUTPUT_PRICE_PER_M,
    SONNET_INPUT_PRICE_PER_M, SONNET_OUTPUT_PRICE_PER_M,
)

MODEL_MAP = {
    "haiku":  (HAIKU_MODEL,  HAIKU_INPUT_PRICE_PER_M,  HAIKU_OUTPUT_PRICE_PER_M),
    "sonnet": (SONNET_MODEL, SONNET_INPUT_PRICE_PER_M, SONNET_OUTPUT_PRICE_PER_M),
}


def run_batch(model_key: str = "haiku") -> None:
    model_id, price_in, price_out = MODEL_MAP[model_key]
    model_slug = model_id.replace("/", "-")

    po = pd.read_csv("data/sample_po_lines.csv")
    answers = pd.read_csv("data/answer_key.csv")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{model_slug}.csv"

    rows = []
    total = len(po)
    for i, row in po.iterrows():
        n = i + 1
        print(f"  [{n:>3}/{total}] {row['line_id']}  {row['supplier'][:35]}", end="\r", flush=True)
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
    results_df.to_csv(out_path, index=False)
    print(f"Results saved → {out_path}")

    # --- Cost ---
    total_in  = results_df["input_tokens"].sum()
    total_out = results_df["output_tokens"].sum()
    cost = (total_in * price_in + total_out * price_out) / 1_000_000
    print(f"\nTokens:  {total_in:,} in  /  {total_out:,} out")
    print(f"Cost:    ${cost:.4f}  ({model_id})")

    # --- Accuracy ---
    merged = results_df.merge(answers, on="line_id")
    error_rows = merged[merged["predicted_category"] == "error"]
    scored = merged[merged["predicted_category"] != "error"].copy()
    scored["correct"] = scored["predicted_category"] == scored["correct_category"]

    n_correct = scored["correct"].sum()
    n_scored  = len(scored)
    n_errors  = len(error_rows)
    overall_acc = n_correct / n_scored if n_scored else 0.0

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
        print("\nAll lines classified correctly.")
    else:
        print(f"\n{'─'*60}")
        print(f"MISCLASSIFIED  ({len(wrong)} lines)")
        print(f"{'─'*60}")
        for _, r in wrong.iterrows():
            note = answers.loc[answers["line_id"] == r["line_id"], "note"].values[0]
            note_str = f"  note: {note}" if pd.notna(note) and note else ""
            print(f"\n{r['line_id']}  {r['supplier'] if 'supplier' in r else ''}")
            print(f"  predicted : {r['predicted_category']}  (conf {r['confidence']:.2f})")
            print(f"  correct   : {r['correct_category']}{note_str}")
            print(f"  reason    : {r['reason']}")

    if not error_rows.empty:
        print(f"\n{'─'*60}")
        print("API ERRORS")
        for _, r in error_rows.iterrows():
            print(f"  {r['line_id']}  {r['reason']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["haiku", "sonnet"], default="haiku",
        help="Model to use (default: haiku)"
    )
    args = parser.parse_args()
    run_batch(args.model)
