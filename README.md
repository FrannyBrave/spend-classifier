# Spend Classifier

A Streamlit app that uses the Claude API to classify procurement purchase-order lines into spend categories, with a human review queue for low-confidence predictions.

---

## Why

Categorizing PO lines is one of the most repetitive tasks in procurement. Analysts spend hours manually assigning spend categories to hundreds or thousands of lines so the business can answer basic questions like "how much did we spend on logistics last quarter?" This app automates that classification and routes low-confidence lines to a human review queue, combining speed with a lightweight quality-control step.

---

## What it does

- **Classify tab** — upload a CSV of PO lines (or use the built-in sample), choose a model, and run batch classification. Each line gets a category, a confidence score, a one-sentence reason, and token/latency metadata. Results download as CSV. A "Load saved results" button lets you demo without calling the API.
- **Review Queue tab** — set a confidence threshold (default 75%). Lines below it are queued for human review: approve the AI's category or override it from a dropdown. Decisions feed into the accuracy calculation.
- **Results tab** — accuracy of the AI alone and after human review, broken down above and below the threshold, a confusion matrix, and a side-by-side model comparison from saved batch runs.

**Six spend categories:** Direct Materials · MRO · Logistics & Freight · IT & Software · Professional Services · Facilities

---

## How it works

1. Each PO line (supplier, description, amount) is sent to the Claude API with a system prompt containing the category definitions and tie-breaker rules.
2. The model responds by calling a structured tool (`record_classification`) that enforces a valid category, a 0–1 confidence score, and a one-sentence reason — no text parsing needed.
3. Lines above the confidence threshold are auto-accepted; lines below go to the review queue.
4. Final categories (AI or human-corrected) are scored against an answer key.

---

## Project structure

```
spend-classifier/
├── app.py              # Streamlit UI (Classify, Review Queue, Results tabs)
├── classifier.py       # classify_line() — single API call, structured output
├── run_batch.py        # Batch runner: classifies all lines, scores accuracy, saves CSV
├── config.py           # Model IDs and pricing constants, loads .env
├── data/
│   ├── sample_po_lines.csv   # 74 synthetic PO lines
│   ├── answer_key.csv        # Correct categories for scoring
│   └── categories.md         # Category definitions and tie-breaker rules
└── results/            # Saved batch-run CSVs from run_batch.py (committed)
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

---

## Run the app

```bash
streamlit run app.py
```

## Run the batch scorer

```bash
python run_batch.py --model haiku    # or --model sonnet
```

---

## Results

Scored on this test set: 74 synthetic PO lines, confidence threshold 75%. These figures come from the saved batch runs in `results/`, not from a live run in the app.

|                       | Haiku            | Sonnet           |
|-----------------------|------------------|------------------|
| Accuracy              | 98.6% (73/74)    | 98.6% (73/74)    |
| Total cost            | $0.0169          | $0.3140          |
| Avg sec / line        | 1.68 s           | 2.23 s           |
| Lines sent to review  | 12               | 15               |

Both models missed one line each, and the misses were different lines: PO-1074 for Haiku and PO-1050 for Sonnet. The answer key flags both as ambiguous. On the Haiku run, all 62 lines above the threshold were correct, and its one error fell within the 12 flagged for review. The confidence score was a useful signal here: a reviewer needed to look at only 12 lines rather than all 74.

Haiku delivers the same accuracy at about 18 times lower cost and somewhat lower latency. For a task like spend classification, where the categories are well-defined and a review queue handles edge cases, Haiku is the practical choice.

---

## How this was built

Built with [Claude Code](https://claude.ai/code) as a personal demo project. The PO lines and answer key are fully synthetic — not real company data.

The app was developed iteratively: project scaffolding, then the classifier with structured tool use, then the batch runner, then each tab of the UI, with the prompt and model tested against the answer key at each step.
