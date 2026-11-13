# Spend Classifier

A Streamlit app that uses the Claude API to automatically classify procurement purchase-order lines into spend categories.

![Screenshot placeholder](docs/screenshot.png)

---

## Why

Categorizing PO lines is one of the most repetitive tasks in procurement. Analysts spend hours manually assigning spend categories to hundreds or thousands of lines so the business can answer basic questions like "how much did we spend on logistics last quarter?" This app automates that classification with an LLM and routes low-confidence lines to a human review queue, combining speed with a lightweight quality-control step.

---

## What it does

- **Classify tab** — upload a CSV of PO lines (or use the built-in sample), choose a model, and run batch classification. Each line gets a category, a confidence score, a one-sentence reason, and token/latency metadata. Results download as CSV.
- **Review Queue tab** — set a confidence threshold (default 75%). Lines below it are queued for human review: approve the AI's category or override it from a dropdown. Decisions are tracked and fed into the accuracy calculation.
- **Results tab** — accuracy of the AI alone and after human review, broken down above and below the confidence threshold, a confusion matrix, and a side-by-side model comparison.

**Six spend categories:** Direct Materials · MRO · Logistics & Freight · IT & Software · Professional Services · Facilities

---

## How it works

1. Each PO line (supplier, description, amount) is sent to the Claude API with a system prompt containing the category definitions and tie-breaker rules.
2. The model responds by calling a structured tool (`record_classification`) that enforces a valid category, a 0–1 confidence score, and a reason. This guarantees well-formed output without text parsing.
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
└── results/            # Output CSVs from run_batch.py (git-ignored)
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

Scored against 74 synthetic PO lines with an answer key. Confidence threshold: 75%.

| | Haiku | Sonnet |
|---|---|---|
| Accuracy | — | — |
| Total cost | — | — |
| Avg sec / line | — | — |
| Lines sent to review | — | — |

*Replace the dashes with your numbers from `run_batch.py` output.*

---

## How this was built

Built with [Claude Code](https://claude.ai/code) as a personal demo project. The PO lines and answer key are fully synthetic — no real company or supplier data is used anywhere.

The app was developed iteratively: project scaffolding, then the classifier, then the batch runner, then each tab of the UI, with the model and prompt tested against the answer key at each step.
