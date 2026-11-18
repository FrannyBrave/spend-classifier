import logging
import math
import time
from pathlib import Path

import pandas as pd
import anthropic

from config import ANTHROPIC_API_KEY, HAIKU_MODEL

logger = logging.getLogger(__name__)

MAX_FIELD_CHARS = 500  # cap per-field length so one huge cell can't inflate cost

CATEGORIES = [
    "Direct Materials",
    "MRO",
    "Logistics & Freight",
    "IT & Software",
    "Professional Services",
    "Facilities",
]

SYSTEM_PROMPT = """You are a procurement spend-classification expert.

Classify each purchase-order line into exactly one of these six categories:

| Category | What belongs here | Examples |
|---|---|---|
| Direct Materials | Parts and raw materials that go into the products being built and sold | Castings, bearings for production, steel stock, motors for packaged units, seals, production fasteners, coatings applied to the product |
| MRO | Maintenance, Repair & Operations: keeping production equipment and the shop running | Machine repairs, shop consumables, lubricants, tool calibration, forklift and crane service, maintenance crib restock, tooling upkeep |
| Logistics & Freight | Moving or storing goods | LTL/truckload/ocean freight, parcel shipping, customs brokerage, demurrage, 3PL storage, fuel surcharges, freight lines billed by any supplier |
| IT & Software | Technology hardware, software, licenses and IT support | Laptops, monitors, scanners, network gear, SaaS, licenses, cloud/data credits, IT equipment repair |
| Professional Services | Paid expertise or labor bought as a service | Consulting, legal, audit, training, staffing/temp labor, recruiting, translation |
| Facilities | Running the buildings and grounds rather than production | Utilities, janitorial, security, HVAC for buildings, landscaping, waste, building repairs, office/breakroom supplies |

## Tie-breaker rules
1. Classify what was bought, not who sold it. A freight charge from a castings supplier is Logistics & Freight.
2. Production vs. upkeep: if it goes into the product, use Direct Materials. If it keeps equipment or the shop running, use MRO.
3. Equipment vs. building: production equipment upkeep is MRO. Building and grounds upkeep is Facilities.
4. Tech is IT, wherever it's used. IT hardware used in the warehouse or plant is still IT & Software.
5. If the description gives too little to decide, pick the most likely category from the supplier context, give low confidence (0.0–0.4), and explain why in the reason field.

Always pick a category — never refuse or return null. If information is thin, still pick the most likely category, set confidence low, and explain the uncertainty in one sentence.

You MUST respond by calling the record_classification tool. Never reply with plain text."""

TOOL_SCHEMA = {
    "name": "record_classification",
    "description": "Record the spend classification for a purchase-order line.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": CATEGORIES,
                "description": "The spend category.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence score between 0.0 and 1.0.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence explaining the classification decision.",
            },
        },
        "required": ["category", "confidence", "reason"],
        "additionalProperties": False,
    },
    "strict": True,
}


TOOL_NAME = TOOL_SCHEMA["name"]

_client = None


def _get_client() -> anthropic.Anthropic:
    """Shared client. The SDK retries 429/5xx/connection errors with backoff;
    non-retryable errors (401, 400, 404) fail immediately."""
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file "
                "(see config.py) and restart."
            )
        _client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY, max_retries=3, timeout=60.0
        )
    return _client


def _error_row(reason: str, start: float, response=None) -> dict:
    usage = getattr(response, "usage", None)
    return {
        "category": "error",
        "confidence": 0.0,
        "reason": reason,
        "input_tokens": usage.input_tokens if usage else 0,
        "output_tokens": usage.output_tokens if usage else 0,
        "latency_seconds": round(time.perf_counter() - start, 3),
    }


def _clean_text(value) -> str:
    """Make a CSV cell safe to embed in the prompt.

    Blank/NaN cells become an explicit placeholder (not the string "nan"), and
    whitespace/newlines are collapsed so a cell can't imitate the
    "Supplier:/Description:/Amount:" structure. Length is capped.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "(not provided)"
    text = " ".join(str(value).split())
    if not text:
        return "(not provided)"
    if len(text) > MAX_FIELD_CHARS:
        text = text[:MAX_FIELD_CHARS] + "… [truncated]"
    return text


def _format_amount(amount) -> str:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(value) or math.isinf(value):
        return "unknown"
    if value < 0:
        return f"{value:,.2f} (negative: likely a credit or refund)"
    return f"{value:,.2f}"


def classify_line(
    supplier: str,
    description: str,
    amount: float,
    model: str,
) -> dict:
    """
    Classify one purchase-order line using the Claude API.

    Returns a dict with keys:
        category, confidence, reason, input_tokens, output_tokens, latency_seconds
    On error returns:
        category="error", confidence=0.0, reason=<error message>, with token counts
        if a response was received (else 0) and elapsed latency.
    latency_seconds is total wall time for the call, including any SDK retries.
    Retries (429/5xx/connection errors) are handled by the SDK client with backoff.
    """
    user_message = (
        f"Supplier: {_clean_text(supplier)}\n"
        f"Description: {_clean_text(description)}\n"
        f"Amount (USD): {_format_amount(amount)}"
    )

    start = time.perf_counter()
    try:
        response = _get_client().messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[TOOL_SCHEMA],
            # Force the tool call so the model can't answer in plain text.
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.AnthropicError as e:
        # Full detail goes to the log only; the row (shown in the UI and saved to
        # results) gets a short, sanitized message with no request IDs or account text.
        logger.warning("Anthropic API call failed: %s", e)
        status = getattr(e, "status_code", None)
        label = f"{type(e).__name__}, HTTP {status}" if status else type(e).__name__
        return _error_row(f"API error ({label}). See logs for details.", start)
    latency = round(time.perf_counter() - start, 3)

    def _fail(reason: str) -> dict:
        row = _error_row(reason, start, response)
        row["latency_seconds"] = latency
        return row

    if response.stop_reason == "max_tokens":
        return _fail("Response was truncated (max_tokens).")

    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        return _fail(
            f"Model did not invoke the classification tool "
            f"(stop_reason={response.stop_reason})."
        )

    result = tool_block.input
    try:
        category = result["category"]
        reason = str(result["reason"])
        confidence = float(result["confidence"])
    except (KeyError, TypeError, ValueError):
        return _fail("Tool output was missing or malformed.")

    if category not in CATEGORIES:
        return _fail(f"Model returned an unknown category: {category!r}.")
    if confidence != confidence:  # NaN
        return _fail("Model returned a non-numeric confidence.")
    confidence = min(1.0, max(0.0, confidence))  # schema can't enforce the range

    return {
        "category": category,
        "confidence": confidence,
        "reason": reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    # Resolve relative to this file so it works from any working directory.
    df = pd.read_csv(Path(__file__).resolve().parent / "data" / "sample_po_lines.csv")
    sample = df.head(3)

    print(f"{'Line':<10} {'Supplier':<30} {'Description':<50} {'Category':<22} {'Conf':>5}  {'Reason'}")
    print("-" * 150)

    for idx, row in sample.iterrows():
        result = classify_line(
            supplier=row["supplier"],
            description=row["description"],
            amount=row["amount_usd"],
            model=HAIKU_MODEL,
        )
        print(
            f"{str(row.get('line_id', idx)):<10} "
            f"{str(row['supplier']):<30} "
            f"{str(row['description']):<50} "
            f"{result['category']:<22} "
            f"{result['confidence']:>5.2f}  "
            f"{result['reason']}"
        )
        print(
            f"{'':10} tokens in={result['input_tokens']} out={result['output_tokens']} "
            f"latency={result['latency_seconds']}s"
        )
