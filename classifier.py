import time
import json
import pandas as pd
import anthropic

from config import ANTHROPIC_API_KEY, HAIKU_MODEL, SONNET_MODEL

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
    On error (after one retry) returns:
        category="error", confidence=0.0, reason=<error message>, tokens=0, latency=0.0
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_message = (
        f"Supplier: {supplier}\n"
        f"Description: {description}\n"
        f"Amount (USD): {amount}"
    )

    def _call() -> anthropic.types.Message:
        return client.messages.create(
            model=model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[TOOL_SCHEMA],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": user_message}],
        )

    start = time.perf_counter()
    try:
        response = _call()
    except anthropic.APIError:
        time.sleep(2)
        try:
            response = _call()
        except anthropic.APIError as e:
            return {
                "category": "error",
                "confidence": 0.0,
                "reason": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_seconds": round(time.perf_counter() - start, 3),
            }
    latency = round(time.perf_counter() - start, 3)

    # Extract tool-use result
    tool_block = next(
        (b for b in response.content if b.type == "tool_use"), None
    )
    if tool_block is None:
        return {
            "category": "error",
            "confidence": 0.0,
            "reason": "Model did not invoke the classification tool.",
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "latency_seconds": latency,
        }

    result = tool_block.input
    return {
        "category": result["category"],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_seconds": latency,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/sample_po_lines.csv")
    sample = df.head(3)

    print(f"{'PO':<10} {'Supplier':<30} {'Description':<50} {'Category':<22} {'Conf':>5}  {'Reason'}")
    print("-" * 150)

    for _, row in sample.iterrows():
        result = classify_line(
            supplier=row["supplier"],
            description=row["description"],
            amount=row["amount_usd"],
            model=HAIKU_MODEL,
        )
        print(
            f"{row['line_id']:<10} "
            f"{row['supplier']:<30} "
            f"{str(row['description']):<50} "
            f"{result['category']:<22} "
            f"{result['confidence']:>5.2f}  "
            f"{result['reason']}"
        )
        print(
            f"{'':10} tokens in={result['input_tokens']} out={result['output_tokens']} "
            f"latency={result['latency_seconds']}s"
        )
