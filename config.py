from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Model IDs
HAIKU_MODEL = "claude-haiku-5-5"
SONNET_MODEL = "claude-sonnet-5-5"

# Pricing per million tokens (fill in actual values)
HAIKU_INPUT_PRICE_PER_M = 0.10   # $ per 1M input tokens
HAIKU_OUTPUT_PRICE_PER_M = 0.50  # $ per 1M output tokens

SONNET_INPUT_PRICE_PER_M = 2.00  # $ per 1M input tokens
SONNET_OUTPUT_PRICE_PER_M = 10.00 # $ per 1M output tokens
