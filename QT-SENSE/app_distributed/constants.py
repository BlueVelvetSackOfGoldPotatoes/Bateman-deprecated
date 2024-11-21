import os

# Semantic Scholar API (no API key needed, but you can set a rate limit if desired)
SEMANTIC_SCHOLAR_RATE_LIMIT = 5  # Number of requests per second

# Model configuration
GPT_MODEL = "gpt-4o-mini"

# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")

# Set SerpAPI Key
# SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "65a4ab6977ea5aa74cb95a5bf2a01df6c811d1a45784b2ba77a242341f0456e5")

# Set Serper API key
SERPER_API_KEY = os.getenv("SERPER_API", "8737c516a0b54a948a09f868cf1a9c38dd12991e")