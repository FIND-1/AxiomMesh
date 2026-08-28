"""Configuration for the LLM Council."""

import os

from dotenv import load_dotenv

from .llm.registry import ModelSpec, list_models_by_name, resolve_model

load_dotenv()

ModelConfig = ModelSpec


# Backward-compatible fallback: the existing .env uses OPENROUTER_API_KEY for
# an OpenAI key. Prefer OPENAI_API_KEY when present, otherwise reuse that value.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
QWEN_API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")

OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/responses")
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions"
)
GEMINI_API_URL = os.getenv(
    "GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta"
)
KIMI_API_URL = os.getenv(
    "KIMI_API_URL",
    os.getenv("KIMI_BASE_URL", "https://api.kimi.com/coding/v1/chat/completions"),
)
QWEN_API_URL = os.getenv(
    "QWEN_API_URL",
    os.getenv(
        "DASHSCOPE_API_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    ),
)

MODEL_REGISTRY = list_models_by_name()

# Council members.
COUNCIL_MODELS = [
    resolve_model("deepseek"),
    resolve_model("kimi"),
    resolve_model("gemini"),
    resolve_model("qwen"),
]

# GPT is kept in the registry, but disabled from the active council for now.

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = resolve_model("gemini")

# Reuse a fast default model for title generation.
TITLE_MODEL = resolve_model("gemini")

# Data directory for conversation storage
DATA_DIR = "data/conversations"
