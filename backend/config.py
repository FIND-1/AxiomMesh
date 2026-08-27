"""Configuration for the LLM Council."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """A single LLM configuration entry."""

    id: str
    provider: str
    api_model: str


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

MODEL_REGISTRY = {
    "openai": ModelConfig(
        id="openai/gpt-5-nano",
        provider="openai",
        api_model="gpt-5-nano",
    ),
    "deepseek": ModelConfig(
        id="deepseek/deepseek-v4-flash",
        provider="deepseek",
        api_model="deepseek-v4-flash",
    ),
    "gemini": ModelConfig(
        id="google/gemini-3.5-flash-lite",
        provider="gemini",
        api_model="gemini-3.5-flash-lite",
    ),
    "kimi": ModelConfig(
        id="moonshot/kimi-k2.7-code",
        provider="kimi",
        api_model="kimi-for-coding",
    ),
    "qwen": ModelConfig(
        id="qwen/qwen3-235b-a22b-instruct-2507",
        provider="qwen",
        api_model="qwen3-235b-a22b-instruct-2507",
    ),
}

# Council members.
COUNCIL_MODELS = [
    MODEL_REGISTRY["deepseek"],
    MODEL_REGISTRY["kimi"],
    MODEL_REGISTRY["gemini"],
    MODEL_REGISTRY["qwen"],
]

# GPT is kept in the registry, but disabled from the active council for now.

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = MODEL_REGISTRY["gemini"]

# Reuse a fast default model for title generation.
TITLE_MODEL = MODEL_REGISTRY["gemini"]

# Data directory for conversation storage
DATA_DIR = "data/conversations"
