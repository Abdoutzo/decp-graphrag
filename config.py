"""Central configuration. Env-overridable, see .env.example."""
import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = _env("DATA_PATH", os.path.join(BASE_DIR, "data", "contracts.json"))
GRAPH_PATH = _env("GRAPH_PATH", os.path.join(BASE_DIR, "data", "graph.json"))

LLM_PROVIDER = _env("LLM_PROVIDER", "mistral")  # mistral | openai | none
MISTRAL_API_KEY = _env("MISTRAL_API_KEY", "")
MISTRAL_MODEL = _env("MISTRAL_MODEL", "mistral-small-latest")
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")

TOKEN_PRICES = {
    "mistral-small-latest": {"input": 0.0001, "output": 0.0003},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Louvain community detection
LOUVAIN_SEED = 42

# TF-IDF baseline retriever
TFIDF_TOP_K = 5

# Local search: entity neighbourhood depth
LOCAL_DEPTH = 2
