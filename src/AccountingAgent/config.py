import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
API_KEY: str = os.getenv("API_KEY", "")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

AGENT_MAX_ITERATIONS: int = int(os.getenv("AGENT_MAX_ITERATIONS", "30"))
AGENT_TIMEOUT_SECONDS: float = float(os.getenv("AGENT_TIMEOUT_SECONDS", "270"))
TRIPLETEX_REQUEST_TIMEOUT: float = float(os.getenv("TRIPLETEX_REQUEST_TIMEOUT", "30"))
