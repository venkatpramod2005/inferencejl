import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    hf_token: str | None
    use_4bit: bool
    max_new_tokens: int
    temperature: float
    top_p: float
    require_cuda: bool

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(
            model_id=os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3"),
            hf_token=os.getenv("HF_TOKEN") or None,
            use_4bit=env_bool("USE_4BIT", False),
            max_new_tokens=env_int("MAX_NEW_TOKENS", 256),
            temperature=env_float("TEMPERATURE", 0.7),
            top_p=env_float("TOP_P", 0.9),
            require_cuda=env_bool("REQUIRE_CUDA", True),
        )
