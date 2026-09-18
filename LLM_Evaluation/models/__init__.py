from .base import LLMProvider, empty_result
from .claude import ClaudeProvider
from .qwen import QwenProvider
from .sarvam import SarvamProvider

PROVIDER_CLASSES = {
    "sarvam": SarvamProvider,
    "qwen": QwenProvider,
    "claude": ClaudeProvider,
}


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    PROVIDER_CLASSES[name] = cls


def build_provider(name: str, **kwargs) -> LLMProvider:
    if name not in PROVIDER_CLASSES:
        raise KeyError(f"Unknown provider '{name}'. Register it in models.PROVIDER_CLASSES.")
    return PROVIDER_CLASSES[name](**kwargs)
