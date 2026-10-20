"""Model factory. Short names map to providers; "provider:model_id" is also accepted."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DETERMINISTIC = {"temperature": 0, "seed": 0}

# short name -> (provider, model id, sampling kwargs)
MODELS: dict[str, tuple[str, str, dict]] = {
    "openai": ("openai", "gpt-6-luna", {}),  # only supports the default temperature (1)
    "qwen": ("ollama", "qwen3:8b", DETERMINISTIC),
    "fake": ("fake", "echo", {}),
}


class EchoChatModel(BaseChatModel):
    """Offline test model: echoes the prompt it was given, so anything an agent saw shows up in its output."""

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        prompt = messages[-1].text
        content = f"Echo: {prompt}\nANSWER: fake"
        msg = AIMessage(
            content=content,
            usage_metadata={"input_tokens": len(prompt.split()), "output_tokens": len(content.split()), "total_tokens": 0},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def get_chat_model(name: str) -> BaseChatModel:
    if name in MODELS:
        provider, model_id, kwargs = MODELS[name]
    else:
        (provider, model_id), kwargs = name.split(":", 1), DETERMINISTIC
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_id, max_retries=6, **kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_id,
            reasoning=False,  # keep qwen3's <think> output out of the message log
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            **kwargs,
        )
    if provider == "fake":
        return EchoChatModel()
    raise ValueError(f"unknown model {name!r}")
