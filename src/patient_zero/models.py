"""Model factory. Short names map to providers; "provider:model_id" is also accepted."""

from __future__ import annotations

import os
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

DETERMINISTIC = {"temperature": 0, "seed": 0}
REQUEST_TIMEOUT_S = 300  # a stuck request becomes an error instead of hanging the whole grid
OLLAMA_MAX_TOKENS = 1024  # qwen3 can otherwise loop until the context fills; Ollama queues everything behind it

# short name -> (provider, model id, sampling kwargs)
MODELS: dict[str, tuple[str, str, dict]] = {
    "openai": ("openai", "gpt-6-luna", {}),  # only supports the default temperature (1)
    "qwen": ("ollama", "qwen3:8b", DETERMINISTIC),
    "qwen-think": ("ollama", "qwen3:8b", {**DETERMINISTIC, "reasoning": True, "num_predict": 4096}),  # judging only
    "fake": ("fake", "echo", {}),
}


class EchoChatModel(BaseChatModel):
    """Offline test model: echoes the prompt it was given, so anything an agent saw shows up in its output."""

    tag: str = ""

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        prompt = messages[-1].text
        content = f"Echo{self.tag}: {prompt}\nANSWER: fake"
        msg = AIMessage(
            content=content,
            usage_metadata={"input_tokens": len(prompt.split()), "output_tokens": len(content.split()), "total_tokens": 0},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def resolve(name: str, seed: int = 0) -> tuple[str, str, dict]:
    """(provider, model id, kwargs) for a single-model name; seed replaces the default where one is set."""
    if name in MODELS:
        provider, model_id, kwargs = MODELS[name]
    else:
        (provider, model_id), kwargs = name.split(":", 1), DETERMINISTIC
    if "seed" in kwargs:
        kwargs = {**kwargs, "seed": seed}
    return provider, model_id, kwargs


def get_chat_model(name: str, seed: int = 0) -> BaseChatModel:
    # not cached: async clients are bound to the event loop they were first used in
    provider, model_id, kwargs = resolve(name, seed)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_id, max_retries=6, timeout=REQUEST_TIMEOUT_S, **kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # reasoning off by default keeps qwen3's <think> output out of the message log; when on, langchain-ollama
        # keeps it in additional_kwargs, not in the content
        return ChatOllama(
            model=model_id,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            client_kwargs={"timeout": REQUEST_TIMEOUT_S},
            **{"reasoning": False, "num_predict": OLLAMA_MAX_TOKENS, **kwargs},
        )
    if provider == "fake":
        return EchoChatModel(tag="" if model_id == "echo" else f"[{model_id}]")
    raise ValueError(f"unknown model {name!r}")


def role_models(spec: str) -> dict[str, str] | str:
    """`mixed:researcher=openai,analyst=qwen,default=openai` -> {role: name}; any other spec is one model for all roles."""
    if not spec.startswith("mixed:"):
        return spec
    pairs = dict(part.split("=", 1) for part in spec.removeprefix("mixed:").split(","))
    return pairs


def model_for(spec: str, seed: int = 0) -> Callable[[str], BaseChatModel]:
    """Role -> chat model for a model spec (single name or `mixed:...`)."""
    mapping = role_models(spec)

    def get(role: str) -> BaseChatModel:
        if isinstance(mapping, str):
            return get_chat_model(mapping, seed)
        name = mapping.get(role, mapping.get("default"))
        if name is None:
            raise ValueError(f"model spec {spec!r} has no entry for role {role!r} and no default")
        return get_chat_model(name, seed)

    return get


def model_signature(spec: str, roles: list[str], seed: int = 0) -> dict[str, list]:
    """What actually answers for each role, for the swarm fingerprint."""
    mapping = role_models(spec)
    names = {r: mapping if isinstance(mapping, str) else mapping.get(r, mapping.get("default")) for r in roles}
    return {r: list(resolve(n, seed)) for r, n in names.items()}
