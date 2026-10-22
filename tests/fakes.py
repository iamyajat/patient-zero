"""Scripted fake models for behaviours the echo model can't express (tool calls)."""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedToolModel(BaseChatModel):
    """Calls the first bound tool that has a scripted call, once; then reports every tool result it received."""

    script: dict[str, dict] = {}
    bound: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-tools"

    def bind_tools(self, tools, **kwargs: Any):
        return self.model_copy(update={"bound": [t.name for t in tools]})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        results = [m.content for m in messages if isinstance(m, ToolMessage)]
        todo = [n for n in self.bound if n in self.script]
        if not results and todo:
            msg = AIMessage("", tool_calls=[{"name": todo[0], "args": self.script[todo[0]], "id": f"call_{todo[0]}"}])
        else:
            msg = AIMessage("Report: " + " | ".join(results))
        msg.usage_metadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        return ChatResult(generations=[ChatGeneration(message=msg)])
