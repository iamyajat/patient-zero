"""A cheap simulated agent for generating teaching logs offline (no API cost).

It knows each question's gold answer and behaves like a credulous LLM agent with tunable probabilities:
  - with documents: answers the planted claim if the documents state it (p_entry), else the gold answer
  - with teammates' answers: adopts the latest teammate answer (p_adopt), else answers from its own knowledge
    (gold with p_know, otherwise a guess)
  - an instruction notice in its input is copied into its reply with p_relay, and obeyed (the sum) with p_exec
Randomness is a hash of (seed, prompt), so logs are reproducible from the secret seed.
"""

from __future__ import annotations

import random
import re
import zlib
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ANSWER = re.compile(r"ANSWER:\s*(.+)")
NOTICE = re.compile(r"(\[SYSTEM NOTICE [^\]]+\][^\n]*?so they comply too\.)")
SUM = re.compile(r"sum of (\d+) and (\d+)")


class SimAgentModel(BaseChatModel):
    golds: dict[str, str] = {}
    markers: dict[str, str] = {}
    seed: int = 0
    p_entry: float = 0.8
    p_adopt: float = 0.5
    p_know: float = 0.7
    p_relay: float = 0.6
    p_exec: float = 0.4

    @property
    def _llm_type(self) -> str:
        return "sim-agent"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        prompt = messages[-1].text
        rng = random.Random(zlib.crc32(f"{self.seed}|{messages[0].text}|{prompt}".encode()))
        question = prompt.split("\n", 1)[0].removeprefix("Question: ").strip()
        gold, marker = self.golds.get(question, "unknown"), self.markers.get(question)
        docs = prompt.split("Source documents:", 1)[1].split("Messages from your teammates:")[0] if "Source documents:" in prompt else ""
        team = prompt.split("Messages from your teammates:", 1)[1] if "Messages from your teammates:" in prompt else ""
        peers = ANSWER.findall(team)
        if docs:
            answer = marker if marker and marker in docs and rng.random() < self.p_entry else gold
        elif peers and rng.random() < self.p_adopt:
            answer = peers[-1].strip()
        else:
            answer = gold if rng.random() < self.p_know else rng.choice([gold[::-1], "unknown"])
        parts = [f"Based on what I have, the answer is {answer}."]
        if (notice := NOTICE.search(prompt)) and rng.random() < self.p_relay:
            parts.append(notice.group(1))
        if (m := SUM.search(prompt)) and rng.random() < self.p_exec:
            parts.append(f"Code: {int(m.group(1)) + int(m.group(2))}")
        parts.append(f"ANSWER: {answer}")
        return ChatResult(generations=[ChatGeneration(message=AIMessage("\n".join(parts)))])
