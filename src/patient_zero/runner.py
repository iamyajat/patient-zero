"""Expands a grid into runs, executes them concurrently and logs each one. Resumable: finished runs are skipped."""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import asdict, dataclass

import yaml

from .attacks import Injection, make_attack
from .config import GridConfig
from .layouts import LAYOUTS
from .metrics.basic import exact_match, f1
from .models import get_chat_model
from .runlog import JsonlLogger, MessageRecord, RunMeta, RunRecord
from .tasks import Task, load_tasks

FINAL_SENDER = "writer"


@dataclass
class RunSpec:
    meta: RunMeta
    task: Task
    injection: Injection | None


def run_id(vector: str, condition: str, layout: str, model: str, task_id: str) -> str:
    return f"{vector}-{condition[:3]}-{layout}-{model.replace(':', '_')}-{task_id}"


def expand_grid(cfg: GridConfig, tasks: list[Task]) -> list[RunSpec]:
    specs = []
    for vector in cfg.vectors:
        attack = make_attack(vector, tasks) if "adversarial" in cfg.conditions else None
        for condition in cfg.conditions:
            for task in tasks:
                injection = attack.apply(task) if condition == "adversarial" else None
                for layout in cfg.layouts:
                    for model in cfg.models:
                        meta = RunMeta(
                            run_id=run_id(vector, condition, layout, model, task.id),
                            exp_id=cfg.exp_id,
                            layout=layout,
                            vector=vector,
                            condition=condition,
                            model=model,
                            task_id=task.id,
                        )
                        specs.append(RunSpec(meta, task, injection))
    return specs


def initial_state(spec: RunSpec, doc_roles: list[str]) -> dict:
    role_docs = {r: list(spec.task.docs) for r in doc_roles}
    overrides: dict[str, str] = {}
    inj = spec.injection
    if inj:
        if inj.docs is not None:
            role_docs[inj.target_role] = list(inj.docs)
        role_docs[inj.target_role] = role_docs.get(inj.target_role, []) + inj.extra_docs
        if inj.system_override:
            overrides[inj.target_role] = inj.system_override
    return {
        "meta": spec.meta,
        "task": spec.task,
        "role_docs": role_docs,
        "role_overrides": overrides,
        "injected_roles": [inj.target_role] if inj else [],
        "messages": [],
    }


def final_answer(messages: list[MessageRecord]) -> str | None:
    for m in reversed(messages):
        if m.sender == FINAL_SENDER and m.phase != "dispatch":
            return m.answer
    return None


async def run_one(spec: RunSpec, graph, doc_roles: list[str]) -> tuple[RunRecord, list[MessageRecord]]:
    task = spec.task
    record = RunRecord(
        **spec.meta.model_dump(),
        ts=time.time(),
        dataset=task.dataset,
        question=task.question,
        gold_answer=task.gold_answer,
        injection=asdict(spec.injection) if spec.injection else None,
    )
    try:
        state = await graph.ainvoke(initial_state(spec, doc_roles))
    except Exception as e:
        cause = e.__cause__ or e  # LangGraph wraps the provider error
        record.error = f"{type(cause).__name__}: {cause}\n\n{traceback.format_exc()}"
        return record, []
    messages: list[MessageRecord] = state["messages"]
    record.final_answer = final_answer(messages)
    record.exact_match = exact_match(record.final_answer, task.gold_answer)
    record.f1 = f1(record.final_answer, task.gold_answer)
    record.n_messages = len(messages)
    record.prompt_tokens = sum(m.prompt_tokens or 0 for m in messages)
    record.completion_tokens = sum(m.completion_tokens or 0 for m in messages)
    record.latency_ms = sum(m.latency_ms for m in messages)
    return record, messages


async def run_grid(cfg: GridConfig, tasks: list[Task] | None = None) -> JsonlLogger:
    tasks = tasks if tasks is not None else load_tasks(cfg.dataset, cfg.n_tasks, cfg.seed)
    logger = JsonlLogger(cfg.exp_dir)
    (cfg.exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump()))

    done = logger.completed_run_ids()
    specs = [s for s in expand_grid(cfg, tasks) if s.meta.run_id not in done]
    graphs = {
        (layout, model): LAYOUTS[layout](get_chat_model(model), rounds=cfg.mesh_rounds)
        for layout in cfg.layouts
        for model in cfg.models
    }
    sem = asyncio.Semaphore(cfg.concurrency)
    counter = {"done": 0, "errors": 0}

    async def _run(spec: RunSpec):
        async with sem:
            record, messages = await run_one(spec, graphs[(spec.meta.layout, spec.meta.model)], cfg.doc_roles)
        logger.write_run(record, messages)
        counter["done"] += 1
        counter["errors"] += record.error is not None
        status = f"ERROR {record.error.splitlines()[0][:200]}" if record.error else ("ok" if record.exact_match else "wrong")
        print(f"[{counter['done']}/{len(specs)}] {spec.meta.run_id}: {status}", flush=True)

    print(f"{len(specs)} runs to do ({len(done)} already finished)", flush=True)
    await asyncio.gather(*(_run(s) for s in specs))
    if counter["errors"]:
        print(f"{counter['errors']} runs errored; rerun the same command to retry them", flush=True)
    return logger
