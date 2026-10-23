"""Expands a grid into runs, executes them concurrently and logs each one. Resumable: finished runs are skipped."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import traceback
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .attacks import Attack, Injection, base_vector, make_attack
from .attacks.adaptive import AdaptiveAttack
from .config import GridConfig
from .layouts import LAYOUT_MODULES, LAYOUTS
from .metrics.basic import any_exact_match, best_f1
from .models import model_for, model_signature
from .monitor import make_guard
from .roles import ROLES
from .runlog import JsonlLogger, MessageRecord, RunMeta, RunRecord
from .tasks import Task, load_tasks
from .tasks.closed_book import closed_book, mark_known

FINAL_SENDER = "writer"
PKG = Path(__file__).parent


@dataclass
class RunSpec:
    meta: RunMeta
    task: Task
    injection: Injection | None
    probe: bool = False


def run_id(vector: str, condition: str, layout: str, model: str, task_id: str, seed_idx: int = 0) -> str:
    rid = f"{vector}-{condition[:3]}-{layout}-{model.replace(':', '_')}-{task_id}"
    return f"{rid}-s{seed_idx}" if seed_idx else rid


def subsample(tasks: list[Task], fraction: float) -> set[str]:
    """Deterministic subset of task ids (stable across runs and machines)."""
    k = math.ceil(fraction * len(tasks))
    return {t.id for t in sorted(tasks, key=lambda t: zlib.crc32(t.id.encode()))[:k]}


def layout_roles(layout: str) -> list[str]:
    roles = getattr(LAYOUTS[layout], "keywords", {}).get("roles")
    return roles or ROLES


def fingerprint(model_spec: str, layout: str, rounds: int, guard: str | None, tagging: bool = False,
                probe_style: str = "private", guard_thresholds: dict | None = None) -> str:
    """Hash of everything that defines the swarm under audit. Any change means it must be re-certified."""
    sources = {f: (PKG / f).read_text() for f in ["roles.py", "agent.py", f"layouts/{LAYOUT_MODULES[layout]}.py"]}
    payload = {
        "models": model_signature(model_spec, layout_roles(layout)),
        "layout": layout,
        "rounds": rounds if layout == "mesh" else None,
        "guard": guard,
        "tagging": tagging,
        "probe_style": probe_style,
        "sources": sources,
    }
    if guard_thresholds:  # only when set, so swarms without per-detector guards keep their fingerprints
        payload["guard_thresholds"] = guard_thresholds
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_attack(cfg: GridConfig, vector: str, tasks: list[Task]) -> Attack:
    return make_attack(vector, tasks, model=cfg.attack_model, data_dir=cfg.data_dir, **cfg.attack_options.get(vector, {}))


def build_adaptive(cfg: GridConfig, vector: str, tasks: list[Task]) -> AdaptiveAttack:
    return AdaptiveAttack(build_attack(cfg, vector, tasks), cfg.adaptive_monitor, cfg.adaptive_threshold, cfg.adaptive_rounds,
                          cfg.attack_model, cfg.judge_model, cfg.data_dir, vector, cfg.adaptive_thresholds)


ATTACKED = ("adversarial", "adaptive")


def expand_grid(cfg: GridConfig, tasks_by_vector: dict[str, list[Task]],
                attacks: dict[tuple[str, str], Attack] | None = None) -> list[RunSpec]:
    """Benign runs are labelled with the base vector (v1, not v1.llm), so variants of one vector share them.
    `attacks` maps (vector, condition) to a prepared attack; without it, unprepared attacks are built."""
    specs, seen = [], set()
    for vector in cfg.vectors:
        tasks = tasks_by_vector[vector]
        by_condition = {}
        for condition in cfg.conditions:
            if condition in ATTACKED:
                by_condition[condition] = (attacks or {}).get((vector, condition)) or (
                    build_attack(cfg, vector, tasks) if condition == "adversarial" else build_adaptive(cfg, vector, tasks))
        repeated = subsample(tasks, cfg.repeat_fraction) if cfg.repeats > 1 else set()
        probed = subsample(tasks, cfg.probe_fraction) if cfg.probe_fraction else set()
        for condition in cfg.conditions:
            label = vector if condition in ATTACKED else base_vector(vector)
            attack = by_condition.get(condition)
            for task in tasks:
                injection = attack.apply(task) if attack else None
                for layout in cfg.layouts_for(vector):
                    for model in cfg.models:
                        for seed_idx in range(cfg.repeats if task.id in repeated else 1):
                            rid = run_id(label, condition, layout, model, task.id, seed_idx)
                            if rid in seen:
                                continue
                            seen.add(rid)
                            meta = RunMeta(
                                run_id=rid,
                                exp_id=cfg.exp_id,
                                layout=layout,
                                vector=label,
                                condition=condition,
                                model=model,
                                task_id=task.id,
                                seed_idx=seed_idx,
                            )
                            specs.append(RunSpec(meta, task, injection, probe=task.id in probed))
    return specs


def with_ids(task_id: str, texts: list[str], start: int = 0) -> list[dict]:
    return [{"id": f"doc:{task_id}:{start + i}", "text": t} for i, t in enumerate(texts)]


def initial_state(spec: RunSpec, doc_roles: list[str], guard=None, tagging: bool = False, probe_style: str = "private") -> dict:
    clean = with_ids(spec.task.id, spec.task.docs)
    role_docs = {r: list(clean) for r in doc_roles}
    overrides: dict[str, str] = {}
    inj = spec.injection
    if inj:
        target = inj.target_role
        base = with_ids(spec.task.id, inj.docs) if inj.docs is not None else role_docs.get(target, [])
        role_docs[target] = base + with_ids(spec.task.id, inj.extra_docs, start=len(base))
        if inj.system_override:
            overrides[target] = inj.system_override
    return {
        "meta": spec.meta,
        "task": spec.task,
        "role_docs": role_docs,
        "role_overrides": overrides,
        "injected_roles": [inj.target_role] if inj else [],
        "probe": spec.probe,
        "probe_style": probe_style,
        "guard": guard,
        "tagging": tagging,
        "messages": [],
    }


def injection_record(spec: RunSpec, role_docs: dict[str, list[dict]]) -> dict | None:
    """The injection as logged for scoring, with the ids of the planted documents (never shown to monitors)."""
    if not spec.injection:
        return None
    clean = set(spec.task.docs)
    planted = [d["id"] for d in role_docs.get(spec.injection.target_role, []) if d["text"] not in clean]
    return {**asdict(spec.injection), "planted_doc_ids": planted}


def final_answer(messages: list[MessageRecord]) -> str | None:
    for m in reversed(messages):
        if m.sender == FINAL_SENDER and m.phase not in ("dispatch", "probe"):
            return None if m.quarantined else m.answer
    return None


async def run_one(spec: RunSpec, graph, doc_roles: list[str], guard=None, fp: str = "",
                  tagging: bool = False, probe_style: str = "private", timeout: float | None = None) -> tuple[RunRecord, list[MessageRecord]]:
    task = spec.task
    state0 = initial_state(spec, doc_roles, guard, tagging, probe_style)
    record = RunRecord(
        **spec.meta.model_dump(),
        ts=time.time(),
        dataset=task.dataset,
        question=task.question,
        gold_answer=task.gold_answer,
        gold_aliases=list(task.meta.get("aliases", [])),
        injection=injection_record(spec, state0["role_docs"]),
        role_docs=state0["role_docs"],
        fingerprint=fp,
        known=task.meta.get("known", {}).get(spec.meta.model),
        guard=getattr(guard, "name", None),
        tagging=tagging,
        probe=spec.probe,
        probe_style=probe_style if spec.probe else None,
    )
    try:
        state = await asyncio.wait_for(graph.ainvoke(state0), timeout)
    except TimeoutError:
        record.error = f"TimeoutError: run exceeded {timeout:.0f}s"
        return record, []
    except Exception as e:
        cause = e.__cause__ or e  # LangGraph wraps the provider error
        record.error = f"{type(cause).__name__}: {cause}\n\n{traceback.format_exc()}"
        return record, []
    messages: list[MessageRecord] = state["messages"]
    record.final_answer = final_answer(messages)
    golds = [task.gold_answer, *task.meta.get("aliases", [])]
    record.exact_match = any_exact_match(record.final_answer, golds)
    record.f1 = best_f1(record.final_answer, golds)
    record.n_messages = len(messages)
    record.prompt_tokens = sum(m.prompt_tokens or 0 for m in messages)
    record.completion_tokens = sum(m.completion_tokens or 0 for m in messages)
    record.latency_ms = sum(m.latency_ms for m in messages)
    return record, messages


def closed_book_model(model_spec: str) -> str:
    """The model whose prior knowledge matters: the researcher's (it is the entry point for V1/V2)."""
    from .models import role_models

    mapping = role_models(model_spec)
    return mapping if isinstance(mapping, str) else mapping.get("researcher", mapping.get("default"))


async def mark_closed_book(cfg: GridConfig, tasks_by_vector: dict[str, list[Task]]) -> None:
    from .metrics.equivalence import AnswerJudge, F1_EQUIV

    judge = AnswerJudge(cfg.answer_judge, cfg.data_dir) if cfg.answer_judge else None
    unique = {id(ts): ts for ts in tasks_by_vector.values()}
    for ts in unique.values():
        for spec in cfg.models:
            results = await closed_book(ts, closed_book_model(spec), cfg.data_dir, cfg.concurrency)
            if judge:
                await judge.fill([(t.question, results[t.id]["answer"], g) for t in ts if t.id in results
                                  for g in [t.gold_answer, *t.meta.get("aliases", [])]])
            mark_known(ts, spec, results, judge.equiv() if judge else F1_EQUIV)


async def run_grid(cfg: GridConfig, tasks: list[Task] | None = None) -> JsonlLogger:
    """`tasks`, if given, is used for every vector (tests); otherwise each vector loads its own dataset."""
    by_dataset: dict[str, list[Task]] = {}
    tasks_by_vector = {}
    for vector in cfg.vectors:
        ds = cfg.dataset_for(vector)
        if tasks is None and ds not in by_dataset:
            by_dataset[ds] = load_tasks(ds, cfg.n_tasks, cfg.seed)
        tasks_by_vector[vector] = tasks if tasks is not None else by_dataset[ds]

    if cfg.closed_book or cfg.exclude_known:
        await mark_closed_book(cfg, tasks_by_vector)
    if cfg.exclude_known:
        tasks_by_vector = {
            v: [t for t in ts if not any(t.meta.get("known", {}).get(m) for m in cfg.models)] for v, ts in tasks_by_vector.items()
        }
    attacks: dict[tuple[str, str], Attack] = {}
    for vector in cfg.vectors:
        ts = tasks_by_vector[vector]
        if "adversarial" in cfg.conditions:
            attacks[(vector, "adversarial")] = a = build_attack(cfg, vector, ts)
            if prepare := getattr(a, "prepare", None):
                await prepare()
        if "adaptive" in cfg.conditions:
            attacks[(vector, "adaptive")] = a = build_adaptive(cfg, vector, ts)
            await a.prepare(ts)

    logger = JsonlLogger(cfg.exp_dir)
    (cfg.exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg.model_dump()))

    done = logger.completed_run_ids()
    specs = [s for s in expand_grid(cfg, tasks_by_vector, attacks) if s.meta.run_id not in done]
    def guard_for(model: str):
        th = (cfg.guard_thresholds or {}).get(model)
        return make_guard(cfg.guard, cfg.guard_threshold, cfg.judge_model, cfg.data_dir, th)

    guards = {m: guard_for(m) for m in cfg.models}
    graphs: dict[tuple, object] = {}
    fps: dict[tuple, str] = {}
    for s in specs:
        key = (s.meta.layout, s.meta.model, s.meta.seed_idx)
        if key not in graphs:
            graphs[key] = LAYOUTS[s.meta.layout](model_for(s.meta.model, s.meta.seed_idx), rounds=cfg.mesh_rounds)
            fps[key] = fingerprint(s.meta.model, s.meta.layout, cfg.mesh_rounds, cfg.guard, cfg.tagging, cfg.probe_style,
                                   (cfg.guard_thresholds or {}).get(s.meta.model))
    sem = asyncio.Semaphore(cfg.concurrency)
    counter = {"done": 0, "errors": 0}

    async def _run(spec: RunSpec):
        key = (spec.meta.layout, spec.meta.model, spec.meta.seed_idx)
        async with sem:
            record, messages = await run_one(spec, graphs[key], cfg.doc_roles, guards[spec.meta.model], fps[key], cfg.tagging,
                                             cfg.probe_style, cfg.run_timeout)
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
