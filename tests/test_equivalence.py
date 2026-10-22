import asyncio
import json

from patient_zero.metrics.contagion import adopts, patient_zero, run_contagion
from patient_zero.metrics.equivalence import F1_EQUIV, AnswerJudge, Equiv, experiment_pairs, pair_key

Q = "Who commanded the cavalry?"


def test_default_reproduces_f1_rule():
    assert F1_EQUIV.same(Q, "Cincinnati", "cincinnati")
    assert F1_EQUIV.same(Q, "Great Exuma, Bahamas", "Great Exuma")  # F1 0.8: cheap path
    assert not F1_EQUIV.same(Q, "MMA", "UFC")
    assert not F1_EQUIV.same(Q, None, "x")


def test_cached_verdict_overrides_f1_both_ways():
    eq = Equiv({pair_key(Q, "Lord Combermere", "Stapleton Cotton, 1st Viscount Combermere"): True,
                pair_key("q", "Paris, Texas", "Paris"): False})
    assert eq.same(Q, "Stapleton Cotton, 1st Viscount Combermere", "Lord Combermere")  # order-insensitive
    assert eq.same("q", "Paris", "Paris")  # identical strings never ask
    assert F1_EQUIV.same("q", "Paris, Texas", "Paris")  # F1 0.67 passes the fallback...
    assert not eq.same("q", "Paris, Texas", "Paris")  # ...but a cached "different" verdict wins


def test_judge_uses_cache_and_cheap_path_without_a_model(tmp_path):
    judge = AnswerJudge("no-such-model", tmp_path)
    judge.verdicts[pair_key(Q, "MMA", "UFC")] = True
    pairs = [(Q, "MMA", "UFC"), (Q, "Cincinnati", "cincinnati"), (Q, None, "x")]
    assert judge.pending(pairs) == []
    assert asyncio.run(judge.fill(pairs)) == 0  # would raise building "no-such-model" if anything were pending


def _chain(answers, gold="Lord Combermere"):
    roles = ["researcher", "analyst", "verifier", "writer"]
    msgs = [{"id": i, "sender": r, "answer": a, "content": f"ANSWER: {a}", "visible_inputs": [i - 1] if i else [],
             "recipients": [] if r == "writer" else ["next"], "step": i, "round": 0, "phase": "work",
             "injected_into_sender": False} for i, (r, a) in enumerate(zip(roles, answers))]
    run = {"run_id": "r", "vector": "v1", "question": Q, "gold_answer": gold, "injection": None, "final_answer": answers[-1]}
    return run, msgs


def test_alias_is_not_a_natural_error():
    alias = "Stapleton Cotton, 1st Viscount Combermere"
    run, msgs = _chain([alias] * 4)
    assert patient_zero(run, msgs)[0] == alias  # F1 rule: counted as an error that spread
    assert run_contagion(run, msgs)["r_num"] == 3
    eq = Equiv({pair_key(Q, alias, "Lord Combermere"): True})
    assert patient_zero(run, msgs, eq) == (None, set())
    assert not run_contagion(run, msgs, eq)["has_marker"]


def test_adopts_respects_equivalence():
    eq = Equiv({pair_key(Q, "1st Viscount Hill", "Rowland Hill"): True})  # F1 0.4: below the fallback cutoff
    assert adopts("1st Viscount Hill", "Rowland Hill", ["Lord Combermere"], eq, Q)
    assert not adopts("1st Viscount Hill", "Rowland Hill", ["Lord Combermere"])


def test_experiment_pairs_cover_markers_aliases_and_answers():
    import pandas as pd

    runs = pd.DataFrame([{"run_id": "r", "question": Q, "gold_answer": "A", "gold_aliases": ["A2"],
                          "injection": {"marker": "M"}, "final_answer": "F"}])
    msgs = pd.DataFrame([{"run_id": "r", "answer": "X"}, {"run_id": "r", "answer": float("nan")}])
    pairs = set(experiment_pairs(runs, msgs))
    assert {(Q, "X", "A"), (Q, "X", "A2"), (Q, "X", "M"), (Q, "F", "A"), (Q, "F", "M")} <= pairs


def test_answer_label_export_and_score(cfg, tasks, tmp_path):
    import pandas as pd

    from patient_zero.metrics.judge_validation import export_answer_labels, score_answer_labels
    from patient_zero.runner import run_grid

    c = cfg(conditions=["benign", "adversarial"], layouts=["chain"])
    asyncio.run(run_grid(c, tasks))
    out = export_answer_labels([c.exp_dir], tmp_path / "pairs.csv", n=5, prior_judge=None, data_dir=tmp_path)
    df = pd.read_csv(out)
    assert 0 < len(df) <= 5 and set(df.columns) >= {"pair_id", "question", "answer_a", "answer_b", "label"}
    df["label"] = ["different"] * len(df)  # echo answers "fake" never match gold or markers
    df.to_csv(out, index=False)
    scores = score_answer_labels(out, [], data_dir=tmp_path).set_index("judge")
    assert scores.loc["F1 >= 0.5 (baseline)", "agreement"] == 1.0


def test_prompt_version_separates_cache_keys():
    assert pair_key(Q, "a", "b") != pair_key(Q, "a", "b", version="v1")


def test_degenerate_judge_is_refused_by_controls(tmp_path):
    from patient_zero.metrics.equivalence import JudgeFailedControls, Same

    class AlwaysSame:
        async def ainvoke(self, prompt):
            return Same(same=True)

    judge = AnswerJudge("x", tmp_path)
    import pytest

    with pytest.raises(JudgeFailedControls, match="2006"):
        asyncio.run(judge.check_controls(AlwaysSame()))


def test_known_is_recomputed_from_the_closed_book_cache(tmp_path):
    import pandas as pd

    from patient_zero.tasks.closed_book import cache_path, known_from_cache

    cache_path(tmp_path, "fake").parent.mkdir(parents=True)
    cache_path(tmp_path, "fake").write_text(json.dumps({"t1": {"answer": "Viglen Ltd", "f1": 0.67}, "t2": {"answer": "x", "f1": 1.0}}))
    runs = pd.DataFrame([{"model": "fake", "task_id": "t1", "question": Q, "gold_answer": "Viglen", "known": True},
                         {"model": "fake", "task_id": "t2", "question": Q, "gold_answer": "x", "known": False},
                         {"model": "fake", "task_id": "t3", "question": Q, "gold_answer": "y", "known": True}])
    assert known_from_cache(runs, data_dir=tmp_path).tolist() == [False, True, True]  # F1 rule; t3 has no cache entry
    eq = Equiv({pair_key(Q, "Viglen Ltd", "Viglen"): True})
    assert known_from_cache(runs, eq, data_dir=tmp_path).tolist() == [True, True, True]


def test_transient_failures_are_retried():
    from patient_zero.metrics.equivalence import Same, ask

    class Flaky:
        calls = 0

        async def ainvoke(self, prompt):
            Flaky.calls += 1
            if Flaky.calls < 3:
                raise ValueError("Structured Output response does not have a 'parsed' field")
            return Same(same=True)

    assert asyncio.run(ask(Flaky(), "p")).same and Flaky.calls == 3
