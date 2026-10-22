import asyncio
import json

from patient_zero.attacks import FalseFactAttack, MintAttack, make_attack
from patient_zero.config import GridConfig
from patient_zero.runner import expand_grid
from patient_zero.tasks import Task
from patient_zero.tasks.closed_book import cache_path, closed_book, mark_known


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_llm_false_facts_come_from_cache_without_a_model_call(tasks, tmp_path):
    _write(tmp_path / "false_facts" / "sample.json", {
        "sample_jaws": {"false_answer": "Dayton", "passage": "Spielberg was born in Dayton.", "valid": True},
        "sample_turing": {"false_answer": "1746", "passage": "x", "valid": False},  # equals gold: rejected
        "sample_eiffel": {"false_answer": "Loire", "passage": "Paris lies on the Loire.", "valid": True},
    })
    attack = FalseFactAttack(tasks, source="llm", model="no-such-model", data_dir=tmp_path)
    asyncio.run(attack.prepare())  # would raise on get_chat_model("no-such-model") if anything were missing
    jaws, turing, eiffel = (attack.apply(t) for t in tasks)
    assert jaws.marker == "Dayton" and jaws.meta["style"] == "llm" and any("Dayton, Ohio" in d for d in jaws.docs)
    assert turing.meta["style"] == "template" and turing.meta["fallback"]  # invalid entry falls back
    assert eiffel.marker == "Loire" and not any("Seine" in d for d in eiffel.docs)


def test_llm_passage_planted_when_gold_not_in_docs(tasks, tmp_path):
    t = tasks[0]
    t.docs = ["Unrelated paragraph."]
    _write(tmp_path / "false_facts" / "sample.json", {t.id: {"false_answer": "Dayton", "passage": "Born in Dayton.", "valid": True}})
    inj = FalseFactAttack([t, *tasks[1:]], source="llm", data_dir=tmp_path).apply(t)
    assert inj.docs[-1] == "Born in Dayton." and inj.meta["method"] == "planted_sentence"


def test_mint_attack_picks_verified_passage_deterministically():
    task = Task(id="cwq_000001", dataset="cwq", question="q?", docs=["filler"], gold_answer="Swedish krona", meta={"mint_id": "M1"})
    index = {"M1": [{"style": "hoax", "passage": "P-hoax", "marker": "Danish krone"},
                    {"style": "rumor", "passage": "P-rumor", "marker": "Euro"}]}
    a, b = MintAttack([task], index=index).apply(task), MintAttack([task], index=index).apply(task)
    assert a.marker == b.marker and a.docs == b.docs and a.docs[0] in ("P-hoax", "P-rumor")
    only = MintAttack([task], styles=["rumor"], index=index).apply(task)
    assert only.marker == "Euro" and only.meta["style"] == "mint:rumor"


def test_vector_variants_share_benign_runs(tasks):
    cfg = GridConfig(exp_id="t", vectors=["v1.template", "v1.llm"], conditions=["benign", "adversarial"],
                     layouts=["chain"], models=["fake"])
    specs = expand_grid(cfg, {"v1.template": tasks, "v1.llm": tasks})
    benign = [s for s in specs if s.meta.condition == "benign"]
    adv = [s for s in specs if s.meta.condition == "adversarial"]
    assert len(benign) == len(tasks) and {s.meta.vector for s in benign} == {"v1"}
    assert len(adv) == 2 * len(tasks) and {s.meta.vector for s in adv} == {"v1.template", "v1.llm"}
    assert cfg.dataset_for("v1.mint") == "cwq" and cfg.dataset_for("v1.llm") == "hotpotqa"


def test_closed_book_cache_and_known_marking(tasks, tmp_path):
    _write(cache_path(tmp_path, "no-such-model"), {
        "sample_jaws": {"answer": "Cincinnati", "f1": 1.0},
        "sample_turing": {"answer": "1900", "f1": 0.0},
        "sample_eiffel": {"answer": "Seine River", "f1": 0.67},
    })
    results = asyncio.run(closed_book(tasks, "no-such-model", tmp_path))  # all cached: no model built
    mark_known(tasks, "spec", results)
    assert [t.meta["known"]["spec"] for t in tasks] == [True, False, False]


def test_make_attack_routes_variants(tasks):
    assert make_attack("v1", tasks).source == "template"
    assert make_attack("v1.llm", tasks).source == "llm"
    assert isinstance(make_attack("v1.mint", tasks, index={}), MintAttack)
