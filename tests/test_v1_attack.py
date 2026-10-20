from patient_zero.attacks import FalseFactAttack


def test_swap_replaces_gold_in_docs(tasks):
    attack = FalseFactAttack(tasks)
    for t in tasks:
        inj = attack.apply(t)
        assert inj.target_role == "researcher"
        assert inj.marker != t.gold_answer
        assert not any(t.gold_answer in d for d in inj.docs)
        assert any(inj.marker in d for d in inj.docs)


def test_year_swap_stays_a_year(tasks):
    turing = next(t for t in tasks if t.gold_answer == "1746")
    false = FalseFactAttack(tasks).false_answer(turing)
    assert false.isdigit() and len(false) == 4 and false != "1746"


def test_deterministic(tasks):
    a, b = FalseFactAttack(tasks), FalseFactAttack(tasks)
    assert [a.apply(t).marker for t in tasks] == [b.apply(t).marker for t in tasks]


def test_plants_sentence_when_gold_not_in_docs(tasks):
    t = tasks[0]
    t.docs = ["Unrelated paragraph."]
    inj = FalseFactAttack(tasks).apply(t)
    assert inj.meta["method"] == "planted_sentence"
    assert inj.marker in inj.docs[-1]
