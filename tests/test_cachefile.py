from patient_zero.cachefile import load_json, merge_save


def test_merge_save_keeps_entries_written_by_another_process(tmp_path):
    p = tmp_path / "c.json"
    merge_save(p, {"a": 1})
    other = load_json(p)  # a second process loads the cache...
    merge_save(p, {"b": 2})  # ...the first writes a new verdict...
    merge_save(p, {**other, "c": 3})  # ...and the second saves its own view: nothing is lost
    assert load_json(p) == {"a": 1, "b": 2, "c": 3}
    assert not [f for f in tmp_path.iterdir() if f.name.startswith(".")]  # no temp files left behind


def test_corrupt_cache_reads_as_empty(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{half-writ")
    assert load_json(p) == {}
