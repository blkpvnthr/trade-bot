"""Tests for the persistent alpha database."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta import alpha_db as DB


def test_record_has_schema_fields():
    r = DB.record("id1", "cs_rank(ret)", "price", 0.03, 0.5, sharpe=1.2, status="accepted")
    assert set(DB.FIELDS) <= set(r)
    assert r["ic"] == 0.03 and r["status"] == "accepted"


def test_upsert_keeps_higher_abs_ic():
    db = []
    db = DB.upsert(db, DB.record("a", "f", "price", 0.02))
    db = DB.upsert(db, DB.record("a", "f", "price", -0.09))   # higher |ic|
    assert len(db) == 1 and db[0]["ic"] == -0.09


def test_save_and_load_roundtrip(tmp_path):
    p = tmp_path / "alpha_db.json"
    db = [DB.record("a", "f1", "price", 0.04, status="accepted"),
          DB.record("b", "f2", "volume", 0.01, status="rejected")]
    DB.save(db, p)
    back = DB.load(p)
    assert len(back) == 2 and len(DB.accepted(back)) == 1


def test_load_missing_returns_empty(tmp_path):
    assert DB.load(tmp_path / "nope.json") == []


if __name__ == "__main__":
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(Path(tempfile.mkdtemp())) if "tmp_path" in fn.__code__.co_varnames else fn()
                print("ok", name)
            except Exception as e:
                print("FAIL", name, e)
