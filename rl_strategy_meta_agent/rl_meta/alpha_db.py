"""Persistent alpha database (Phase 1 of the AlphaEvolve workflow).

A flat JSON list of alpha records. Each record is the schema from the brief:
  alpha_id, formula, category, ic, ic_ir, sharpe, corr_to_existing,
  effective_breadth, oos_score, creation_source, status
Status transitions: candidate -> accepted | rejected. Research-only.
"""
from __future__ import annotations

import json
from pathlib import Path

FIELDS = ["alpha_id", "formula", "category", "ic", "ic_ir", "sharpe",
          "corr_to_existing", "effective_breadth", "oos_score",
          "creation_source", "status"]


def record(alpha_id, formula, category, ic, ic_ir=None, sharpe=None,
           corr_to_existing=None, effective_breadth=None, oos_score=None,
           creation_source="random", status="candidate") -> dict:
    return {"alpha_id": alpha_id, "formula": formula, "category": category,
            "ic": round(float(ic), 5), "ic_ir": _r(ic_ir), "sharpe": _r(sharpe),
            "corr_to_existing": _r(corr_to_existing),
            "effective_breadth": _r(effective_breadth), "oos_score": _r(oos_score),
            "creation_source": creation_source, "status": status}


def _r(x):
    return None if x is None else round(float(x), 5)


def load(path) -> list:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else []


def save(records, path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, default=str))


def upsert(records, rec) -> list:
    """Replace by alpha_id (keep the higher |ic| on conflict), else append."""
    by_id = {r["alpha_id"]: r for r in records}
    cur = by_id.get(rec["alpha_id"])
    if cur is None or abs(rec["ic"]) > abs(cur["ic"]):
        by_id[rec["alpha_id"]] = rec
    return list(by_id.values())


def accepted(records) -> list:
    return [r for r in records if r["status"] == "accepted"]


def _demo():
    db = []
    r1 = record("a1", "cs_rank(ret)", "price", 0.03, 0.5, status="accepted")
    r2 = record("a1", "cs_rank(ret)", "price", 0.05, 0.6)        # same id, higher ic
    db = upsert(db, r1); db = upsert(db, r2)
    assert len(db) == 1 and db[0]["ic"] == 0.05
    assert set(FIELDS) <= set(db[0])
    db = upsert(db, record("a2", "cs_zscore(volume)", "volume", 0.02, status="accepted"))
    assert len(accepted(db)) == 1            # a1 kept candidate status on the higher-ic overwrite
    print("alpha_db._demo OK", len(db), "records")


if __name__ == "__main__":
    _demo()
