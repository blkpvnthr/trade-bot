import json

from rl_meta.registry import load_strategies, write_registry, read_registry, PINE_DIR


def test_registry_roundtrip(tmp_path):
    metas = load_strategies(PINE_DIR)
    path = tmp_path / "strategy_registry.json"
    write_registry(metas, path)

    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    assert payload["count"] == len(metas)
    assert payload["strategies"]

    back = read_registry(path)
    assert len(back) == len(metas)
    assert back[0].strategy_id == metas[0].strategy_id
    assert back[0].version_hash == metas[0].version_hash


def test_registry_records_required_fields(tmp_path):
    metas = load_strategies(PINE_DIR)
    path = tmp_path / "r.json"
    write_registry(metas, path)
    s = json.loads(path.read_text())["strategies"][0]
    for field in ("strategy_id", "strategy_name", "version_hash",
                  "date_added", "status"):
        assert field in s
