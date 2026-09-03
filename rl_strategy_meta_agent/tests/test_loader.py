from pathlib import Path

from rl_meta.registry import load_strategies, parse_pine, file_hash, discover

PINE = Path(__file__).resolve().parent.parent / "pinescripts"


def test_discovers_sample():
    files = discover(PINE)
    assert any(f.name == "sample_strategy.txt" for f in files)


def test_parse_sample_metadata():
    text = (PINE / "sample_strategy.txt").read_text()
    p = parse_pine(text)
    assert p["strategy_name"] == "SMA Crossover Sample"
    assert p["inputs"]["fast"] == "10"
    assert p["inputs"]["slow"] == "30"
    assert "sma" in p["indicators"]
    assert "crossover" in p["indicators"]
    assert p["timeframe_hints"] == ["D"]
    assert p["entry_logic"], "should detect entry lines"
    assert p["stop_loss"], "should detect stop-loss lines"
    assert p["take_profit"], "should detect take-profit lines"


def test_hash_is_stable_and_version_sensitive():
    a = "strategy('A')\nx = 1\n"
    assert file_hash(a) == file_hash(a)
    assert file_hash(a) != file_hash(a + "// changed\n")


def test_load_strategies_assigns_ids():
    metas = load_strategies(PINE)
    assert metas, "should load at least the sample"
    ids = [m.strategy_id for m in metas]
    assert len(ids) == len(set(ids)), "strategy ids must be unique"
    for m in metas:
        assert m.version_hash and m.date_added and m.source_file
