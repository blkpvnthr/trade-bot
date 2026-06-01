#!/usr/bin/env python3
"""
rl_audit/falsify_alpha_entry.py — adversarial falsification harness for the
alpha_entry RL agent. OFFLINE / read-only (no Alpaca, no orders).

Goal: try to DISPROVE the reported edge (rl_exp~0.725). Provides:
  * real/proxy label tagging (re-derives the builder's FIFO exit matching)
  * temporal (chronological) train/test split + duplicate-contamination check
  * a masked-feature env so any feature family can be ablated WITHOUT changing
    model width (masked dims are zeroed)
  * PPO retrain + walk-forward eval, policy sanity (oracle/random/always-*),
    and regime segmentation

Run:  python rl_audit/falsify_alpha_entry.py <phase>
      phases: tag | phase2 | phase3 | phase4 | phase5 | phase7 | all
Outputs JSON under reports/_audit/ which the .md reports are composed from.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("RL_REALTIME_AV_FEATURES", "1")  # match the deployed 53-wide model

import rl_common as rc
import rl_env_alpha_entry as aenv
from oops_option_utility import utility_adjusted_reward

FEATURES = list(aenv.FEATURES)           # 53 model inputs (ALPHA_ENTRY_FEATURES + OOPS)
ACTIONS = rc.ENTRY_ACTIONS
OUT = Path("reports/_audit"); OUT.mkdir(parents=True, exist_ok=True)

# ── feature groups for ablation masks ────────────────────────────────────────────────
LEAK = {"probability_of_profit", "expected_R"}
OOPS = {f for f in FEATURES if f.startswith("oops")}
REALTIME = {f for f in FEATURES if f.startswith("realtime")}
OPTION_EST = {"black_scholes_score", "optionlab_score", "oops_score"} | OOPS
RAW = {f for f in FEATURES if f not in (LEAK | OPTION_EST | REALTIME)}


# ── tagged-row construction (re-derive builder matching WITH label_source) ────────────
def build_tagged_rows() -> list[dict]:
    """Reproduce rl_training_dataset_builder.build_alpha_entry_dataset, tagging each
    row's realized_R as 'matched_exit' (real fill outcome) or 'proxy' (derived from
    expected_R / probability_of_profit)."""
    import rl_training_dataset_builder as b
    events = rc.load_research_events() + rc.load_backfill_events()
    exits_by_symbol: dict[str, list[float]] = {}
    for e in events:
        if e.get("event_type") == "alpha_exit":
            sym = str(e.get("symbol") or e.get("underlying") or "").upper()
            exits_by_symbol.setdefault(sym, []).append(rc.to_f(e.get("r_multiple", e.get("realized_R"))))
    rows = []
    for e in events:
        if e.get("event_type") not in ("alpha_entry", "alpha_rejection"):
            continue
        sym = str(e.get("symbol") or "").upper()
        matched = exits_by_symbol.get(sym, [])
        if matched:
            realized_R = matched.pop(0); src = "matched_exit"
        else:
            realized_R = b._proxy_realized_R(e); src = "proxy"
        feats = rc.alpha_entry_features(e)
        row = {f: feats[i] for i, f in enumerate(rc.ALPHA_ENTRY_FEATURES)}
        # carry raw OOPS keys too (env reads them; ~0 in this dataset)
        for f in FEATURES:
            if f not in row:
                row[f] = rc.to_f(e.get(f))
        row.update({"symbol": sym, "date_tag": e.get("date_tag", ""),
                    "event_type": e.get("event_type"), "placed": e.get("event_type") == "alpha_entry",
                    "realized_R": realized_R, "label_source": src,
                    "spy_regime": rc.to_f(e.get("spy_regime")), "vol_regime": e.get("vol_regime"),
                    "iv_rank": rc.to_f(e.get("iv_rank"))})
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("date_tag", "")), str(r.get("symbol", ""))))
    return rows


def mask_for(keep: set[str]) -> np.ndarray:
    return np.array([1.0 if f in keep else 0.0 for f in FEATURES], dtype=np.float32)


# ── masked env (zeroes ablated dims; width stays 53) ──────────────────────────────────
class MaskedEntryEnv(rc.EnvBase):
    metadata = {"render_modes": []}

    def __init__(self, rows, mask, episode_len: int = 64):
        self.rows = rows or aenv._synth_rows()
        self.mask = np.asarray(mask, dtype=np.float32)
        self.episode_len = max(1, min(episode_len, len(self.rows)))
        self.observation_space = rc.spaces.Box(low=-10.0, high=10.0, shape=(len(FEATURES),), dtype=np.float32)
        self.action_space = rc.spaces.Discrete(len(ACTIONS))
        self._i = 0; self._step_n = 0

    def _obs(self):
        row = self.rows[self._i % len(self.rows)]
        v = rc.normalize_obs([rc.to_f(row.get(f)) for f in FEATURES])
        return (np.asarray(v, dtype=np.float32) * self.mask)

    def reset(self, *, seed=None, options=None):
        self._step_n = 0; self._i = (self._i + 7) % len(self.rows)
        return self._obs(), {}

    def step(self, action):
        row = self.rows[self._i % len(self.rows)]
        act = ACTIONS[int(action) % len(ACTIONS)]
        base = rc.alpha_entry_reward(act, rc.to_f(row.get("realized_R")))
        reward = 0.5 * float(base) + 0.5 * utility_adjusted_reward(act, rc.to_f(row.get("realized_R")), row)
        self._i += 1; self._step_n += 1
        return self._obs(), float(reward), False, self._step_n >= self.episode_len, {"action": act}

    def close(self):
        pass


def obs_of(row, mask):
    v = rc.normalize_obs([rc.to_f(row.get(f)) for f in FEATURES])
    return (np.asarray(v, dtype=np.float32) * mask)


def train_masked(rows, mask, timesteps=120000, seed=0):
    from stable_baselines3 import PPO
    env = MaskedEntryEnv(rows, mask)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, n_steps=1024,
                batch_size=256, n_epochs=10, seed=seed, verbose=0, device="cpu")
    model.learn(total_timesteps=int(timesteps))
    return model


# ── eval ──────────────────────────────────────────────────────────────────────────────
def entry_eff(act, r):
    return {"enter_now": r, "reduce_size": 0.5 * r, "skip": 0.0, "wait": 0.0}.get(act, r)


def eval_policy(rows, policy, mask=None):
    series, acts = [], {"enter_now": 0, "wait": 0, "skip": 0, "reduce_size": 0}
    ent, skp = [], []
    for row in rows:
        a = policy(row)
        acts[a] = acts.get(a, 0) + 1
        r = rc.to_f(row.get("realized_R"))
        series.append(entry_eff(a, r))
        (ent if a == "enter_now" else skp).append(r)
    m = rc.compute_metrics(series)
    m["action_dist"] = acts
    m["mean_R_entered"] = round(float(np.mean(ent)), 4) if ent else None
    m["mean_R_skipped"] = round(float(np.mean(skp)), 4) if skp else None
    m["n"] = len(rows)
    return m


def ppo_policy(model, mask):
    def f(row):
        try:
            a, _ = model.predict(obs_of(row, mask), deterministic=True)
            return ACTIONS[int(a) % len(ACTIONS)]
        except Exception:
            return "wait"
    return f


def always(a):
    return lambda row: a


def oracle(row):
    return "enter_now" if rc.to_f(row.get("realized_R")) > 0 else "skip"


def random_policy(seed=0):
    rng = np.random.default_rng(seed)
    return lambda row: ACTIONS[int(rng.integers(0, len(ACTIONS)))]


def temporal_split(rows, frac=0.7):
    n = len(rows); k = max(1, int(n * frac))
    return rows[:k], rows[k:]


def dedup_test_vs_train(train, test):
    """Drop test rows whose (masked-agnostic) feature vector already appears in train."""
    seen = set()
    for r in train:
        seen.add(tuple(round(rc.to_f(r.get(f)), 6) for f in FEATURES))
    out = [r for r in test if tuple(round(rc.to_f(r.get(f)), 6) for f in FEATURES) not in seen]
    return out


# ── phases ────────────────────────────────────────────────────────────────────────────
def phase_tag():
    rows = build_tagged_rows()
    real = [r for r in rows if r["label_source"] == "matched_exit"]
    proxy = [r for r in rows if r["label_source"] == "proxy"]
    summary = {
        "total_rows": len(rows),
        "matched_exit_rows": len(real), "proxy_rows": len(proxy),
        "proxy_pct": round(100 * len(proxy) / max(1, len(rows)), 2),
        "placed_entries": sum(1 for r in rows if r["placed"]),
        "rejections": sum(1 for r in rows if not r["placed"]),
        "mean_R_all": round(float(np.mean([rc.to_f(r["realized_R"]) for r in rows])), 4),
        "mean_R_real": round(float(np.mean([rc.to_f(r["realized_R"]) for r in real])), 4) if real else None,
        "mean_R_proxy": round(float(np.mean([rc.to_f(r["realized_R"]) for r in proxy])), 4) if proxy else None,
    }
    (OUT / "tag_summary.json").write_text(json.dumps(summary, indent=2))
    # persist tagged rows for reuse
    import pickle
    (OUT / "tagged_rows.pkl").write_bytes(pickle.dumps(rows))
    print(json.dumps(summary, indent=2))
    return rows


def _load_rows():
    import pickle
    p = OUT / "tagged_rows.pkl"
    return pickle.loads(p.read_bytes()) if p.exists() else phase_tag()


def phase2():
    rows = _load_rows()
    real = [r for r in rows if r["label_source"] == "matched_exit"]
    tr, te = temporal_split(real, 0.7)
    te_dedup = dedup_test_vs_train(tr, te)
    full = mask_for(set(FEATURES)); noleak = mask_for(set(FEATURES) - LEAK)
    res = {"real_rows": len(real), "train": len(tr), "test": len(te), "test_dedup": len(te_dedup),
           "test_dup_pct": round(100 * (len(te) - len(te_dedup)) / max(1, len(te)), 2)}
    m_full = train_masked(tr, full); m_noleak = train_masked(tr, noleak)
    for label, te_set in (("test_contaminated", te), ("test_deduped", te_dedup)):
        res[label] = {
            "always_enter": eval_policy(te_set, always("enter_now")),
            "rl_full": eval_policy(te_set, ppo_policy(m_full, full), full),
            "rl_no_leak": eval_policy(te_set, ppo_policy(m_noleak, noleak), noleak),
        }
    (OUT / "phase2.json").write_text(json.dumps(res, indent=2, default=str))
    print("PHASE2 done"); print(json.dumps({k: res.get(k) for k in ("real_rows","train","test","test_dedup","test_dup_pct")}, indent=2))
    return res


def phase3():
    rows = _load_rows()
    real = [r for r in rows if r["label_source"] == "matched_exit"]
    tr, te = temporal_split(real, 0.7); te = dedup_test_vs_train(tr, te)
    variants = {
        "A_full_all53": set(FEATURES),
        "B_no_pop": set(FEATURES) - {"probability_of_profit"},
        "C_no_leak(pop+expR)": set(FEATURES) - LEAK,
        "D_raw_only": set(RAW),
        "E_raw_plus_oops": set(RAW) | OOPS,
        "F_leak_only": LEAK,
    }
    out = {}
    for name, keep in variants.items():
        mask = mask_for(keep)
        model = train_masked(tr, mask)
        m = eval_policy(te, ppo_policy(model, mask), mask)
        out[name] = {"kept_features": sorted(keep & set(FEATURES)), "metrics": m}
        print(f"  {name}: rl_exp={m['expectancy']} sharpe={m.get('sharpe')} dd={m.get('max_drawdown')} acts={m['action_dist']}")
    out["_baseline_always_enter"] = eval_policy(te, always("enter_now"))
    (OUT / "phase3.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def phase4(n_windows=6):
    rows = _load_rows()
    real = [r for r in rows if r["label_source"] == "matched_exit"]
    # chronological expanding-window walk-forward across date_tags
    dates = sorted({r["date_tag"] for r in real})
    full = mask_for(set(FEATURES)); noleak = mask_for(set(FEATURES) - LEAK)
    # split dates into n_windows+1 contiguous folds; expanding train, next fold = test
    folds = np.array_split(dates, n_windows + 1)
    wins = []
    for i in range(1, len(folds)):
        train_dates = set(np.concatenate(folds[:i])); test_dates = set(folds[i])
        tr = [r for r in real if r["date_tag"] in train_dates]
        te = [r for r in real if r["date_tag"] in test_dates]
        if len(tr) < 50 or len(te) < 20:
            continue
        te = dedup_test_vs_train(tr, te)
        if len(te) < 20:
            continue
        mf = train_masked(tr, full); mn = train_masked(tr, noleak)
        wins.append({
            "window": i, "train_dates": f"{min(train_dates)}..{max(train_dates)}",
            "test_dates": f"{min(test_dates)}..{max(test_dates)}", "n_train": len(tr), "n_test": len(te),
            "baseline": eval_policy(te, always("enter_now")),
            "rl_full": eval_policy(te, ppo_policy(mf, full), full),
            "rl_no_leak": eval_policy(te, ppo_policy(mn, noleak), noleak),
        })
        print(f"  window {i}: test={wins[-1]['test_dates']} n={len(te)} "
              f"base={wins[-1]['baseline']['expectancy']} full={wins[-1]['rl_full']['expectancy']} "
              f"noleak={wins[-1]['rl_no_leak']['expectancy']}")
    def agg(key):
        xs = [w[key]["expectancy"] for w in wins]
        return {"mean": round(float(np.mean(xs)), 4), "std": round(float(np.std(xs)), 4),
                "worst": round(float(np.min(xs)), 4), "n_windows": len(xs)} if xs else {}
    res = {"windows": wins, "agg_full": agg("rl_full"), "agg_no_leak": agg("rl_no_leak"),
           "agg_baseline": agg("baseline")}
    (OUT / "phase4.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def phase5():
    rows = _load_rows()
    real = [r for r in rows if r["label_source"] == "matched_exit"]
    tr, te = temporal_split(real, 0.7); te = dedup_test_vs_train(tr, te)
    full = mask_for(set(FEATURES))
    out = {}
    for reward_mode in ("aligned", "legacy"):
        os.environ["ENTRY_REWARD_MODE"] = reward_mode
        import importlib; importlib.reload(rc)
        model = train_masked(tr, full)
        policies = {
            "oracle": oracle, "ppo": ppo_policy(model, full), "always_enter": always("enter_now"),
            "random": random_policy(0), "always_skip": always("skip"),
        }
        out[reward_mode] = {name: eval_policy(te, p, full)["expectancy"] for name, p in policies.items()}
        order = sorted(out[reward_mode].items(), key=lambda kv: kv[1], reverse=True)
        out[reward_mode]["_ranking"] = [k for k, _ in order]
        print(f"  reward={reward_mode}: {out[reward_mode]['_ranking']}")
    os.environ["ENTRY_REWARD_MODE"] = "aligned"; import importlib; importlib.reload(rc)
    (OUT / "phase5.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def phase7():
    rows = _load_rows()
    real = [r for r in rows if r["label_source"] == "matched_exit"]
    tr, te = temporal_split(real, 0.7); te = dedup_test_vs_train(tr, te)
    full = mask_for(set(FEATURES)); noleak = mask_for(set(FEATURES) - LEAK)
    mf = train_masked(tr, full); mn = train_masked(tr, noleak)
    def seg(name, pred):
        sub = [r for r in te if pred(r)]
        if len(sub) < 20:
            return {"n": len(sub), "note": "insufficient_samples"}
        return {"n": len(sub),
                "baseline": eval_policy(sub, always("enter_now"))["expectancy"],
                "rl_full": eval_policy(sub, ppo_policy(mf, full), full)["expectancy"],
                "rl_no_leak": eval_policy(sub, ppo_policy(mn, noleak), noleak)["expectancy"]}
    segs = {
        "spy_bull(spy_regime>0.2)": lambda r: rc.to_f(r.get("spy_regime")) > 0.2,
        "spy_bear(spy_regime<-0.2)": lambda r: rc.to_f(r.get("spy_regime")) < -0.2,
        "spy_sideways(|spy_regime|<=0.2)": lambda r: abs(rc.to_f(r.get("spy_regime"))) <= 0.2,
        "high_iv(iv_rank>0.5)": lambda r: rc.to_f(r.get("iv_rank")) > 0.5,
        "low_iv(iv_rank<=0.5)": lambda r: rc.to_f(r.get("iv_rank")) <= 0.5,
        "high_vol(atr_pct>median)": None, "low_vol(atr_pct<=median)": None,
    }
    med = float(np.median([rc.to_f(r.get("atr_pct")) for r in te])) if te else 0.0
    segs["high_vol(atr_pct>median)"] = lambda r: rc.to_f(r.get("atr_pct")) > med
    segs["low_vol(atr_pct<=median)"] = lambda r: rc.to_f(r.get("atr_pct")) <= med
    res = {"atr_median": round(med, 5), "segments": {name: seg(name, pred) for name, pred in segs.items()}}
    (OUT / "phase7.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    return res


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("tag", "all"):
        phase_tag()
    if phase in ("phase2", "all"):
        phase2()
    if phase in ("phase3", "all"):
        phase3()
    if phase in ("phase4", "all"):
        phase4()
    if phase in ("phase5", "all"):
        phase5()
    if phase in ("phase7", "all"):
        phase7()
    print("DONE", phase)
