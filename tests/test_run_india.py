"""Tests for run_india.py probes/helpers (no encoder pretrain - fast, CPU)."""

import numpy as np

from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
from run_india import _visible_features, build_rail_examples, demo, intake_eval


def _data():
    pay, evt, accs = build_dataset(IndiaConfig(num_accounts=300, num_payments=3000, seed=23))
    return pay, evt, build_schema(pay, accs)


def test_visible_features_shape_and_no_leak():
    pay, _, _ = _data()
    X = _visible_features(pay)
    assert X.shape == (len(pay), 6) and np.isfinite(X).all()
    # the helper must not read rail-derived columns
    src = _visible_features.__code__.co_consts
    assert "rail" not in src and "settlement_kind" not in src and "SttlmMtd" not in src


def test_intake_eval_keys_and_baselines():
    pay, _, schema = _data()
    rng = np.random.default_rng(0)
    # a weak but non-trivial representation: visible features + noise (so probes train)
    e = np.hstack([_visible_features(pay), rng.normal(size=(len(pay), 8))])
    idx = rng.permutation(len(pay)); cut = int(len(idx) * 0.8)
    out = intake_eval(e, pay, schema, idx[:cut], idx[cut:])
    assert {"rail_routing", "exception_pr_auc", "status", "eta"} <= set(out)
    r = out["rail_routing"]
    assert {"probe_accuracy", "tree_accuracy", "majority_baseline"} <= set(r)
    assert r["tree_accuracy"] >= r["majority_baseline"]            # tree learns something
    assert "limit_exceeded" in out["exception_pr_auc"] and "sla_breach" in out["exception_pr_auc"]
    assert out["eta"]["mae_min"] >= 0 and out["eta"]["baseline_mae_min"] >= 0


def test_demo_runs_and_clamps_eta():
    pay, _, schema = _data()
    rng = np.random.default_rng(1)
    e = np.hstack([_visible_features(pay), rng.normal(size=(len(pay), 8))])
    idx = rng.permutation(len(pay)); cut = int(len(idx) * 0.8)
    lines = []
    demo(e, pay, schema, idx[:cut], idx[cut:], log=lines.append)
    text = "\n".join(lines)
    assert "per-payment forecast" in text and "routing eligibility" in text
    # ETA is clamped to >= 0 in every forecast line
    etas = [int(s.split("ETA:")[1].split("min")[0]) for s in lines if "ETA:" in s]
    assert etas and all(v >= 0 for v in etas)


def test_build_rail_examples_conditions_on_rail():
    pay, evt, schema = _data()
    twin = schema["twin"]
    step_vocab = {s: i for i, s in
                  enumerate(sorted({s for w in twin["workflow"].values() for s in w}))}
    exc_vocab = {c: i for i, c in enumerate(["none"] + twin["exception_codes"])}
    rail_vocab = {r: i for i, r in enumerate(twin["rails"])}
    ex = build_rail_examples(evt, pay, step_vocab, exc_vocab, rail_vocab, exc_vocab, 16)
    assert len(ex) > 0
    one = ex[0]
    assert set(one) == {"step", "exc", "t", "direction", "target"}
    assert 0 <= one["direction"] < len(rail_vocab)                 # rail id in the "direction" slot
    assert len(one["step"]) == len(one["exc"]) == len(one["t"]) >= 1
