"""Round-trip test for the persisted India model (serve_india.py)."""

import numpy as np
import pandas as pd
import torch

from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
from encoder.tabular_encoder import pretrain as enc_pretrain
from encoders.quantizer import AdaptiveQuantizer
from run_india import train_probes
from run_seq import embed_all_rows
from serve_india import load_india_model, save_india_model


def test_save_load_predict_roundtrip(tmp_path):
    pay, evt, accs = build_dataset(IndiaConfig(num_accounts=150, num_payments=800, seed=23))
    schema = build_schema(pay, accs)
    cfg = EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(pay, schema, cfg, party_epochs=1)
    enc_pretrain(enc, vocabs.encode(pay), cfg, batch_size=128)
    enc.freeze()
    e = embed_all_rows(enc, vocabs.encode(pay), len(pay), "cpu").cpu().numpy()

    probes = train_probes(e, pay, schema, np.arange(len(pay)))
    from data.synth_india_rails import build_messages
    from serve_india import fit_inflight_heads, fit_velocity
    msg = build_messages(pay, evt)
    probes["inflight"] = fit_inflight_heads(enc, vocabs, msg, pay, "cpu", max_uetrs=300)
    # velocity fits on a DENSER fleet (~15 payments/account) - the intake fleet is too sparse
    dense, _, _ = build_dataset(IndiaConfig(num_accounts=40, num_payments=600, seed=7))
    vel = fit_velocity(enc, vocabs, dense, "cpu", hist_epochs=1)
    quant = AdaptiveQuantizer().fit(pay[vocabs.numerical_col].to_numpy(),
                                    pay[vocabs.ccy_col].to_numpy())
    save_india_model(tmp_path / "m", enc_cfg=cfg, vocabs=vocabs, quantizer=quant,
                     encoder=enc, schema=schema, probes=probes, velocity=vel)

    # reload in a clean scorer (no retraining) and predict
    scorer = load_india_model(tmp_path / "m", device="cpu")
    sub = pay.head(20)
    res = scorer.predict(sub)
    assert {"rail_pred", "rail_conf", "status_pred", "eta_min_pred",
            "top_exception_risks"} <= set(res.columns)
    assert (res["eta_min_pred"] >= 0).all()                  # ETA clamped
    assert set(res["rail_pred"]) <= set(probes["rail"].classes_)
    # §5 single-record heads ride on the same backbone
    assert {"risk_pred", "geography_pred", "expense_pred"} <= set(res.columns)

    # eligibility guard: every predicted rail is actually eligible for that payment
    from data.rails import IDENTIFIER_TYPES, eligible_rails
    for _, r in pd.concat([sub.reset_index(drop=True), res], axis=1).iterrows():
        ident = r["identifier_type"] if r["identifier_type"] in IDENTIFIER_TYPES else None
        xb = r["Dbtr_Ctry"] != r["Cdtr_Ctry"]
        elig = eligible_rails(float(r["IntrBkSttlmAmt"]), ident, bool(xb))
        if elig:
            assert r["rail_pred"] in elig

    # no-eligible-rail fallback: a contradictory instrument (VPA over the UPI cap, domestic)
    # has no eligible rail under the instrument constraint; predict must still emit a rail
    # that is valid by amount + cross-border.
    bad = sub.iloc[[0]].copy()
    bad["identifier_type"] = "VPA"; bad["IntrBkSttlmAmt"] = 150_000.0
    bad["Dbtr_Ctry"] = "IN"; bad["Cdtr_Ctry"] = "IN"
    from data.rails import eligible_rails
    assert scorer.predict(bad)["rail_pred"].iloc[0] in eligible_rails(150_000, None, False)

    # round-trip fidelity: a fresh in-memory scorer and the reloaded one produce identical
    # predictions on the same rows (save/load is lossless, guard applied in both).
    from serve_india import IndiaScorer
    ref = IndiaScorer(enc, vocabs, probes, "cpu").predict(sub)
    assert list(res["rail_pred"]) == list(ref["rail_pred"])
    assert list(res["status_pred"]) == list(ref["status_pred"])
    assert list(res["risk_pred"]) == list(ref["risk_pred"])

    # in-flight streaming: the persisted heads emit the predicted-lifecycle object per UETR.
    some = msg[msg["payment_id"].isin(set(pay["payment_id"].head(30)))]
    st = scorer.predict_stream(some)
    assert {"end_to_end_id", "n_msgs", "last_msg_type", "booked_proba",
            "settlement_outcome", "eta_remaining_min", "reject_reason_if_failed",
            "cancel_proba", "return_proba"} <= set(st.columns)
    assert len(st) and st["booked_proba"].between(0, 1).all()
    assert (st["eta_remaining_min"] >= 0).all()
    dist = st["settlement_outcome"].iloc[0]                  # a proper distribution
    assert dist is not None and abs(sum(dist.values()) - 1.0) < 1e-6
    # a prefix WITHOUT outcome messages must also score (the real-time case)
    pre = some[~some["msg_type"].isin(["pacs.002", "camt.054", "pacs.004"])]
    st_pre = scorer.predict_stream(pre)
    assert len(st_pre) and st_pre["booked_proba"].between(0, 1).all()
    # snap-on-outcome: once camt.054 (BOOK) is the last message seen, the score must be high -
    # the last-message one-hot makes the outcome directly readable, not diluted by the pool.
    snapped = st[st["last_msg_type"] == "camt.054"]
    assert len(snapped) and (snapped["booked_proba"] > 0.5).all()
    # an old-style bundle (no in-flight heads) fails loudly, not silently
    import pytest
    bare = {k: v for k, v in probes.items() if k != "inflight"}
    with pytest.raises(SystemExit):
        IndiaScorer(enc, vocabs, bare, "cpu").predict_stream(some)
    # feature-vector contract: pool + msg_type one-hot + tx_sts one-hot + 2 time features
    from data.iso_lifecycle import MSG_TYPES, TX_STS
    from serve_india import _prefix_features
    x = _prefix_features(np.zeros(cfg.hidden), msg.head(2))
    assert x.shape[0] == cfg.hidden + len(MSG_TYPES) + len(TX_STS) + 2

    # velocity: served statelessly from caller-supplied history (>=2 rows per account)
    if vel is not None and vel["head"] is not None:
        actors = dense["DbtrAcct_Id"].value_counts()
        busy = dense[dense["DbtrAcct_Id"].isin(actors[actors >= 5].index[:5])]
        v = scorer.predict_velocity(busy)
        assert {"actor", "n_txns", "burst_proba", "burst_rule"} <= set(v.columns)
        assert len(v) and v["burst_proba"].between(0, 1).all()
        bare_scorer = IndiaScorer(enc, vocabs, probes, "cpu")   # no hist -> loud failure
        with pytest.raises(SystemExit):
            bare_scorer.predict_velocity(busy)

    # explain: column-occlusion drivers - faithful fields, bounded top-k
    drv = scorer.explain(sub.head(2), top_k=4)
    assert len(drv) == 2 and all(len(d) <= 4 for d in drv)
    assert {"field", "rail_impact", "risk_impact", "eta_impact_min"} <= set(drv[0][0])
