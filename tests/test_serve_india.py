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
    pay, _, accs = build_dataset(IndiaConfig(num_accounts=150, num_payments=800, seed=23))
    schema = build_schema(pay, accs)
    cfg = EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(pay, schema, cfg, party_epochs=1)
    enc_pretrain(enc, vocabs.encode(pay), cfg, batch_size=128)
    enc.freeze()
    e = embed_all_rows(enc, vocabs.encode(pay), len(pay), "cpu").cpu().numpy()

    probes = train_probes(e, pay, schema, np.arange(len(pay)))
    quant = AdaptiveQuantizer().fit(pay[vocabs.numerical_col].to_numpy(),
                                    pay[vocabs.ccy_col].to_numpy())
    save_india_model(tmp_path / "m", enc_cfg=cfg, vocabs=vocabs, quantizer=quant,
                     encoder=enc, schema=schema, probes=probes)

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

    # round-trip fidelity: a fresh in-memory scorer and the reloaded one produce identical
    # predictions on the same rows (save/load is lossless, guard applied in both).
    from serve_india import IndiaScorer
    ref = IndiaScorer(enc, vocabs, probes, "cpu").predict(sub)
    assert list(res["rail_pred"]) == list(ref["rail_pred"])
    assert list(res["status_pred"]) == list(ref["status_pred"])
    assert list(res["risk_pred"]) == list(ref["risk_pred"])
