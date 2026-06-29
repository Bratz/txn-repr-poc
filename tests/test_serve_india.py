"""Round-trip test for the persisted India model (serve_india.py)."""

import numpy as np
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

    # round-trip fidelity: reloaded encoder reproduces the same embeddings, so the
    # loaded model's rail predictions match the in-memory probe on the same rows.
    in_mem = probes["rail"].classes_[probes["rail"].predict_proba(e[:20]).argmax(1)]
    assert list(res["rail_pred"]) == list(in_mem)
