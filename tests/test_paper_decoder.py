"""Paper-exact serving tier: one frozen LLM + adapters answers the classification
menu via instructions (Raman et al. Sec. 4); the head farm for those tasks is gone."""

import numpy as np
import pytest
import torch

from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
from encoder.tabular_encoder import pretrain as enc_pretrain
from encoders.quantizer import AdaptiveQuantizer
from run_india import train_probes
from run_seq import embed_all_rows
from serve_india import (IndiaScorer, fit_instruction_decoder, india_task_menu,
                         load_india_model, save_india_model)


def test_paper_serving_roundtrip(tmp_path):
    pay, evt, accs = build_dataset(IndiaConfig(num_accounts=120, num_payments=700, seed=23))
    schema = build_schema(pay, accs)
    cfg = EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(pay, schema, cfg, party_epochs=1)
    enc_pretrain(enc, vocabs.encode(pay), cfg, batch_size=128)
    enc.freeze()
    e = embed_all_rows(enc, vocabs.encode(pay), len(pay), "cpu").cpu().numpy()

    # the menu comes from the data, and covers the five class tasks + the ETA band
    menu = {t["name"] for t in india_task_menu(pay)}
    assert {"rail", "status", "risk", "geography", "expense", "eta_band"} <= menu

    tr = np.arange(len(pay) // 2)
    dec_spec = fit_instruction_decoder(enc, vocabs, pay, tr, "cpu", smoke=True, epochs=1)
    assert dec_spec["n_tasks"] == len(menu) and dec_spec["llm"] == "mock"

    # the authorized removal: classification probes are NOT shipped
    probes = train_probes(e, pay, schema, tr)
    for k in ("rail", "status", "tasks"):
        probes.pop(k, None)
    quant = AdaptiveQuantizer().fit(pay[vocabs.numerical_col].to_numpy(),
                                    pay[vocabs.ccy_col].to_numpy())
    save_india_model(tmp_path / "m", enc_cfg=cfg, vocabs=vocabs, quantizer=quant,
                     encoder=enc, schema=schema, probes=probes, paper_decoder=dec_spec)

    scorer = load_india_model(tmp_path / "m", device="cpu")
    sub = pay.head(12)

    # ask(): the paper interface - every task answered by the ONE decoder, per row
    res = scorer.ask(sub)
    assert len(res) == len(sub)
    first = res[0]["tasks"]
    assert set(first) == menu
    for name, r in first.items():
        assert r["pred"] in r["proba"] and abs(sum(r["proba"].values()) - 1.0) < 1e-3
    # single-task ask + unknown task fails loudly
    only = scorer.ask(sub.head(3), task="rail")
    assert set(only[0]["tasks"]) == {"rail"}
    with pytest.raises(SystemExit):
        scorer.ask(sub.head(1), task="nope")

    # predict() keeps its API shape, routing the class menu through the decoder;
    # rail predictions still respect the eligibility guard.
    out = scorer.predict(sub)
    assert {"rail_pred", "rail_conf", "status_pred", "eta_min_pred", "risk_pred",
            "geography_pred", "expense_pred", "top_exception_risks"} <= set(out.columns)
    from data.rails import IDENTIFIER_TYPES, eligible_rails
    for _, r in pay.head(12).reset_index(drop=True).join(out, rsuffix="_p").iterrows():
        ident = r["identifier_type"] if r["identifier_type"] in IDENTIFIER_TYPES else None
        elig = eligible_rails(float(r["IntrBkSttlmAmt"]), ident,
                              bool(r["Dbtr_Ctry"] != r["Cdtr_Ctry"]))
        if elig:
            assert r["rail_pred"] in elig

    # meta advertises the tier; label sets come from the decoder registry
    import json
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["paper_decoder"] and meta["decoder_llm"] == "mock"
    assert set(meta["rails"]) == set(pay["rail"].astype(str).unique())

    # explain() is a probe-tier feature - loud failure, not silence
    with pytest.raises(SystemExit):
        scorer.explain(sub.head(1))
    # a bundle with NO decoder cannot ask
    with pytest.raises(SystemExit):
        IndiaScorer(enc, vocabs, probes, "cpu").ask(sub.head(1))
