"""API round-trip for api_india (FastAPI wrapper over the persisted model)."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from data.synth_india_rails import IndiaConfig, build_dataset, build_messages, build_schema
from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
from encoder.tabular_encoder import pretrain as enc_pretrain
from encoders.quantizer import AdaptiveQuantizer
from run_india import train_probes
from run_seq import embed_all_rows
from serve_india import fit_inflight_head, save_india_model


def test_api_endpoints_roundtrip(tmp_path, monkeypatch):
    pay, evt, accs = build_dataset(IndiaConfig(num_accounts=120, num_payments=600, seed=23))
    schema = build_schema(pay, accs)
    cfg = EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(pay, schema, cfg, party_epochs=1)
    enc_pretrain(enc, vocabs.encode(pay), cfg, batch_size=128)
    enc.freeze()
    e = embed_all_rows(enc, vocabs.encode(pay), len(pay), "cpu").cpu().numpy()
    probes = train_probes(e, pay, schema, np.arange(len(pay)))
    msg = build_messages(pay, evt)
    probes["inflight_booked"] = fit_inflight_head(enc, vocabs, msg, "cpu", max_uetrs=200)
    quant = AdaptiveQuantizer().fit(pay[vocabs.numerical_col].to_numpy(),
                                    pay[vocabs.ccy_col].to_numpy())
    save_india_model(tmp_path / "m", enc_cfg=cfg, vocabs=vocabs, quantizer=quant,
                     encoder=enc, schema=schema, probes=probes)

    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    import api_india
    with TestClient(api_india.app) as c:
        meta = c.get("/model/meta").json()
        assert meta["inflight"] is True and set(meta["rails"]) >= {"IMPS", "NEFT"}

        r = c.post("/score/intake",
                   json={"payments": json.loads(pay.head(3).to_json(orient="records"))})
        assert r.status_code == 200
        res = r.json()["results"]
        assert len(res) == 3 and {"rail_pred", "eta_min_pred"} <= set(res[0])
        assert isinstance(res[0]["top_exception_risks"], list)       # structured, not a string
        assert all("code" in x and "score" in x for x in res[0]["top_exception_risks"])

        some = msg[msg["payment_id"].isin(set(pay["payment_id"].head(10)))]
        m = c.post("/score/inflight",
                   json={"messages": json.loads(some.to_json(orient="records"))})
        assert m.status_code == 200
        assert all(0.0 <= x["booked_proba"] <= 1.0 for x in m.json()["results"])

        assert c.post("/score/intake", json={}).status_code == 422    # neither rows nor XML
        bad = c.post("/score/inflight", json={"messages": [{"foo": 1}]})
        assert bad.status_code == 422                                 # missing stream columns
