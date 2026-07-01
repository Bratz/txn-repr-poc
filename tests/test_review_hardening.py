"""Tests for the post-review hardening fixes (see tasks/todo.md)."""

import numpy as np
import pandas as pd
import pytest
import torch

from encoders.coerce import canon_categorical
from encoders.column_assembler import build_vocabs
from encoders.party_encoder import (
    PARTY_STRUCT_ATTRS, build_field_vocabs, encode_role_parties,
)


# --- #3 canonical coercion: int / float / str forms collapse to one key --------- #

def test_canon_categorical_collapses_dtypes():
    assert list(canon_categorical(pd.Series([123, 789]))) == ["123", "789"]
    assert list(canon_categorical(pd.Series([123.0, 789.0]))) == ["123", "789"]   # no ".0"
    assert list(canon_categorical(pd.Series(["123", "789"]))) == ["123", "789"]
    assert canon_categorical(pd.Series([1e8]))[0] == "100000000"                  # no sci
    assert canon_categorical(pd.Series([123.5]))[0] == "123.5"                    # kept
    assert canon_categorical(pd.Series(["007"]))[0] == "007"                      # leading zero


def test_vocab_build_and_encode_dtype_invariant():
    schema = {"buckets": {"high_card_categorical": ["cust_id"], "numerical": ["amt"],
                          "core": ["Ccy"]}}
    train = pd.DataFrame({"cust_id": [123456, 789012], "amt": [1.0, 2.0], "Ccy": ["INR", "INR"]})
    v = build_vocabs(train, schema)
    # serve the SAME ids but as float (the dtype-drift hazard) -> must hit, not UNK
    serve = pd.DataFrame({"cust_id": [123456.0, 789012.0], "amt": [1.0, 1.0],
                          "Ccy": ["INR", "INR"]})
    enc = v.encode(serve)
    unk = v.hc_size("cust_id") - 1
    assert enc["high_card"]["cust_id"].tolist() == [0, 1]      # both resolved, neither UNK
    assert unk not in enc["high_card"]["cust_id"].tolist()


# --- P0-1 party encoder: unseen attribute -> MASK row, not NaN/out-of-bounds ----- #

def test_party_encoder_unseen_attribute_maps_to_mask_not_crash():
    role = {"key": "acct", "attrs": {a: a for a in PARTY_STRUCT_ATTRS}}
    train = pd.DataFrame({"acct": ["A", "B"], **{a: ["seen", "seen2"] for a in PARTY_STRUCT_ATTRS}})
    vocabs = build_field_vocabs(train, {"r": role})
    serve = pd.DataFrame({"acct": ["C"], **{a: ["NEVER_SEEN"] for a in PARTY_STRUCT_ATTRS}})
    keys, ids = encode_role_parties(serve, role, vocabs)       # must not raise
    assert keys == ["C"]
    for j, a in enumerate(PARTY_STRUCT_ATTRS):
        assert int(ids[0, j]) == len(vocabs[a])                # MASK row (graceful UNK)
    assert ids.dtype.is_floating_point is False                # no NaN-via-float corruption


# --- P0-2 mis-routed flag + per-class routing metrics ---------------------------- #

def test_is_mis_routed_flags_injected_gate_rows():
    from data.synth_india_rails import IndiaConfig, build_dataset
    pay, _, _ = build_dataset(IndiaConfig(num_accounts=300, num_payments=4000, seed=23))
    assert "is_mis_routed" in pay.columns
    # over-cap UPI and below-min RTGS attempts must be flagged mis-routed...
    over = pay[(pay.rail == "UPI") & (pay.IntrBkSttlmAmt > 100_000)]
    assert len(over) and (over["is_mis_routed"] == 1).all()
    # ...while a normal small UPI payment is not
    ok = pay[(pay.rail == "UPI") & (pay.IntrBkSttlmAmt < 50_000)]
    assert len(ok) and (ok["is_mis_routed"] == 0).all()


# --- backlog: XML writer/parser robustness ------------------------------------- #

def test_writer_canonicalises_numeric_ids_and_roundtrips_ultmt_dbtr():
    from data.iso20022_pacs008 import parse_pacs008, write_pacs008
    rows = [{
        "IntrBkSttlmAmt": 1000.0, "Ccy": "INR", "IntrBkSttlmDt": "2026-06-29",
        "Dbtr_Nm": "D Co", "Dbtr_Ctry": "IN", "DbtrAcct_Id": 12345.0,     # numeric (float)
        "Cdtr_Nm": "C Co", "Cdtr_Ctry": "IN", "CdtrAcct_Id": 67890,       # numeric (int)
        "UltmtDbtr_Id": 111, "UltmtDbtr_Nm": "D Co",
        "UltmtCdtr_Id": 222, "UltmtCdtr_Nm": "C Co",
        "identifier_type": "ACCT_IFSC", "SttlmMtd": "CLRG",
    }]
    xml = write_pacs008(rows)
    assert "12345.0" not in xml and "12345" in xml                        # no ".0"
    p = parse_pacs008(xml)[0]
    assert p["DbtrAcct_Id"] == "12345" and p["CdtrAcct_Id"] == "67890"
    assert p["UltmtDbtr_Id"] == "111" and p["UltmtCdtr_Id"] == "222"      # symmetric Ultmt


def test_parser_guards_malformed_amount_and_accepts_bytes():
    from data.iso20022_pacs008 import parse_pacs008
    xml = ('<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08">'
           '<FIToFICstmrCdtTrf><GrpHdr><SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf></GrpHdr>'
           '<CdtTrfTxInf><IntrBkSttlmAmt Ccy="INR">N/A</IntrBkSttlmAmt>'
           '<Dbtr><Nm>D</Nm><PstlAdr><Ctry>IN</Ctry></PstlAdr></Dbtr>'
           '<DbtrAcct><Id><Othr><Id>A1</Id></Othr></Id></DbtrAcct>'
           '<Cdtr><Nm>C</Nm><PstlAdr><Ctry>IN</Ctry></PstlAdr></Cdtr>'
           '<CdtrAcct><Id><Othr><Id>A2</Id></Othr></Id></CdtrAcct>'
           '</CdtTrfTxInf></FIToFICstmrCdtTrf></Document>')
    assert parse_pacs008(xml)[0]["IntrBkSttlmAmt"] == 0.0                 # malformed -> 0
    assert parse_pacs008(xml.encode())[0]["Ccy"] == "INR"                # bytes accepted


def test_first_id_priority_prefers_bic():
    import xml.etree.ElementTree as ET
    from data.iso20022_pacs008 import _first_id
    party = ET.fromstring("<Cdtr><Id><OrgId><AnyBIC>CHASUS33</AnyBIC>"
                          "<Othr><Id>OTHER</Id></Othr></OrgId></Id></Cdtr>")
    assert _first_id(party) == "CHASUS33"                                 # BIC over Othr/Id


def test_rail_routing_reports_per_class_and_domestic():
    from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
    from run_india import _visible_features, intake_eval
    pay, _, accs = build_dataset(IndiaConfig(num_accounts=300, num_payments=3000, seed=23))
    schema = build_schema(pay, accs)
    rng = np.random.default_rng(0)
    e = np.hstack([_visible_features(pay), rng.normal(size=(len(pay), 8))])
    idx = rng.permutation(len(pay)); cut = int(len(idx) * 0.8)
    rr = intake_eval(e, pay, schema, idx[:cut], idx[cut:])["rail_routing"]
    assert {"per_class", "domestic_probe_accuracy", "clean_probe_accuracy"} <= set(rr)
    assert "SWIFT" in rr["per_class"] and "RTGS" in rr["per_class"]


# --- backlog: numerical / training robustness ---------------------------------- #

def test_quantizer_rejects_nonfinite():
    from encoders.quantizer import AdaptiveQuantizer
    with pytest.raises(ValueError):
        AdaptiveQuantizer(condition_on_currency=False).fit([1.0, np.nan, 3.0])
    q = AdaptiveQuantizer(condition_on_currency=False).fit([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        q.transform([np.inf])


def test_triplet_loss_no_nan_grad_on_identical_embeddings():
    from encoder.tabular_encoder import batch_hard_triplet_loss
    emb = torch.zeros(4, 8, requires_grad=True)            # all identical -> d == 0
    loss = batch_hard_triplet_loss(emb, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(emb.grad).all()   # no NaN at d=0


def test_history_pretrain_asserts_e_all_alignment():
    from encoder.history_encoder import HistoryConfig, HistoryEncoder, pretrain
    cfg = HistoryConfig(hidden=16, layers=1, heads=2, ff_mult=2, epochs=1)
    hist = HistoryEncoder(recon_fields={"step": 4}, config=cfg)
    e_all = torch.randn(10, 16)
    targets = {"step": torch.zeros(8, dtype=torch.long)}   # misaligned (8 != 10)
    with pytest.raises(AssertionError):
        pretrain(hist, e_all, targets, [], cfg)


def test_twin_intake_eval_keys_and_baselines():
    from data.synth_workflow import WfConfig, build_schema, build_workflow_dataset
    from run_twin import intake_eval as twin_intake
    pay, _, accs = build_workflow_dataset(WfConfig(num_accounts=120, num_payments=1500, seed=1))
    schema = build_schema(pay, accs)
    rng = np.random.default_rng(0)
    e = rng.normal(size=(len(pay), 12))
    idx = rng.permutation(len(pay)); cut = int(len(idx) * 0.8)
    out = twin_intake(e, pay, schema["twin"], idx[:cut], idx[cut:])
    assert {"exception_pr_auc", "status_accuracy", "status_majority_baseline",
            "eta_mae_min", "eta_baseline_mae_min"} <= set(out)
    assert out["eta_mae_min"] >= 0 and 0 <= out["status_majority_baseline"] <= 1
