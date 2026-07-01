"""Tests for the Layer-1 pacs.008 projection (data/iso20022_pacs008.py)."""

from pathlib import Path

import pytest

from data.iso20022_pacs008 import UNKNOWN, parse_pacs008, parse_pacs008_frame, write_pacs008
from data.synth_pacs008 import COLUMN_BUCKETS

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_pacs008.xml"


def test_parses_both_transactions_and_core_fields():
    rows = parse_pacs008(SAMPLE)
    assert len(rows) == 2
    a, b = rows
    assert a["IntrBkSttlmAmt"] == 125000.0 and a["Ccy"] == "USD"
    assert a["DbtrAcct_Id"] == "IN9988776655"
    assert a["CdtrAcct_Id"] == "US64SVBKUS6S3300958879"        # IBAN extracted
    assert a["Dbtr_Ctry"] == "IN" and a["Cdtr_Ctry"] == "US"
    assert a["Dbtr_Nm"] == "Meridian Pte Trading"
    assert a["SttlmMtd"] == "COVE"                              # from GrpHdr/SttlmInf
    assert b["IntrBkSttlmAmt"] == 45000.0 and b["Ccy"] == "INR"


def test_identifier_type_derivation():
    a, b = parse_pacs008(SAMPLE)
    assert a["identifier_type"] == "BIC_IBAN"                   # cross-border + IBAN
    assert b["identifier_type"] == "ACCT_IFSC"                  # domestic Othr/Id


def test_industry_defaults_to_unknown_without_enrichment():
    a, _ = parse_pacs008(SAMPLE)
    assert a["Dbtr_Industry"] == UNKNOWN and a["Cdtr_SubIndustry"] == UNKNOWN


def test_enrich_supplies_industry():
    rows = parse_pacs008(SAMPLE, enrich={"IN9988776655": {"industry": "Financial",
                                                          "sub_industry": "Banks"}})
    assert rows[0]["Dbtr_Industry"] == "Financial" and rows[0]["Dbtr_SubIndustry"] == "Banks"


def test_namespace_version_agnostic():
    # a different pacs.008 version namespace must still parse (we match local tag names)
    xml = SAMPLE.read_text().replace("pacs.008.001.08", "pacs.008.001.10")
    rows = parse_pacs008(xml)
    assert len(rows) == 2 and rows[0]["Ccy"] == "USD"


def test_frame_has_encoder_feature_columns():
    df = parse_pacs008_frame(SAMPLE)
    feats = COLUMN_BUCKETS["high_card_categorical"] + COLUMN_BUCKETS["numerical"] + \
        COLUMN_BUCKETS["meta_party"] + ["Ccy", "IntrBkSttlmDt", "identifier_type"]
    for c in feats:
        assert c in df.columns                                 # encodable row
    assert "payment_id" in df.columns


def test_non_pacs008_raises():
    with pytest.raises(ValueError):
        parse_pacs008("<Document><Foo/></Document>")


def test_write_then_parse_roundtrips_native_fields():
    rows = parse_pacs008(SAMPLE)
    reparsed = parse_pacs008(write_pacs008(rows))
    assert len(reparsed) == len(rows)
    for a, b in zip(rows, reparsed):
        for k in ("IntrBkSttlmAmt", "Ccy", "IntrBkSttlmDt", "DbtrAcct_Id", "CdtrAcct_Id",
                  "Dbtr_Ctry", "Cdtr_Ctry", "Dbtr_Nm", "Cdtr_Nm", "identifier_type"):
            assert a[k] == b[k]
