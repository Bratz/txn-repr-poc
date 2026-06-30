"""Tests for the enrichment stage map (data/enrichment_stages.py)."""

from data.enrichment_stages import STAGE_ADDS, missing_mask, stage_names

RECON = ["DbtrAcct_Id", "CdtrAcct_Id", "IntrBkSttlmAmt", "Ccy",
         "IntrBkSttlmDt", "UltmtDbtr_Id", "UltmtCdtr_Id", "identifier_type"]


def _missing(stage_idx):
    return {RECON[j] for j, m in enumerate(missing_mask(RECON, stage_idx)) if m}


def test_order_stage_masks_not_yet_available_fields():
    assert _missing(0) == {"IntrBkSttlmDt", "UltmtDbtr_Id", "UltmtCdtr_Id", "identifier_type"}


def test_complete_stage_masks_nothing():
    assert not any(missing_mask(RECON, len(STAGE_ADDS) - 1))


def test_stages_are_nested_monotonic():
    # each later stage's missing set is a subset of the earlier one's (fields only get added)
    for i in range(len(STAGE_ADDS) - 1):
        assert _missing(i + 1) <= _missing(i)
    assert stage_names()[0] == "order" and stage_names()[-1] == "complete"
