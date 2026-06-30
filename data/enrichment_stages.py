"""
Outbound enrichment lifecycle: which pacs.008 fields exist at each stage.

Outbound payments start partial (order/acquisition) and are enriched until a complete
pacs.008 is assembled; inbound payments are born 'complete'. We model a stage by the SET of
fields available, and derive a partial view by masking the not-yet-available fields. The
encoder's masked-column reconstruction then IMPUTES those fields, and downstream heads can
predict in-flight from the partial representation.

ponytail: the mask IS the stage signal (the stages are nested sets, so the set of UNK fields
uniquely identifies the stage) — no separate stage feature needed. Stages are a documented
synthetic choice; reorder for a real engine's enrichment sequence.
"""

# Cumulative: each stage ADDS fields to all previous stages. Field names are encoder
# reconstructable columns (high-card ids / core / amount). Anything not listed before
# 'complete' is treated as arriving at 'complete'.
STAGE_ADDS = [
    ("order",     ["DbtrAcct_Id", "CdtrAcct_Id", "IntrBkSttlmAmt", "Ccy"]),  # who/whom/how much
    ("validated", ["IntrBkSttlmDt", "UltmtDbtr_Id", "identifier_type"]),     # dated, payer resolved
    ("enriched",  ["UltmtCdtr_Id"]),                                          # payee chain resolved
    ("complete",  []),                                                        # everything present
]


def stage_names():
    return [n for n, _ in STAGE_ADDS]


def available_at(stage_idx: int) -> set:
    s = set()
    for i in range(stage_idx + 1):
        s |= set(STAGE_ADDS[i][1])
    return s


def missing_mask(recon_names, stage_idx: int) -> list:
    """Bool list aligned to recon_names: True = field not yet available at this stage
    (i.e. masked/UNK, to be imputed). 'complete' masks nothing."""
    if STAGE_ADDS[stage_idx][0] == "complete":
        return [False] * len(recon_names)
    avail = available_at(stage_idx)
    return [n not in avail for n in recon_names]
