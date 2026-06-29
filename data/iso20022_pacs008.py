"""
Layer-1 projection (LIVE): raw ISO 20022 pacs.008 XML -> a projected feature row.

This is the inverse of synth_pacs008.project_to_pacs008: instead of building a row from
synthetic Account objects, it parses a real pacs.008 (`FIToFICstmrCdtTrf`) message and emits
the same columns the encoder consumes, so serve_india.py can score actual message XML.

What pacs.008 DOES carry (parsed directly):
  IntrBkSttlmAmt + @Ccy, IntrBkSttlmDt, Dbtr/Cdtr Nm, Dbtr/Cdtr PstlAdr/Ctry,
  Dbtr/Cdtr Acct Id (IBAN or Othr/Id), UltmtDbtr/UltmtCdtr Id+Nm (or Dbtr/Cdtr as fallback),
  GrpHdr SttlmInf/SttlmMtd.

What pacs.008 does NOT carry (honest gaps - defaulted, or supplied via `enrich`):
  * industry / sub-industry are ENRICHMENT attributes (from a party master), not message
    fields -> default "Unknown" (the encoder's vocab maps unseen categories to its OOV
    bucket), or pass `enrich={party_or_acct_id: {"industry":..,"sub_industry":..}}`.
  * identifier_type is India-rail-specific. UPI(VPA)/IMPS(MMID) proxies are NOT standard
    pacs.008 elements (those rails use NPCI's own rails/APIs); cross-border pacs.008 is
    BIC/IBAN. We derive: IBAN or cross-border -> BIC_IBAN, else ACCT_IFSC. Override by
    setting row["identifier_type"] before scoring if you know better.

Namespaces: pacs.008 has versioned namespaces (pacs.008.001.08/.09/.10/...). We match by
LOCAL tag name so any version parses. Multi-transaction messages yield one row per
CdtTrfTxInf.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# Columns the projection fills (the encoder reads these via the schema buckets).
UNKNOWN = "Unknown"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    if el is None:
        return None
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _find(el, path):
    """Descend by a list of local tag names; return the first matching element or None."""
    cur = el
    for name in path:
        cur = _child(cur, name)
        if cur is None:
            return None
    return cur


def _findall(el, name):
    return [c for c in (el or []) if _local(c.tag) == name]


def _text(el, path, default=None):
    node = _find(el, path)
    return node.text.strip() if (node is not None and node.text) else default


def _first_id(party):
    """First non-empty Id leaf under a party's Id (OrgId/AnyBIC, Othr/Id, PrvtId/...)."""
    idel = _child(party, "Id")
    if idel is None:
        return None
    for node in idel.iter():
        if node is not party and node.text and node.text.strip() and _local(node.tag) in (
                "AnyBIC", "BICFI", "LEI", "Id"):
            return node.text.strip()
    return None


def _acct_id(acct):
    """Account id: IBAN if present, else Othr/Id."""
    if acct is None:
        return None
    iban = _text(acct, ["Id", "IBAN"])
    return iban or _text(acct, ["Id", "Othr", "Id"])


def _agent_bic(tx, tag):
    return _text(_find(tx, [tag]), ["FinInstnId", "BICFI"]) or \
        _text(_find(tx, [tag]), ["FinInstnId", "BIC"])


def _enrich(row, key, enrich):
    info = (enrich or {}).get(key) if key else None
    if info:
        return info.get("industry", UNKNOWN), info.get("sub_industry", UNKNOWN)
    return UNKNOWN, UNKNOWN


def _project_tx(tx, sttlm_mtd, enrich):
    amt = _find(tx, ["IntrBkSttlmAmt"])
    amount = float(amt.text) if (amt is not None and amt.text) else 0.0
    ccy = amt.get("Ccy") if amt is not None else None

    dbtr, cdtr = _find(tx, ["Dbtr"]), _find(tx, ["Cdtr"])
    udbtr, ucdtr = _find(tx, ["UltmtDbtr"]), _find(tx, ["UltmtCdtr"])
    dbtr_acct, cdtr_acct = _acct_id(_find(tx, ["DbtrAcct"])), _acct_id(_find(tx, ["CdtrAcct"]))
    dbtr_ctry = _text(dbtr, ["PstlAdr", "Ctry"])
    cdtr_ctry = _text(cdtr, ["PstlAdr", "Ctry"])

    ultmt_dbtr_id = _first_id(udbtr) or _first_id(dbtr) or _agent_bic(tx, "DbtrAgt")
    ultmt_cdtr_id = _first_id(ucdtr) or _first_id(cdtr) or _agent_bic(tx, "CdtrAgt")

    d_ind, d_sub = _enrich({}, dbtr_acct, enrich)
    c_ind, c_sub = _enrich({}, cdtr_acct, enrich)

    # identifier_type (India-rail feature): IBAN / cross-border -> BIC_IBAN, else ACCT_IFSC.
    xborder = bool(dbtr_ctry and cdtr_ctry and dbtr_ctry != cdtr_ctry)
    has_iban = (_text(_find(tx, ["CdtrAcct"]), ["Id", "IBAN"]) is not None)
    identifier_type = "BIC_IBAN" if (xborder or has_iban) else "ACCT_IFSC"

    return {
        # high-cardinality categorical (partitioning embedder)
        "DbtrAcct_Id": dbtr_acct or UNKNOWN,
        "CdtrAcct_Id": cdtr_acct or UNKNOWN,
        "UltmtDbtr_Id": ultmt_dbtr_id or UNKNOWN,
        "UltmtCdtr_Id": ultmt_cdtr_id or UNKNOWN,
        # numerical
        "IntrBkSttlmAmt": amount,
        # core
        "Ccy": ccy or UNKNOWN,
        "IntrBkSttlmDt": _text(tx, ["IntrBkSttlmDt"]) or UNKNOWN,
        "SttlmMtd": sttlm_mtd or UNKNOWN,
        "identifier_type": identifier_type,
        # meta party (offline encoder)
        "Dbtr_Nm": _text(dbtr, ["Nm"]) or UNKNOWN,
        "Cdtr_Nm": _text(cdtr, ["Nm"]) or UNKNOWN,
        "UltmtDbtr_Nm": _text(udbtr, ["Nm"]) or _text(dbtr, ["Nm"]) or UNKNOWN,
        "UltmtCdtr_Nm": _text(ucdtr, ["Nm"]) or _text(cdtr, ["Nm"]) or UNKNOWN,
        "Dbtr_Ctry": dbtr_ctry or UNKNOWN,
        "Cdtr_Ctry": cdtr_ctry or UNKNOWN,
        "Dbtr_Industry": d_ind, "Cdtr_Industry": c_ind,
        "Dbtr_SubIndustry": d_sub, "Cdtr_SubIndustry": c_sub,
    }


def parse_pacs008(source, enrich: dict | None = None) -> list[dict]:
    """Parse a pacs.008 message (path, XML string, or bytes) -> list of projected rows.

    `enrich` optionally supplies {account_id: {"industry":.., "sub_industry":..}} since those
    attributes are not in the message. One row per CdtTrfTxInf.
    """
    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        root = ET.parse(str(source)).getroot()
    else:
        root = ET.fromstring(source.encode() if isinstance(source, str) else source)

    body = _find(root, ["FIToFICstmrCdtTrf"]) or root      # tolerate Document or bare body
    grp = _find(body, ["GrpHdr"])
    sttlm_mtd = _text(grp, ["SttlmInf", "SttlmMtd"])
    txs = _findall(body, "CdtTrfTxInf")
    if not txs:
        raise ValueError("no CdtTrfTxInf found - is this a pacs.008 FIToFICstmrCdtTrf message?")
    return [_project_tx(tx, sttlm_mtd, enrich) for tx in txs]


def parse_pacs008_frame(source, enrich: dict | None = None):
    import pandas as pd
    rows = parse_pacs008(source, enrich)
    df = pd.DataFrame(rows)
    df.insert(0, "payment_id", range(len(df)))
    return df


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Project pacs.008 XML -> feature rows")
    ap.add_argument("--input", required=True, help="pacs.008 .xml file")
    ap.add_argument("--out", default=None, help="optional CSV out; else prints JSON")
    args = ap.parse_args()
    df = parse_pacs008_frame(args.input)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"projected {len(df)} transaction(s) -> {args.out}")
    else:
        print(json.dumps(df.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
