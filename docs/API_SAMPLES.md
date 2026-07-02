# API samples — txn-repr India advisory scorer

All requests/responses below were captured live against `api_india:app` (bundle `model_india`, CPU). Start the server with:

```bash
MODEL_DIR=model_india uvicorn api_india:app --port 8000
```

> Scores are uncalibrated balanced-probe rankings (see docs/CBPR_TWIN_GAP.md): use them to rank and alert, not as literal percentages. cancel/return probas especially are rank-only (rare classes).

## 1. GET /model/meta — the bundle contract

```bash
curl http://localhost:8000/model/meta
```

```json
{
  "rails": [
    "IMPS",
    "NEFT",
    "RTGS",
    "SWIFT"
  ],
  "statuses": [
    "MANUAL_REVIEW",
    "REJECTED",
    "REPAIRED",
    "STP"
  ],
  "exceptions": [
    "account_closed",
    "batch_return",
    "below_min",
    "beneficiary_unreachable",
    "format_error",
    "fraud_hold",
    "fx_fail",
    "insufficient_liquidity",
    "limit_exceeded",
    "missing_field",
    "no_cover",
    "no_route",
    "sanctions_hit",
    "settlement_fail",
    "sla_breach",
    "technical_decline"
  ],
  "tasks": [
    "risk",
    "geography",
    "expense"
  ],
  "hidden": 512,
  "inflight": true,
  "model_dir": "model_india"
}
```

## 2. POST /score/intake — JSON payment row (minimal: the 19 feature fields)

Labels/extra columns are ignored if present; this is all the model reads.

```bash
curl -X POST http://localhost:8000/score/intake -H 'Content-Type: application/json' -d @intake.json
```

**intake.json**
```json
{
  "payments": [
    {
      "DbtrAcct_Id": "AZLN7144",
      "CdtrAcct_Id": "0QF6OXNV",
      "UltmtDbtr_Id": "HT8JC5",
      "UltmtCdtr_Id": "UWK0LS",
      "IntrBkSttlmAmt": 832.91,
      "Ccy": "INR",
      "IntrBkSttlmDt": "2023-11-30",
      "SttlmMtd": "CLRG",
      "identifier_type": "ACCT_IFSC",
      "Dbtr_Nm": "Granite GmbH Capital Services",
      "Cdtr_Nm": "Harbor Inc Sales Capital",
      "UltmtDbtr_Nm": "Granite GmbH Capital",
      "UltmtCdtr_Nm": "Harbor Inc Sales",
      "Dbtr_Ctry": "IN",
      "Cdtr_Ctry": "IN",
      "Dbtr_Industry": "Communications",
      "Cdtr_Industry": "Energy",
      "Dbtr_SubIndustry": "Media",
      "Cdtr_SubIndustry": "Renewables"
    }
  ]
}
```

**Response 200**
```json
{
  "model": "model_india",
  "results": [
    {
      "rail_pred": "IMPS",
      "rail_conf": 0.599,
      "status_pred": "STP",
      "eta_min_pred": 4.7,
      "risk_pred": "Low",
      "geography_pred": "Asia",
      "expense_pred": "Operational",
      "top_exception_risks": [
        {
          "code": "sla_breach",
          "score": 0.7
        },
        {
          "code": "beneficiary_unreachable",
          "score": 0.39
        },
        {
          "code": "below_min",
          "score": 0.3
        }
      ]
    }
  ]
}
```

## 3. POST /score/intake — raw ISO 20022 pacs.008 XML

One result per `CdtTrfTxInf`. Enrichment-only fields (industries) default to Unknown unless supplied.

```json
{"pacs008_xml": "<?xml version=\"1.0\" ... <FIToFICstmrCdtTrf> ... "}
```

**Response 200**
```json
{
  "model": "model_india",
  "results": [
    {
      "payment_id": 0,
      "rail_pred": "SWIFT",
      "rail_conf": 1.0,
      "status_pred": "REPAIRED",
      "eta_min_pred": 2001.1,
      "risk_pred": "High",
      "geography_pred": "International",
      "expense_pred": "Operational",
      "top_exception_risks": [
        {
          "code": "no_route",
          "score": 0.97
        },
        {
          "code": "fx_fail",
          "score": 0.93
        },
        {
          "code": "sanctions_hit",
          "score": 0.86
        }
      ]
    },
    {
      "payment_id": 1,
      "rail_pred": "NEFT",
      "rail_conf": 0.013,
      "status_pred": "REPAIRED",
      "eta_min_pred": 1603.4,
      "risk_pred": "Medium",
      "geography_pred": "Asia",
      "expense_pred": "Operational",
      "top_exception_risks": [
        {
          "code": "fx_fail",
          "score": 0.99
        },
        {
          "code": "sanctions_hit",
          "score": 0.99
        },
        {
          "code": "no_route",
          "score": 0.52
        }
      ]
    }
  ]
}
```

## 4. POST /score/inflight — pain.001 only (all predicted future states)

The minimal prefix: one initiation message. Unowned fields are 'Unknown' by design.

**Request**
```json
{
  "messages": [
    {
      "end_to_end_id": "E2E-00020002",
      "payment_id": 20002,
      "seq": 0,
      "t_offset_min": 0.0,
      "msg_type": "pain.001",
      "msg_direction": "IN",
      "tx_sts": null,
      "sts_reason": null,
      "rail": "IMPS",
      "direction": "outward",
      "DbtrAcct_Id": "YAEN26UU",
      "CdtrAcct_Id": "IYYSS6AE",
      "UltmtDbtr_Id": "Y2Q6OP",
      "UltmtCdtr_Id": "Unknown",
      "IntrBkSttlmAmt": 5934.94,
      "Ccy": "INR",
      "IntrBkSttlmDt": "Unknown",
      "SttlmMtd": "Unknown",
      "identifier_type": "ACCT_IFSC",
      "Dbtr_Nm": "Granite Pte Holdings",
      "Cdtr_Nm": "Harbor Inc Capital",
      "UltmtDbtr_Nm": "Granite Pte",
      "UltmtCdtr_Nm": "Unknown",
      "Dbtr_Ctry": "IN",
      "Cdtr_Ctry": "IN",
      "Dbtr_Industry": "Industrials",
      "Cdtr_Industry": "Communications",
      "Dbtr_SubIndustry": "Manufacturing",
      "Cdtr_SubIndustry": "Media",
      "visible": 1
    }
  ]
}
```

**Response 200 — the predicted-lifecycle object**
```json
{
  "model": "model_india",
  "results": [
    {
      "end_to_end_id": "E2E-00020002",
      "n_msgs": 1,
      "last_msg_type": "pain.001",
      "booked_proba": 0.1867,
      "settlement_outcome": {
        "MANUAL_REVIEW": 0.6741,
        "REJECTED": 0.1057,
        "REPAIRED": 0.1089,
        "STP": 0.1113
      },
      "eta_remaining_min": 0.4,
      "reject_reason_if_failed": {
        "code": "BE06",
        "score": 0.5042
      },
      "cancel_proba": 0.9384,
      "return_proba": 0.6303
    }
  ]
}
```

## 5. POST /score/inflight — full stream (outcome messages arrived)

Same UETR after pacs.002 ACSC + camt.054 BOOK: booked snaps ~1, ETA-remaining -> 0, outcome converges to STP.

**Request**: same shape, all 5 messages.

**Response 200**
```json
{
  "model": "model_india",
  "results": [
    {
      "end_to_end_id": "E2E-00020002",
      "n_msgs": 5,
      "last_msg_type": "camt.054",
      "booked_proba": 0.9941,
      "settlement_outcome": {
        "MANUAL_REVIEW": 0.0015,
        "REJECTED": 0.0029,
        "REPAIRED": 0.3439,
        "STP": 0.6517
      },
      "eta_remaining_min": 0.0,
      "reject_reason_if_failed": {
        "code": "RR04",
        "score": 0.0015
      },
      "cancel_proba": 0.8425,
      "return_proba": 0.4231
    }
  ]
}
```

## 6. Error responses

```json
// POST /score/intake with empty body -> 422
{
  "detail": "provide `payments` rows or `pacs008_xml`"
}

// POST /score/inflight with malformed rows -> 422
{
  "detail": "messages are missing columns: ['end_to_end_id', 'msg_type', 'seq']"
}
```

A bundle without the in-flight heads returns **409** with a retrain hint.
