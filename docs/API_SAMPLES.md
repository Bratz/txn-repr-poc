# API samples — txn-repr India advisory scorer

Captured live against `api_india:app` (bundle `model_india`, CPU, **calibrated**). Start:

```bash
MODEL_DIR=model_india uvicorn api_india:app --port 8000
```

> All probabilities are **isotonic-calibrated on held-out data** (`meta.calibrated: true`) — they are readable as probabilities. Remaining caveats: synthetic training data; weak heads (status/reason) stay weak, calibration fixes the scale not the skill; cold-start accounts degrade (watch rail_conf).

## 1. GET /model/meta
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
  "velocity": true,
  "calibrated": true,
  "model_dir": "model_india"
}
```

## 2. POST /score/intake — JSON payment row (the 19 feature fields)

**Request**
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

**Response 200** (exception scores are calibrated rates now, not balanced rankings)

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
          "code": "below_min",
          "score": 0.04
        },
        {
          "code": "format_error",
          "score": 0.04
        },
        {
          "code": "technical_decline",
          "score": 0.03
        }
      ]
    }
  ]
}
```

## 3. POST /score/intake — raw ISO 20022 pacs.008 XML

```json
{"pacs008_xml": "<?xml ... <FIToFICstmrCdtTrf> ..."}
```

**Response 200** — one result per CdtTrfTxInf
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
          "score": 0.09
        },
        {
          "code": "sanctions_hit",
          "score": 0.07
        },
        {
          "code": "format_error",
          "score": 0.05
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
          "code": "sanctions_hit",
          "score": 0.18
        },
        {
          "code": "no_route",
          "score": 0.05
        },
        {
          "code": "fx_fail",
          "score": 0.04
        }
      ]
    }
  ]
}
```

## 4. POST /score/inflight — the predicted-lifecycle object

**pain.001 only** (all future states, calibrated):
```json
{
  "model": "model_india",
  "results": [
    {
      "end_to_end_id": "E2E-00020002",
      "n_msgs": 1,
      "last_msg_type": "pain.001",
      "booked_proba": 0.8421,
      "settlement_outcome": {
        "MANUAL_REVIEW": 0.121,
        "REJECTED": 0.0339,
        "REPAIRED": 0.0964,
        "STP": 0.7487
      },
      "eta_remaining_min": 0.4,
      "reject_reason_if_failed": {
        "code": "BE06",
        "score": 0.5042
      },
      "cancel_proba": 0.1907,
      "return_proba": 0.046
    }
  ]
}
```

**Full stream** (outcome messages arrived — booked snaps, ETA-remaining hits 0):

```json
{
  "model": "model_india",
  "results": [
    {
      "end_to_end_id": "E2E-00020002",
      "n_msgs": 5,
      "last_msg_type": "camt.054",
      "booked_proba": 0.9967,
      "settlement_outcome": {
        "MANUAL_REVIEW": 0.0081,
        "REJECTED": 0.0177,
        "REPAIRED": 0.115,
        "STP": 0.8591
      },
      "eta_remaining_min": 0.0,
      "reject_reason_if_failed": {
        "code": "RR04",
        "score": 0.0015
      },
      "cancel_proba": 0.0588,
      "return_proba": 0.0352
    }
  ]
}
```

## 5. POST /score/velocity — per-account burst (stateless; engine sends the history)

Needs `DbtrAcct_Id` + `IntrBkSttlmDt` + feature columns, >=2 rows per account. `burst_proba` calibrated; `burst_rule` = transparent last-gaps rule for comparison.

```json
{
  "model": "model_india",
  "results": [
    {
      "actor": "ET6FRDFR",
      "n_txns": 11,
      "burst_proba": 0.4444,
      "burst_rule": 0
    },
    {
      "actor": "KWQ5Q4GZ",
      "n_txns": 11,
      "burst_proba": 0.5,
      "burst_rule": 1
    },
    {
      "actor": "RNVX5X4Q",
      "n_txns": 11,
      "burst_proba": 0.32,
      "burst_rule": 0
    },
    {
      "actor": "STIZ7XQD",
      "n_txns": 11,
      "burst_proba": 0.1455,
      "burst_rule": 0
    },
    {
      "actor": "TTUFMYXQ",
      "n_txns": 11,
      "burst_proba": 0.5,
      "burst_rule": 0
    }
  ]
}
```

## 6. POST /score/intake with `explain: true` — faithful drivers (column occlusion)

Capped at 10 payments (~20x embed cost). Occlude a field -> re-embed -> measure the shift. No LLM.

```json
[
  {
    "field": "CdtrAcct_Id",
    "rail_impact": 0.5544,
    "risk_impact": 0.1276,
    "eta_impact_min": -466.3
  },
  {
    "field": "Ccy",
    "rail_impact": -0.032,
    "risk_impact": 0.644,
    "eta_impact_min": 34.0
  },
  {
    "field": "DbtrAcct_Id",
    "rail_impact": 0.5477,
    "risk_impact": -0.1028,
    "eta_impact_min": -371.6
  },
  {
    "field": "UltmtCdtr_Id",
    "rail_impact": -0.0664,
    "risk_impact": 0.2246,
    "eta_impact_min": -14.8
  },
  {
    "field": "identifier_type",
    "rail_impact": -0.2732,
    "risk_impact": 0.0023,
    "eta_impact_min": -177.5
  }
]
```

## 7. Errors
```json
// empty intake -> 422
{
  "detail": "provide `payments` rows or `pacs008_xml`"
}

// malformed stream -> 422
{
  "detail": "messages are missing columns: ['end_to_end_id', 'msg_type', 'seq']"
}
```

Bundle without in-flight/velocity heads -> **409** with a retrain hint.
