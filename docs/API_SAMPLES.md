# API samples — txn-repr India advisory scorer

Captured live against `api_india:app` (bundle `model_india`, GPU-trained on an H200, served on
CPU, **calibrated**). Start:

```bash
MODEL_DIR=model_india uvicorn api_india:app --port 8000
```

> All probabilities are **isotonic-calibrated on held-out data** (`meta.calibrated: true`) — they
> are readable as probabilities. Remaining caveats: synthetic training data; weak heads
> (status/reason) stay weak, calibration fixes the scale not the skill; cold-start accounts
> degrade (watch rail_conf).

## 1. GET /model/meta
```bash
curl http://localhost:8000/model/meta
```

```json
{
  "rails": ["IMPS", "NEFT", "RTGS", "SWIFT"],
  "statuses": ["MANUAL_REVIEW", "REJECTED", "REPAIRED", "STP"],
  "exceptions": [
    "account_closed", "batch_return", "below_min", "beneficiary_unreachable",
    "format_error", "fraud_hold", "fx_fail", "insufficient_liquidity",
    "limit_exceeded", "missing_field", "no_cover", "no_route",
    "sanctions_hit", "settlement_fail", "sla_breach", "technical_decline"
  ],
  "tasks": ["risk", "geography", "expense"],
  "inflight": true,
  "velocity": true,
  "next": true,
  "calibrated": true,
  "hidden": 512,
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

**Response 200** (exception scores are calibrated rates, not balanced rankings)

```json
{
  "model": "model_india",
  "results": [
    {
      "rail_pred": "IMPS",
      "rail_conf": 0.624,
      "status_pred": "STP",
      "eta_min_pred": 0.0,
      "risk_pred": "Low",
      "geography_pred": "Asia",
      "expense_pred": "Operational",
      "top_exception_risks": [
        {"code": "format_error", "score": 0.04},
        {"code": "below_min", "score": 0.04},
        {"code": "fraud_hold", "score": 0.03}
      ]
    }
  ]
}
```

Raw ISO 20022 works too: `{"pacs008_xml": "<?xml ... <FIToFICstmrCdtTrf> ..."}` returns one
result per `CdtTrfTxInf` (see the XML fields parsed in `data/iso20022_pacs008.py`).

## 3. POST /score/inflight — the predicted-lifecycle object

**First message only** (all future states predicted, calibrated):
```json
{
  "model": "model_india",
  "results": [
    {
      "end_to_end_id": "E2E-00020000",
      "n_msgs": 1,
      "last_msg_type": "pacs.008",
      "booked_proba": 0.9571,
      "settlement_outcome": {
        "MANUAL_REVIEW": 0.0739,
        "REJECTED": 0.0248,
        "REPAIRED": 0.1538,
        "STP": 0.7475
      },
      "eta_remaining_min": 1.0,
      "reject_reason_if_failed": {"code": "ED05", "score": 0.5882},
      "cancel_proba": 0.0259,
      "return_proba": 0.0063
    }
  ]
}
```

**Full stream** (outcome messages arrived — booked snaps to certainty, ETA-remaining hits 0):

```json
{
  "model": "model_india",
  "results": [
    {
      "end_to_end_id": "E2E-00020000",
      "n_msgs": 3,
      "last_msg_type": "camt.054",
      "booked_proba": 1.0,
      "settlement_outcome": {
        "MANUAL_REVIEW": 0.0075,
        "REJECTED": 0.0006,
        "REPAIRED": 0.1217,
        "STP": 0.8701
      },
      "eta_remaining_min": 0.1,
      "reject_reason_if_failed": {"code": "FRAD", "score": 0.0019},
      "cancel_proba": 0.0259,
      "return_proba": 0.0156
    }
  ]
}
```

## 4. POST /score/velocity — per-account burst (stateless; engine sends the history)

Needs `DbtrAcct_Id` + `IntrBkSttlmDt` + feature columns, >=2 rows per account. `burst_proba`
calibrated; `burst_rule` = transparent last-gaps rule for comparison. These sample accounts are
quiet (proba 0.0) — the rule fires once where the model, seeing amounts and counterparties too,
disagrees.

```json
{
  "model": "model_india",
  "results": [
    {"actor": "2I4F3EFZ", "n_txns": 6, "burst_proba": 0.0, "burst_rule": 0},
    {"actor": "99BKVXQD", "n_txns": 7, "burst_proba": 0.0, "burst_rule": 1},
    {"actor": "A05F7IOL", "n_txns": 6, "burst_proba": 0.0, "burst_rule": 0},
    {"actor": "CUCJ6QFK", "n_txns": 7, "burst_proba": 0.0, "burst_rule": 0}
  ]
}
```

## 5. POST /forecast/next — next-payment forecast per account

Same request shape as velocity (engine-supplied history, >=3 rows per account). Occurrence
probability is calibrated against the trained 35-day horizon; gap/amount are Ridge point
estimates; `top_payee_hint` is the retrieval baseline (most-frequent payee).

```json
{
  "model": "model_india",
  "results": [
    {
      "actor": "99BKVXQD",
      "n_history": 7,
      "next_event_proba": 0.6103,
      "expected_gap_days": 27.5,
      "expected_amount": 2518032.75,
      "top_payee_hint": "47DCA90J",
      "horizon_days": 35
    },
    {
      "actor": "CUCJ6QFK",
      "n_history": 7,
      "next_event_proba": 0.0,
      "expected_gap_days": 12.8,
      "expected_amount": 441268.81,
      "top_payee_hint": "3DL979KR",
      "horizon_days": 35
    },
    {
      "actor": "A05F7IOL",
      "n_history": 6,
      "next_event_proba": 0.0,
      "expected_gap_days": 366.4,
      "expected_amount": 377884.38,
      "top_payee_hint": "2W1L4HCR",
      "horizon_days": 35
    }
  ]
}
```

Honesty note (measured, results_next.json): the "when" head beats the naive gap baselines
(MAE 23.5d vs 25.9/32.5); occurrence and amount do NOT yet beat trees/median-naive — use
`expected_gap_days` with confidence, treat the other two as advisory.

## 6. POST /forecast/liquidity — treasury outflow curve

Send the day's outward book; get predicted settlement-time buckets per rail
(amount-conserving: bucket sums equal the book total).

```json
{
  "model": "model_india",
  "buckets": ["<1m", "1-15m", "15-60m", "60-240m", "240-1440m", ">1440m"],
  "by_rail": {
    "IMPS":  [37050.91, 0.0, 3975.88, 14208.57, 176610.42, 51850.99],
    "NEFT":  [113832.09, 178928.58, 0.0, 54790.55, 602484.36, 62716.79],
    "RTGS":  [0.0, 0.0, 250799.11, 0.0, 436447.05, 200199.28],
    "SWIFT": [0.0, 0.0, 0.0, 0.0, 1093808.58, 380952.58]
  },
  "total": [150883.0, 178928.58, 254774.99, 68999.12, 2309350.41, 695719.64],
  "n_payments": 100,
  "total_amount": 3658655.74
}
```

## 7. POST /score/intake with `explain: true` — faithful drivers (column occlusion)

Capped at 10 payments (~20x embed cost). Occlude a field -> re-embed -> measure the shift. No LLM.

```json
[
  {"field": "CdtrAcct_Id",     "rail_impact": 0.493,   "risk_impact": -0.1355, "eta_impact_min": -602.4},
  {"field": "Ccy",             "rail_impact": -0.0329, "risk_impact": 0.4939,  "eta_impact_min": 23.0},
  {"field": "DbtrAcct_Id",     "rail_impact": 0.3669,  "risk_impact": -0.1443, "eta_impact_min": -197.2},
  {"field": "identifier_type", "rail_impact": -0.3526, "risk_impact": 0.0852,  "eta_impact_min": -131.3},
  {"field": "IntrBkSttlmAmt",  "rail_impact": -0.2099, "risk_impact": 0.0458,  "eta_impact_min": -18.3}
]
```

## 8. Errors
```json
// empty intake -> 422
{"detail": "provide `payments` rows or `pacs008_xml`"}

// malformed stream -> 422
{"detail": "messages are missing columns: ['end_to_end_id', 'msg_type', 'seq']"}
```

Bundle without in-flight/velocity/next heads -> **409** with a retrain hint.
