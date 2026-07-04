"""Fire the PULSE APIs with distinct use cases; save labeled round-trips for annexures."""
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
B = "http://127.0.0.1:8123"
FEATS = ["DbtrAcct_Id", "CdtrAcct_Id", "UltmtDbtr_Id", "UltmtCdtr_Id", "IntrBkSttlmAmt",
         "Ccy", "IntrBkSttlmDt", "SttlmMtd", "identifier_type", "Dbtr_Nm", "Cdtr_Nm",
         "UltmtDbtr_Nm", "UltmtCdtr_Nm", "Dbtr_Ctry", "Cdtr_Ctry", "Dbtr_Industry",
         "Cdtr_Industry", "Dbtr_SubIndustry", "Cdtr_SubIndustry"]

inp = pd.read_csv(ROOT / "data" / "test_input_100.csv")
msgs = pd.read_csv(ROOT / "data" / "test_input_messages_100.csv")
cases = {}


def row(pid):
    return json.loads(inp[inp["payment_id"] == pid][FEATS].to_json(orient="records"))[0]


def intake(pid, label, truth_cols=("rail", "terminal_status")):
    r = row(pid)
    resp = requests.post(f"{B}/score/intake", json={"payments": [r]}).json()
    truth = inp[inp["payment_id"] == pid].iloc[0]
    cases[label] = {"request": r, "response": resp["results"][0],
                    "truth": {c: (None if pd.isna(truth[c]) else truth[c]) for c in truth_cols}}


# UC1-UC4: intake scenarios from the held-out 100
intake(20000, "uc1_small_domestic")            # 833 INR, IMPS, STP
intake(20004, "uc2_high_value_rtgs")           # 250,799 INR, RTGS, High risk
intake(20021, "uc3_cross_border_swift")        # 208,988 x-border SWIFT
intake(20050, "uc4_misrouted_below_floor")     # 7,410 tagged RTGS (below 2L floor), true REJECTED

# UC5: cold start - accounts the model has never seen (honest degradation)
cold = dict(row(20000))
cold.update({"DbtrAcct_Id": "ZZNEW0001", "CdtrAcct_Id": "ZZNEW0002",
             "UltmtDbtr_Id": "ZZNEW1", "UltmtCdtr_Id": "ZZNEW2",
             "Dbtr_Nm": "Brand New Trading Co", "Cdtr_Nm": "Unknown Beneficiary Ltd"})
resp = requests.post(f"{B}/score/intake", json={"payments": [cold]}).json()
cases["uc5_cold_start"] = {"request": cold, "response": resp["results"][0],
                           "truth": {"note": "all-new accounts; no truth - the point is the confidence"}}

# UC6: explain - occlusion drivers for the SWIFT payment
resp = requests.post(f"{B}/score/intake", json={"payments": [row(20021)], "explain": True}).json()
cases["uc6_explain_swift"] = {"request": "uc3 payment + explain:true",
                              "response": resp["results"][0]}

# UC7/UC8: in-flight - healthy first-message vs troubled (recall camt.056 + return pacs.004)
def stream(e2e, upto=None, label=None):
    m = msgs[msgs["end_to_end_id"] == e2e].sort_values("seq")
    if upto is not None:
        m = m.head(upto)
    recs = json.loads(m.to_json(orient="records"))
    resp = requests.post(f"{B}/score/inflight", json={"messages": recs}).json()
    cases[label] = {"request_msgs": [f"{r['msg_type']}({r.get('tx_sts')})" for r in recs],
                    "response": resp["results"][0]}


healthy = "E2E-00020000"
troubled = "E2E-00020005"                       # has camt.056 recall AND pacs.004 return
stream(healthy, upto=1, label="uc7_inflight_fresh")
n_pre = len(msgs[(msgs["end_to_end_id"] == troubled)]) - 1
stream(troubled, upto=None, label="uc8_inflight_troubled")

# UC9: velocity - bursting vs quiet account from the dense fleet (same generator, seed 41)
from data.synth_india_rails import IndiaConfig, build_dataset
dense, _, _ = build_dataset(IndiaConfig(num_accounts=500, num_payments=12000, seed=41))
best, quiet = None, None
for a, g in dense.groupby("DbtrAcct_Id"):
    if len(g) < 6 or len(g) > 40:
        continue
    recs = json.loads(g[FEATS].sort_values("IntrBkSttlmDt").to_json(orient="records"))
    r = requests.post(f"{B}/score/velocity", json={"transactions": recs}).json()["results"][0]
    if best is None or r["burst_proba"] > best[1]["burst_proba"]:
        best = (a, r)
    if quiet is None or r["burst_proba"] < quiet[1]["burst_proba"]:
        quiet = (a, r)
    if best[1]["burst_proba"] > 0.6 and quiet[1]["burst_proba"] == 0.0:
        break
cases["uc9_velocity"] = {"request": "per-account history rows (engine-supplied)",
                         "bursting": best[1], "quiet": quiet[1]}

# UC10: next-payment forecast - salaried (cadence) vs irregular customer
cad, _, _ = build_dataset(IndiaConfig(num_accounts=500, num_payments=12000, seed=29,
                                      cadence_frac=0.5))
sal = cad[cad["cadence"] == "salary"]["DbtrAcct_Id"].value_counts()
sal_acct = sal[sal >= 6].index[0]
irr = cad[cad["cadence"] == "none"]["DbtrAcct_Id"].value_counts()
irr_acct = irr[(irr >= 6)].index[0]
h = cad[cad["DbtrAcct_Id"].isin([sal_acct, irr_acct])][FEATS].sort_values("IntrBkSttlmDt")
recs = json.loads(h.to_json(orient="records"))
r = requests.post(f"{B}/forecast/next", json={"transactions": recs}).json()
cases["uc10_forecast_next"] = {"request": "history of one salaried + one irregular customer",
                               "salary_account": sal_acct, "irregular_account": irr_acct,
                               "response": r["results"]}

# UC11: treasury liquidity - the day's outward book (the held-out 100)
recs = json.loads(inp[FEATS].to_json(orient="records"))
r = requests.post(f"{B}/forecast/liquidity", json={"payments": recs}).json()
cases["uc11_liquidity"] = {"request": "the day's book: 100 outward payments", "response": r}

out = Path(__file__).parent / "annex_capture.json"
out.write_text(json.dumps(cases, indent=2, default=str))
print("wrote", out)
for k, v in cases.items():
    print("=" * 8, k)
    print(json.dumps(v, indent=1, default=str)[:700])
