"""Unique-case test data (momentum + cadence fleet) fired at the live APIs."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
B = "http://127.0.0.1:8123"
FEATS = ["DbtrAcct_Id", "CdtrAcct_Id", "UltmtDbtr_Id", "UltmtCdtr_Id", "IntrBkSttlmAmt",
         "Ccy", "IntrBkSttlmDt", "SttlmMtd", "identifier_type", "Dbtr_Nm", "Cdtr_Nm",
         "UltmtDbtr_Nm", "UltmtCdtr_Nm", "Dbtr_Ctry", "Cdtr_Ctry", "Dbtr_Industry",
         "Cdtr_Industry", "Dbtr_SubIndustry", "Cdtr_SubIndustry"]

from data.synth_india_rails import IndiaConfig, build_dataset, build_messages

pay, evt, _ = build_dataset(IndiaConfig(num_accounts=500, num_payments=12000, seed=23,
                                        cadence_frac=0.4, recall_momentum=1.0))
msg = build_messages(pay, evt)
out = {}


def hist_records(actor):
    h = pay[pay["DbtrAcct_Id"] == actor][FEATS].sort_values("IntrBkSttlmDt")
    return json.loads(h.to_json(orient="records"))


def trail(pid):
    m = msg[msg["payment_id"] == pid].sort_values("seq")
    return json.loads(m.to_json(orient="records")), \
        [f"{r.msg_type}({r.tx_sts})" for r in m.itertuples()]


# U1: repeat-recall account under momentum - full entity view via three APIs
rec = pay[pay["cancel_requested"] == 1].groupby("DbtrAcct_Id").size()
actor1 = rec[rec >= 2].index[0]
a1 = pay[pay["DbtrAcct_Id"] == actor1].sort_values("IntrBkSttlmDt")
recalled_pid = int(a1[a1["cancel_requested"] == 1]["payment_id"].iloc[-1])
recs, types = trail(recalled_pid)
out["U1_repeat_recall_entity"] = {
    "actor": actor1, "n_payments": int(len(a1)),
    "n_recalls": int(a1["cancel_requested"].sum()),
    "recalled_payment_trail": types,
    "inflight": requests.post(f"{B}/score/inflight", json={"messages": recs}).json()["results"][0],
    "velocity": requests.post(f"{B}/score/velocity",
                              json={"transactions": hist_records(actor1)}).json()["results"][0],
    "forecast_next": requests.post(f"{B}/forecast/next",
                                   json={"transactions": hist_records(actor1)}).json()["results"][0],
}

# U2: recall outcomes contrast - CNCL (funds returned) vs RJCR (recall too late)
cncl = pay[(pay["cancel_status"] == "CNCL") & (pay["returned"] == 1)]["payment_id"].iloc[0]
rjcr = pay[pay["cancel_status"] == "RJCR"]["payment_id"].iloc[0]
for tag, pid in (("recall_succeeded_CNCL", int(cncl)), ("recall_too_late_RJCR", int(rjcr))):
    recs, types = trail(pid)
    out.setdefault("U2_recall_contrast", {})[tag] = {
        "trail": types,
        "inflight": requests.post(f"{B}/score/inflight",
                                  json={"messages": recs}).json()["results"][0]}

# U3: salary-cadence account -> forecast
sal = pay[pay["cadence"] == "salary"]["DbtrAcct_Id"].value_counts()
actor3 = sal[sal >= 8].index[0]
out["U3_salary_cadence_forecast"] = {
    "actor": actor3,
    "forecast_next": requests.post(f"{B}/forecast/next",
                                   json={"transactions": hist_records(actor3)}).json()["results"][0]}

# U4: burst account - tightest recent gaps in the fleet
g = pay.groupby("DbtrAcct_Id")["IntrBkSttlmDt"].agg(
    lambda s: np.median(np.diff(np.sort(pd.to_datetime(s).values))
                        .astype("timedelta64[D]").astype(int)) if len(s) > 5 else 999)
actor4 = g.idxmin()
out["U4_burst_account"] = {
    "actor": actor4, "median_gap_days": float(g.min()),
    "velocity": requests.post(f"{B}/score/velocity",
                              json={"transactions": hist_records(actor4)}).json()["results"][0]}

# U5: /score/ask guard - the local bundle carries no decoder tier
r = requests.post(f"{B}/score/ask", json={"payments": hist_records(actor1)[:1]})
out["U5_ask_guard"] = {"status_code": r.status_code, "detail": r.json().get("detail")}

# U6: four rails, one intake batch (eligibility guard visible in one response)
rows = []
for rail_amt, xb in ((5_000, False), (150_000, False), (900_000, False), (200_000, True)):
    row = dict(hist_records(actor1)[0])
    row["IntrBkSttlmAmt"] = float(rail_amt)
    if xb:
        row["Cdtr_Ctry"] = "US"; row["identifier_type"] = "BIC_IBAN"; row["SttlmMtd"] = "COVE"
    rows.append(row)
out["U6_four_rails_one_batch"] = requests.post(
    f"{B}/score/intake", json={"payments": rows}).json()["results"]

p = Path(__file__).parent / "unique_capture.json"
p.write_text(json.dumps(out, indent=2, default=str))
print("wrote", p)
print(json.dumps(out, indent=1, default=str)[:5200])
