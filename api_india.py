"""
Thin advisory API over the persisted India model (serve_india) - FastAPI.

  MODEL_DIR=model_india uvicorn api_india:app --port 8000

Endpoints:
  GET  /model/meta      the bundle contract (rails / statuses / exceptions / heads, inflight)
  POST /score/intake    complete payment rows (JSON) or a raw pacs.008 XML -> intake heads
  POST /score/inflight  a UETR's engine-visible messages seen so far -> booked probability

Advisory-only by design: no endpoint gates a payment - deterministic caps/floors/eligibility
stay in the engine (docs/CBPR_TWIN_GAP.md capability routing). Scores are uncalibrated
balanced-probe rankings; calibrate before showing them to a customer.

ponytail: no auth/rate-limiting/async workers - this is the integration contract, not the
gateway. Put it behind the bank's API gateway for all of that.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_scorer = None
_meta: dict = {}


@asynccontextmanager
async def _lifespan(app):
    global _scorer, _meta
    from serve_india import load_india_model
    d = Path(os.environ.get("MODEL_DIR", "model_india"))
    _scorer = load_india_model(d, device="cpu")
    _meta = {**json.loads((d / "meta.json").read_text()), "model_dir": str(d)}
    yield


app = FastAPI(title="txn-repr India advisory scorer", lifespan=_lifespan)


class IntakeRequest(BaseModel):
    payments: list[dict] | None = None
    pacs008_xml: str | None = None
    explain: bool = False          # column-occlusion drivers (faithful; ~20x embed cost)


class InflightRequest(BaseModel):
    messages: list[dict]


class VelocityRequest(BaseModel):
    transactions: list[dict]       # an account's recent payment rows (>=2), engine-supplied


def _clean(rec: dict) -> dict:
    """JSON-safe record: numpy scalars -> python, floats rounded."""
    out = {}
    for k, v in rec.items():
        if isinstance(v, (np.floating, float)):
            out[k] = round(float(v), 4)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _risks_as_list(s: str) -> list[dict]:
    """'sla_breach 0.70, below_min 0.30' -> [{code, score}, ...]"""
    out = []
    for part in str(s).split(", "):
        code, _, score = part.rpartition(" ")
        if code:
            out.append({"code": code, "score": float(score)})
    return out


@app.get("/model/meta")
def model_meta():
    return _meta


@app.post("/score/intake")
def score_intake(req: IntakeRequest):
    if req.pacs008_xml:
        from data.iso20022_pacs008 import parse_pacs008_frame
        df = parse_pacs008_frame(req.pacs008_xml)
    elif req.payments:
        df = pd.DataFrame(req.payments)
    else:
        raise HTTPException(422, "provide `payments` rows or `pacs008_xml`")
    try:
        res = _scorer.predict(df)
    except KeyError as e:
        raise HTTPException(422, f"payment rows are missing a required feature column: {e}")
    recs = [_clean(r) for r in res.to_dict(orient="records")]
    for r in recs:
        r["top_exception_risks"] = _risks_as_list(r["top_exception_risks"])
    if req.explain:
        if len(df) > 10:
            raise HTTPException(422, "explain=true is capped at 10 payments per request")
        for r, drv in zip(recs, _scorer.explain(df)):
            r["drivers"] = drv
    return {"model": _meta.get("model_dir"), "results": recs}


@app.post("/forecast/next")
def forecast_next(req: VelocityRequest):
    """Next-payment forecast per account from its recent history (engine-supplied):
    probability of another payment within the trained horizon, expected gap and amount,
    and the most-frequent-payee hint."""
    df = pd.DataFrame(req.transactions)
    need = {"DbtrAcct_Id", "CdtrAcct_Id", "IntrBkSttlmDt"}
    if need - set(df.columns):
        raise HTTPException(422, f"transactions need columns: {sorted(need)}")
    try:
        res = _scorer.predict_next(df)
    except SystemExit as e:
        raise HTTPException(409, str(e))
    return {"model": _meta.get("model_dir"),
            "results": [_clean(r) for r in res.to_dict(orient="records")]}


@app.post("/forecast/liquidity")
def forecast_liquidity(req: IntakeRequest):
    """Treasury outflow curve: rail x settlement-time bucket, aggregated from per-payment
    ETA predictions (outward payments only). Point-estimate ETAs - a planning view, not a
    guarantee."""
    if not req.payments:
        raise HTTPException(422, "provide `payments` rows")
    df = pd.DataFrame(req.payments)
    try:
        res = _scorer.liquidity_forecast(df)
    except KeyError as e:
        raise HTTPException(422, f"payment rows are missing a required column: {e}")
    return {"model": _meta.get("model_dir"), **res}


@app.post("/score/velocity")
def score_velocity(req: VelocityRequest):
    df = pd.DataFrame(req.transactions)
    need = {"DbtrAcct_Id", "IntrBkSttlmDt"}
    if need - set(df.columns):
        raise HTTPException(422, f"transactions need columns: {sorted(need)}")
    try:
        res = _scorer.predict_velocity(df)
    except SystemExit as e:
        raise HTTPException(409, str(e))
    return {"model": _meta.get("model_dir"),
            "results": [_clean(r) for r in res.to_dict(orient="records")]}


@app.post("/score/inflight")
def score_inflight(req: InflightRequest):
    df = pd.DataFrame(req.messages)
    missing = {"end_to_end_id", "seq", "msg_type"} - set(df.columns)
    if missing:
        raise HTTPException(422, f"messages are missing columns: {sorted(missing)}")
    try:
        res = _scorer.predict_stream(df)
    except SystemExit as e:                      # bundle without / with stale in-flight head
        raise HTTPException(409, str(e))
    return {"model": _meta.get("model_dir"),
            "results": [_clean(r) for r in res.to_dict(orient="records")]}
