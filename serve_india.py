"""
Persisted India multi-rail model - save / load / predict without retraining.

The deployable model is the FROZEN v1 encoder + the intake probes (no LLM, CPU). This mirrors
predict.py's encoder-reconstruction convention: the party-summary table rides inside the
encoder state_dict, so the assembler is rebuilt with party_store=None and the real weights
arrive via load_state_dict; the quantizer grids and column vocabs are persisted alongside.

Bundle (a directory):
  encoder.pt     torch bundle - enc_cfg, schema buckets, vocabs, quantizer, encoder_state
  probes.joblib  the sklearn intake probes (rail / status / eta / per-exception)
  meta.json      human-readable summary

  python run_india.py --save model_india           # train + persist
  python serve_india.py --model-dir model_india --input data/india_rails_payments.parquet
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from encoder.tabular_encoder import EncoderConfig, TabularEncoder
from encoders.column_assembler import ColumnAssembler, ColumnVocabs
from encoders.quantizer import AdaptiveQuantizer
from run_seq import embed_all_rows

_ENC = "encoder.pt"
_PROBES = "probes.joblib"


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

def save_india_model(save_dir, *, enc_cfg, vocabs, quantizer, encoder, schema, probes):
    """Persist the frozen encoder bundle + intake probes."""
    import joblib
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "enc_cfg": asdict(enc_cfg),
        "schema_buckets": schema["buckets"],
        "twin": schema["twin"],
        "numerical_col": vocabs.numerical_col,
        "ccy_col": vocabs.ccy_col,
        "vocabs": {
            "high_card": vocabs.high_card,
            "high_card_freq": {c: v.tolist() for c, v in vocabs.high_card_freq.items()},
            "core": vocabs.core,
        },
        "quantizer": quantizer.to_dict(),
        "encoder_state": encoder.state_dict(),
    }, save_dir / _ENC)
    joblib.dump(probes, save_dir / _PROBES)
    (save_dir / "meta.json").write_text(json.dumps({
        "rails": list(probes["rail"].classes_),
        "statuses": list(probes["status"].classes_),
        "exceptions": list(probes["exc"].keys()),
        "hidden": enc_cfg.hidden,
    }, indent=2))
    return save_dir


# --------------------------------------------------------------------------- #
# Load + predict
# --------------------------------------------------------------------------- #

class IndiaScorer:
    """Loaded India model: predict rail / status / ETA / exception risks for payment rows."""

    def __init__(self, encoder, vocabs, probes, device):
        self.encoder = encoder
        self.vocabs = vocabs
        self.probes = probes
        self.device = device

    @torch.no_grad()
    def _embed(self, df):
        full = self.vocabs.encode(df)
        return embed_all_rows(self.encoder, full, len(df), self.device).cpu().numpy()

    def predict(self, df, top_exceptions: int = 3):
        """Return a DataFrame: predicted rail (+ confidence), status, ETA, and the top-k
        exception risk scores per payment (uncalibrated balanced-probe rankings)."""
        import pandas as pd
        e = self._embed(df)
        rail_p = self.probes["rail"].predict_proba(e)
        rail_cls = self.probes["rail"].classes_
        out = pd.DataFrame(index=df.index)
        if "payment_id" in df.columns:
            out["payment_id"] = df["payment_id"].to_numpy()
        out["rail_pred"] = rail_cls[rail_p.argmax(1)]
        out["rail_conf"] = rail_p.max(1).round(3)
        out["status_pred"] = self.probes["status"].predict(e)
        out["eta_min_pred"] = np.clip(self.probes["eta"].predict(e), 0, None).round(1)
        risks = {n: m.predict_proba(e)[:, 1] for n, m in self.probes["exc"].items()}
        names = list(risks)
        R = np.vstack([risks[n] for n in names]).T if names else np.zeros((len(df), 0))
        topk = []
        for i in range(len(df)):
            order = np.argsort(-R[i])[:top_exceptions]
            topk.append(", ".join(f"{names[j]} {R[i, j]:.2f}" for j in order))
        out["top_exception_risks"] = topk
        return out.reset_index(drop=True)


def load_india_model(save_dir, device=None) -> IndiaScorer:
    import joblib
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    b = torch.load(Path(save_dir) / _ENC, map_location="cpu", weights_only=False)
    enc_cfg = EncoderConfig(**b["enc_cfg"])
    vocabs = ColumnVocabs(
        high_card=b["vocabs"]["high_card"],
        high_card_freq={c: np.asarray(v) for c, v in b["vocabs"]["high_card_freq"].items()},
        core=b["vocabs"]["core"],
        numerical_col=b["numerical_col"], ccy_col=b["ccy_col"],
    )
    quantizer = AdaptiveQuantizer.from_dict(b["quantizer"])
    schema = {"buckets": b["schema_buckets"]}
    assembler = ColumnAssembler(schema, vocabs, quantizer, party_store=None,
                                embedding_dim=enc_cfg.hidden, high_card_embedder="partitioned")
    encoder = TabularEncoder(assembler, enc_cfg)
    encoder.load_state_dict(b["encoder_state"])
    encoder.freeze()
    encoder.to(device)
    probes = joblib.load(Path(save_dir) / _PROBES)
    return IndiaScorer(encoder, vocabs, probes, device)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Predict with a saved India multi-rail model")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True, help="parquet/csv of projected payment rows")
    ap.add_argument("--out", default="india_predictions.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    scorer = load_india_model(args.model_dir)
    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    if args.limit:
        df = df.head(args.limit)
    res = scorer.predict(df)
    res.to_csv(args.out, index=False)
    print(f"predicted {len(res):,} payments -> {args.out}")
    print(f"rail pred dist: {res['rail_pred'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
