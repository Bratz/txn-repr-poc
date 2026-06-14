"""
Online scoring path — save / load the trained model and score new transactions.

Architecture.md §2 online plane:
  projected pacs.008 row → field encoders (party store = LOOKUP) → frozen tabular
  encoder f → frozen LLM + trained adapters → risk label / score.

What persists (`save_model`): ONLY our own weights — the frozen tabular encoder
`f`, the trainable adapter trio {Φ, ψ, φ} + [R1], the column vocabs, the quantizer
grids, and the resolved instruction / answer tokens. The LLM (Phi) is NOT saved —
it is frozen and re-downloaded by name at load time. The party-summary table rides
inside the encoder state_dict, so the assembler is rebuilt with `party_store=None`
(zero tables of the right shape) and the real values arrive via load_state_dict.

NOTE: input is an already-projected row (the column_schema.json columns). Parsing
raw pacs.008 XML into a row is the live-Layer-1 piece and is out of scope for v1
(the generator's `project_to_pacs008` is the reference projection).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from decoder.multimodal_decoder import DecoderConfig, MultimodalDecoder
from encoder.tabular_encoder import EncoderConfig, TabularEncoder
from encoders.column_assembler import ColumnAssembler, ColumnVocabs
from encoders.quantizer import AdaptiveQuantizer

_BUNDLE = "model.pt"


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #

def save_model(save_dir, *, enc_cfg, dec_cfg, vocabs, quantizer, encoder, decoder,
               llm_name, label_values, instruction_ids, answer_token_ids, schema):
    """Persist everything needed to reconstruct the scorer (minus the frozen LLM)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trio_state = {k: v for k, v in decoder.state_dict().items()
                  if not (k.startswith("encoder.") or k.startswith("llm."))}

    bundle = {
        "enc_cfg": asdict(enc_cfg),
        "dec_cfg": asdict(dec_cfg),
        "llm_name": llm_name,
        "phi_mode": decoder.phi_mode,
        "label_values": list(label_values),
        "numerical_col": vocabs.numerical_col,
        "ccy_col": vocabs.ccy_col,
        "instruction_ids": list(map(int, instruction_ids)),
        "answer_token_ids": list(map(int, answer_token_ids)),
        "schema_buckets": schema["buckets"],
        "vocabs": {
            "high_card": vocabs.high_card,
            "high_card_freq": {c: v.tolist() for c, v in vocabs.high_card_freq.items()},
            "core": vocabs.core,
        },
        "quantizer": quantizer.to_dict(),
        "encoder_state": encoder.state_dict(),
        "trio_state": trio_state,
    }
    torch.save(bundle, save_dir / _BUNDLE)
    (save_dir / "meta.json").write_text(json.dumps({
        "llm_name": llm_name, "label_values": list(label_values),
        "phi_mode": decoder.phi_mode, "hidden": enc_cfg.hidden,
    }, indent=2))
    return save_dir / _BUNDLE


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

class Scorer:
    """Loaded model that scores projected transaction rows → risk distribution."""

    def __init__(self, decoder, vocabs, label_values, instruction_ids,
                 answer_token_ids, device):
        self.decoder = decoder
        self.vocabs = vocabs
        self.label_values = label_values
        self.instruction_ids = instruction_ids
        self.answer_token_ids = answer_token_ids
        self.device = device

    @torch.no_grad()
    def score(self, df, batch_size: int = 256):
        """Return an (N, n_labels) probability array aligned to label_values."""
        full = self.vocabs.encode(df)
        n = len(df)
        out = []
        for s in range(0, n, batch_size):
            idx = slice(s, min(s + batch_size, n))
            sub = {
                "high_card": {c: t[idx].to(self.device) for c, t in full["high_card"].items()},
                "core": {c: t[idx].to(self.device) for c, t in full["core"].items()},
                "amount": full["amount"][idx], "ccy": full["ccy"][idx],
            }
            b = sub["high_card"][next(iter(sub["high_card"]))].shape[0]
            instr = self.instruction_ids.unsqueeze(0).expand(b, -1).to(self.device)
            task = torch.zeros(b, dtype=torch.long, device=self.device)
            p = self.decoder.predict_proba(sub, task, instr, self.answer_token_ids)
            out.append(p.cpu().numpy())
        return np.concatenate(out, axis=0)

    def label(self, df, **kw):
        """Return a DataFrame copy with per-class probs + predicted risk label."""
        proba = self.score(df, **kw)
        res = df.copy()
        for j, name in enumerate(self.label_values):
            res[f"p_{name}"] = proba[:, j]
        res["risk_pred"] = [self.label_values[i] for i in proba.argmax(axis=1)]
        return res


def load_model(save_dir, device=None, llm=None) -> Scorer:
    """Rebuild encoder + decoder from a checkpoint. `llm` overrides the frozen LLM
    (used by tests with a MockLLM); otherwise an HFCausalLM(llm_name) is built."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    b = torch.load(Path(save_dir) / _BUNDLE, map_location="cpu", weights_only=False)

    enc_cfg = EncoderConfig(**b["enc_cfg"])
    dec_cfg = DecoderConfig(**b["dec_cfg"])
    vocabs = ColumnVocabs(
        high_card=b["vocabs"]["high_card"],
        high_card_freq={c: np.asarray(v) for c, v in b["vocabs"]["high_card_freq"].items()},
        core=b["vocabs"]["core"],
        numerical_col=b["numerical_col"], ccy_col=b["ccy_col"],
    )
    quantizer = AdaptiveQuantizer.from_dict(b["quantizer"])
    schema = {"buckets": b["schema_buckets"]}

    # assembler with NO store (zero party tables) — weights come from encoder_state.
    assembler = ColumnAssembler(schema, vocabs, quantizer, party_store=None,
                                embedding_dim=enc_cfg.hidden, high_card_embedder="partitioned")
    encoder = TabularEncoder(assembler, enc_cfg)
    encoder.load_state_dict(b["encoder_state"])
    encoder.freeze()

    if llm is None:
        from decoder.multimodal_decoder import HFCausalLM
        llm = HFCausalLM(b["llm_name"])
    decoder = MultimodalDecoder(encoder, llm, dec_cfg).to(device)
    decoder.load_state_dict(b["trio_state"], strict=False)   # encoder/llm keys absent
    decoder.eval()

    return Scorer(
        decoder, vocabs, b["label_values"],
        torch.tensor(b["instruction_ids"], dtype=torch.long),
        b["answer_token_ids"], device,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    import argparse

    import pandas as pd

    ap = argparse.ArgumentParser(description="Score projected pacs.008 rows with a saved model")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True, help="parquet/csv of projected rows")
    ap.add_argument("--out", default="scored.csv")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    scorer = load_model(args.model_dir)
    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    res = scorer.label(df, batch_size=args.batch)
    res.to_csv(args.out, index=False)
    dist = res["risk_pred"].value_counts().to_dict()
    print(f"scored {len(res):,} rows -> {args.out}  (pred dist: {dist})")


if __name__ == "__main__":
    main()
