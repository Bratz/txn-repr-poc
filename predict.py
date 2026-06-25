"""
Online scoring path — save / load the trained model and score new transactions.

Architecture.md §2 online plane:
  projected pacs.008 row → field encoders (party store = LOOKUP) → frozen tabular
  encoder f → frozen LLM + trained adapters → task label / score.

Multi-task: the saved bundle carries every §5 task (risk / geography / expense /
recurrence) — its task id, instruction tokens, answer tokens, and label set. The
`Scorer` picks one by name; single-record tasks score per row, the multi-record
recurrence task scores per (debtor,creditor) group (Eq. 5). Legacy single-task
checkpoints (only `instruction_ids` / `answer_token_ids`) still load as a lone
"risk" task.

What persists (`save_model`): ONLY our own weights — the frozen tabular encoder
`f`, the trainable adapter trio {Φ, ψ, φ} + [R1…RM], the column vocabs, the
quantizer grids, and the per-task instruction / answer tokens. The LLM (Phi) is
NOT saved — it is frozen and re-downloaded by name at load time. The party-summary
table rides inside the encoder state_dict, so the assembler is rebuilt with
`party_store=None` (zero tables of the right shape) and the real values arrive via
load_state_dict.

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

def _serialize_tasks(tasks, instruction_ids, answer_token_ids, label_values):
    """Build the persisted task list. `tasks` is the run_gpu spec list (with `instr`
    tensors); if None, fall back to a single 'risk' task from the legacy args."""
    if tasks is None:
        return [{"name": "risk", "task_id": 0,
                 "instruction_ids": list(map(int, instruction_ids)),
                 "answer_token_ids": list(map(int, answer_token_ids)),
                 "label_values": list(label_values),
                 "records": "single", "group_column": None}]
    return [{"name": t["name"], "task_id": t["task_id"],
             "instruction_ids": list(map(int, t["instr"])),
             "answer_token_ids": list(map(int, t["answers"])),
             "label_values": list(t["label_values"]),
             "records": t.get("records", "single"),
             "group_column": t.get("group_column")} for t in tasks]


def save_model(save_dir, *, enc_cfg, dec_cfg, vocabs, quantizer, encoder, decoder,
               llm_name, label_values, instruction_ids, answer_token_ids, schema,
               tasks=None):
    """Persist everything needed to reconstruct the scorer (minus the frozen LLM).

    `tasks` (optional) = the run_gpu task-spec list to persist the full §5 suite;
    omitted → a single 'risk' task from instruction_ids/answer_token_ids.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trio_state = {k: v for k, v in decoder.state_dict().items()
                  if not (k.startswith("encoder.") or k.startswith("llm."))}
    tasks_ser = _serialize_tasks(tasks, instruction_ids, answer_token_ids, label_values)
    core = schema["buckets"].get("core", [])
    date_col = "IntrBkSttlmDt" if "IntrBkSttlmDt" in core else None

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
        "tasks": tasks_ser,
        "date_col": date_col,
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
        "tasks": [t["name"] for t in tasks_ser],
    }, indent=2))
    return save_dir / _BUNDLE


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

class Scorer:
    """Loaded model that scores projected transaction rows → per-task distribution.

    `tasks` maps name → {task_id, instruction_ids (tensor), answer_token_ids,
    label_values, records, group_column}. `score(df, task=...)` defaults to the
    risk task; multi-record tasks score per group.
    """

    def __init__(self, decoder, vocabs, tasks, device,
                 default_task="risk", date_col="IntrBkSttlmDt"):
        self.decoder = decoder
        self.vocabs = vocabs
        self.tasks = tasks
        self.device = device
        self.default_task = default_task if default_task in tasks else next(iter(tasks))
        self.date_col = date_col

    @property
    def task_names(self):
        return list(self.tasks)

    def _spec(self, task):
        name = task or self.default_task
        if name not in self.tasks:
            raise ValueError(f"unknown task {name!r}; available: {self.task_names}")
        return name, self.tasks[name]

    def _index(self, full, pos):
        """Slice the encoded batch by integer positions onto the model device."""
        p = np.asarray(pos)
        pt = torch.as_tensor(p, dtype=torch.long)
        return {
            "high_card": {c: t[pt].to(self.device) for c, t in full["high_card"].items()},
            "core": {c: t[pt].to(self.device) for c, t in full["core"].items()},
            "amount": full["amount"][p], "ccy": full["ccy"][p],
        }

    def _predict(self, records, spec, B):
        instr = spec["instruction_ids"].unsqueeze(0).expand(B, -1).to(self.device)
        task = torch.full((B,), spec["task_id"], dtype=torch.long, device=self.device)
        return self.decoder.predict_proba(records, task, instr,
                                          spec["answer_token_ids"]).cpu().numpy()

    @torch.no_grad()
    def _score_single(self, df, spec, batch_size):
        full = self.vocabs.encode(df)
        n, out = len(df), []
        for s in range(0, n, batch_size):
            pos = np.arange(s, min(s + batch_size, n))
            out.append(self._predict(self._index(full, pos), spec, len(pos)))
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(spec["label_values"])))

    @torch.no_grad()
    def _score_multi(self, df, spec, batch_size):
        """Group by (debtor,creditor) and score R-record examples (Eq. 5).

        Returns (proba, group_keys). Groups with fewer than R transactions can't
        fill an R-record example and are skipped.
        """
        R = self.decoder.max_records
        gcol = spec["group_column"]
        dfx = df.reset_index(drop=True)
        full = self.vocabs.encode(dfx)
        groups, keys = [], []
        for gid, sub in dfx.groupby(gcol):
            if self.date_col in dfx.columns:
                sub = sub.sort_values(self.date_col)
            pos = sub.index.to_numpy()
            if len(pos) < R:
                continue
            groups.append(pos[:R]); keys.append(gid)
        out = []
        for s in range(0, len(groups), batch_size):
            gs = groups[s:s + batch_size]
            records = [self._index(full, [g[j] for g in gs]) for j in range(R)]
            out.append(self._predict(records, spec, len(gs)))
        proba = np.concatenate(out, axis=0) if out else np.zeros((0, len(spec["label_values"])))
        return proba, keys

    def score(self, df, task=None, batch_size: int = 256):
        """(N, n_labels) probabilities for `task` (per row, or per group if multi)."""
        _, spec = self._spec(task)
        if spec.get("records") == "multi":
            return self._score_multi(df, spec, batch_size)[0]
        return self._score_single(df, spec, batch_size)

    def label(self, df, task=None, batch_size: int = 256):
        """DataFrame with per-class probs + a `{task}_pred` column.

        Single-record: one row per input row. Multi-record (recurrence): one row
        per (debtor,creditor) group, keyed by the group column.
        """
        name, spec = self._spec(task)
        lv = spec["label_values"]
        if spec.get("records") == "multi":
            import pandas as pd
            proba, keys = self._score_multi(df, spec, batch_size)
            res = pd.DataFrame({spec["group_column"]: keys})
        else:
            proba = self._score_single(df, spec, batch_size)
            res = df.copy()
        for j, cls in enumerate(lv):
            res[f"p_{cls}"] = proba[:, j] if len(proba) else []
        res[f"{name}_pred"] = [lv[i] for i in proba.argmax(axis=1)] if len(proba) else []
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

    # Task map: prefer the saved multi-task list; fall back to a lone "risk" task
    # for legacy checkpoints that only carry instruction_ids / answer_token_ids.
    raw_tasks = b.get("tasks") or [{
        "name": "risk", "task_id": 0,
        "instruction_ids": b["instruction_ids"],
        "answer_token_ids": b["answer_token_ids"],
        "label_values": b["label_values"],
        "records": "single", "group_column": None,
    }]
    tasks = {t["name"]: {
        "task_id": t["task_id"],
        "instruction_ids": torch.tensor(t["instruction_ids"], dtype=torch.long),
        "answer_token_ids": t["answer_token_ids"],
        "label_values": t["label_values"],
        "records": t.get("records", "single"),
        "group_column": t.get("group_column"),
    } for t in raw_tasks}
    default = "risk" if "risk" in tasks else next(iter(tasks))
    return Scorer(decoder, vocabs, tasks, device, default_task=default,
                  date_col=b.get("date_col") or "IntrBkSttlmDt")


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
    ap.add_argument("--task", default=None,
                    help="task to score: risk (default) | geography | expense | recurrence")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    scorer = load_model(args.model_dir)
    name = args.task or scorer.default_task
    if name not in scorer.tasks:
        raise SystemExit(f"unknown --task {name!r}; available: {scorer.task_names}")
    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    res = scorer.label(df, task=name, batch_size=args.batch)
    res.to_csv(args.out, index=False)
    unit = "groups" if scorer.tasks[name]["records"] == "multi" else "rows"
    dist = res[f"{name}_pred"].value_counts().to_dict()
    print(f"scored {len(res):,} {unit} for task '{name}' -> {args.out}  (pred dist: {dist})")


if __name__ == "__main__":
    main()
