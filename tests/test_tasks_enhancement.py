"""Tests for the four-task enhancement (risk / geography / expense / recurrence).

Covers the generator labels + tasks manifest, the per-task metric router, and the
run_gpu multi-task / multi-record helpers. The multi-record decoder path is also
exercised here (single-record paths are covered in test_multimodal_decoder.py).
"""

import numpy as np
import pandas as pd
import torch

from data.synth_pacs008 import (
    EXPENSE_TYPES,
    GEO_SPANS,
    GenConfig,
    TASKS,
    build_dataset,
)


# --------------------------------------------------------------------------- #
# Generator: labels + grouping
# --------------------------------------------------------------------------- #

def test_generator_emits_all_four_task_labels():
    df, _ = build_dataset(GenConfig(num_parents=80, num_transactions=2000, seed=1))
    for col in ("risk_label", "geo_label", "expense_label", "recurrence_label", "group_id"):
        assert col in df.columns
    assert set(df["geo_label"]).issubset(set(GEO_SPANS))
    assert set(df["expense_label"]).issubset(set(EXPENSE_TYPES))
    assert set(df["recurrence_label"]).issubset({"No", "Yes"})
    # both recurrence classes are present at this scale
    assert {"No", "Yes"} <= set(df["recurrence_label"])


def test_recurring_groups_are_regularly_spaced():
    df, _ = build_dataset(GenConfig(num_parents=80, num_transactions=4000, seed=2))
    rec = df[df["recurrence_label"] == "Yes"]
    assert len(rec) > 0
    # every recurring group has >= recur_min_txns records...
    sizes = rec.groupby("group_id").size()
    assert (sizes >= GenConfig.recur_min_txns).all()
    # ...at a single fixed interval (the day gaps within a group are constant)
    def day_gaps(s):
        days = pd.to_datetime(s).sort_values().diff().dropna().dt.days
        return days.to_numpy()
    gaps = rec.groupby("group_id")["IntrBkSttlmDt"].apply(day_gaps)
    a_group = gaps.iloc[0]
    assert len(set(a_group.tolist())) == 1          # one constant interval


def test_schema_tasks_manifest_well_formed():
    names = [t["name"] for t in TASKS]
    assert names == ["risk", "geography", "expense", "recurrence"]
    recur = next(t for t in TASKS if t["name"] == "recurrence")
    assert recur["records"] == "multi" and recur["group_column"] == "group_id"
    assert all(t["records"] == "single" for t in TASKS if t["name"] != "recurrence")


# --------------------------------------------------------------------------- #
# Eval: per-task metric routing
# --------------------------------------------------------------------------- #

def test_evaluate_task_routes_multiclass_vs_binary():
    from eval.metrics import evaluate_task

    multiclass = {"name": "geography", "metric": "multiclass",
                  "label_values": ["A", "B", "C"]}
    y = np.array(["A", "B", "C", "A"])
    proba = np.eye(3)[[0, 1, 2, 0]]                  # perfect predictions
    m = evaluate_task(multiclass, y, proba)
    assert m["metric"] == "multiclass" and m["accuracy"] == 1.0

    binary = {"name": "recurrence", "metric": "binary", "positive_class": "Yes",
              "label_values": ["No", "Yes"]}
    yb = np.array(["No", "Yes", "No", "Yes"])
    pb = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    mb = evaluate_task(binary, yb, pb)
    assert mb["metric"] == "binary" and mb["pr_auc"] > 0.9


# --------------------------------------------------------------------------- #
# run_gpu: task specs + example construction
# --------------------------------------------------------------------------- #

def test_build_task_specs_smoke_distinct_answers(schema):
    from decoder.multimodal_decoder import MockLLM
    from run_gpu import build_task_specs

    llm = MockLLM(vocab_size=64, hidden=16, num_layers=1, num_heads=2)
    specs = build_task_specs(schema, llm, smoke=True, device="cpu")
    assert [s["task_id"] for s in specs] == list(range(len(specs)))
    for s in specs:
        assert len(set(s["answers"])) == len(s["label_values"])  # distinct per label


def test_recurrence_groups_have_R_records(sample_df):
    from run_gpu import _recurrence_groups

    task = next(t for t in TASKS if t["name"] == "recurrence")
    df = sample_df.reset_index(drop=True)
    groups, labels = _recurrence_groups(df, task, R=3)
    assert len(groups) == len(labels)
    if groups:                                       # sample may be small
        assert all(len(g) == 3 for g in groups)
        assert set(labels) <= {"No", "Yes"}


# --------------------------------------------------------------------------- #
# Decoder: multi-record interleaving (Eq. 5)
# --------------------------------------------------------------------------- #

def test_multi_record_build_inputs_grows_by_two_tokens_per_record(schema, sample_df):
    from decoder.multimodal_decoder import DecoderConfig, MockLLM, MultimodalDecoder
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack

    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(
        sample_df, schema, EncoderConfig(hidden=16, layers=1, heads=2, ff_mult=2),
        party_epochs=1)
    llm = MockLLM(vocab_size=64, hidden=16, num_layers=1, num_heads=2)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(
        n_tasks=4, max_records=3, adapter_heads=2, prefix_len=4, phi_mode="prompt"))
    batch = vocabs.encode(sample_df.head(6))
    task = torch.zeros(6, dtype=torch.long)
    instr = torch.randint(0, 64, (6, 4))

    z1, _ = dec.build_inputs(batch, task, instr)                # single record
    z3, _ = dec.build_inputs([batch, batch, batch], task, instr)  # three records
    # each extra record adds exactly its sentinel + one adapter token
    assert (z3.shape[1] - z1.shape[1]) == 2 * (3 - 1)
    # predict_proba over the multi-record input is a well-formed 2-class dist
    p = dec.predict_proba([batch, batch, batch], task, instr, [0, 1])
    assert p.shape == (6, 2)
    assert torch.allclose(p.sum(1), torch.ones(6), atol=1e-4)


def test_too_many_records_raises(schema, sample_df):
    from decoder.multimodal_decoder import DecoderConfig, MockLLM, MultimodalDecoder
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    import pytest

    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(
        sample_df, schema, EncoderConfig(hidden=16, layers=1, heads=2, ff_mult=2),
        party_epochs=1)
    llm = MockLLM(vocab_size=64, hidden=16, num_layers=1, num_heads=2)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(n_tasks=4, max_records=2,
                                                    adapter_heads=2, prefix_len=4))
    batch = vocabs.encode(sample_df.head(4))
    task = torch.zeros(4, dtype=torch.long)
    instr = torch.randint(0, 64, (4, 4))
    with pytest.raises(ValueError, match="exceeds max_records"):
        dec.build_inputs([batch, batch, batch], task, instr)
