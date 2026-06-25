"""Save / load / score round-trip for the online inference path (predict.py)."""

import numpy as np
import torch

from decoder.multimodal_decoder import DecoderConfig, MockLLM, MultimodalDecoder
from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
from predict import Scorer, load_model, save_model

TINY = EncoderConfig(hidden=32, layers=2, heads=2, ff_mult=2, dropout=0.0, epochs=1)


def _train_tiny(schema, df):
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY, party_epochs=1)
    enc.freeze()
    llm = MockLLM(vocab_size=64, hidden=32, num_layers=2, num_heads=4)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(n_tasks=1, prefix_len=4,
                                                    adapter_heads=4, phi_mode="prompt"))
    return enc, asm, vocabs, llm, dec


def test_save_load_predict_roundtrip(tmp_path, schema, sample_df):
    head = sample_df.head(12)
    enc, asm, vocabs, llm, dec = _train_tiny(schema, sample_df)
    instr = torch.randint(0, 64, (4,))
    answers = [0, 1, 2]

    # reference prediction from the in-memory model
    batch = vocabs.encode(head)
    p1 = dec.predict_proba(batch, torch.zeros(12, dtype=torch.long),
                           instr.unsqueeze(0).expand(12, -1), answers).numpy()

    save_model(tmp_path, enc_cfg=TINY, dec_cfg=dec.config, vocabs=vocabs,
               quantizer=asm.amt_emb.quantizer, encoder=enc, decoder=dec,
               llm_name="mock", label_values=schema["label_values"],
               instruction_ids=instr, answer_token_ids=answers, schema=schema)

    # reload (reuse the same frozen MockLLM, as Phi weights would be deterministic)
    scorer = load_model(tmp_path, device="cpu", llm=llm)
    assert isinstance(scorer, Scorer)
    p2 = scorer.score(head)

    # identical weights + same frozen LLM → identical scores
    np.testing.assert_allclose(p1, p2, atol=1e-5)


def test_label_output_shape_and_values(tmp_path, schema, sample_df):
    head = sample_df.head(10)
    enc, asm, vocabs, llm, dec = _train_tiny(schema, sample_df)
    save_model(tmp_path, enc_cfg=TINY, dec_cfg=dec.config, vocabs=vocabs,
               quantizer=asm.amt_emb.quantizer, encoder=enc, decoder=dec,
               llm_name="mock", label_values=schema["label_values"],
               instruction_ids=torch.tensor([1, 2, 3, 4]), answer_token_ids=[0, 1, 2],
               schema=schema)
    scorer = load_model(tmp_path, device="cpu", llm=llm)

    res = scorer.label(head)
    assert len(res) == len(head)
    for name in schema["label_values"]:
        assert f"p_{name}" in res.columns
    assert set(res["risk_pred"]) <= set(schema["label_values"])
    # per-row probabilities sum to 1
    probs = res[[f"p_{n}" for n in schema["label_values"]]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(head)), atol=1e-5)


def _train_multitask(schema, df):
    torch.manual_seed(0)
    enc, asm, vocabs = build_pretraining_stack(df, schema, TINY, party_epochs=1)
    enc.freeze()
    llm = MockLLM(vocab_size=64, hidden=32, num_layers=2, num_heads=4)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(
        n_tasks=len(schema["tasks"]), max_records=3, prefix_len=4,
        adapter_heads=4, phi_mode="prompt"))
    specs = [{"name": t["name"], "task_id": i, "instr": torch.randint(0, 64, (4,)),
              "answers": list(range(len(t["label_values"]))),
              "label_values": t["label_values"], "records": t.get("records", "single"),
              "group_column": t.get("group_column")}
             for i, t in enumerate(schema["tasks"])]
    return enc, asm, vocabs, llm, dec, specs


def test_multitask_scorer_routes_single_and_multi(tmp_path, schema, sample_df):
    enc, asm, vocabs, llm, dec, specs = _train_multitask(schema, sample_df)
    risk = next(s for s in specs if s["name"] == "risk")
    save_model(tmp_path, enc_cfg=TINY, dec_cfg=dec.config, vocabs=vocabs,
               quantizer=asm.amt_emb.quantizer, encoder=enc, decoder=dec,
               llm_name="mock", label_values=risk["label_values"],
               instruction_ids=risk["instr"], answer_token_ids=risk["answers"],
               schema=schema, tasks=specs)
    scorer = load_model(tmp_path, device="cpu", llm=llm)

    assert scorer.task_names == ["risk", "geography", "expense", "recurrence"]

    # single-record task: one row per input row, probs aligned to that task's labels
    geo = next(t for t in schema["tasks"] if t["name"] == "geography")
    rg = scorer.label(sample_df.head(20), task="geography")
    assert len(rg) == 20
    for cls in geo["label_values"]:
        assert f"p_{cls}" in rg.columns
    assert set(rg["geography_pred"]) <= set(geo["label_values"])

    # multi-record task: one row per (debtor,creditor) group, keyed by group_id
    rr = scorer.label(sample_df, task="recurrence")
    assert "group_id" in rr.columns and len(rr) > 0
    assert {"p_No", "p_Yes"} <= set(rr.columns)
    probs = rr[["p_No", "p_Yes"]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(rr)), atol=1e-5)


def test_unknown_task_raises(tmp_path, schema, sample_df):
    import pytest
    enc, asm, vocabs, llm, dec, specs = _train_multitask(schema, sample_df)
    risk = next(s for s in specs if s["name"] == "risk")
    save_model(tmp_path, enc_cfg=TINY, dec_cfg=dec.config, vocabs=vocabs,
               quantizer=asm.amt_emb.quantizer, encoder=enc, decoder=dec,
               llm_name="mock", label_values=risk["label_values"],
               instruction_ids=risk["instr"], answer_token_ids=risk["answers"],
               schema=schema, tasks=specs)
    scorer = load_model(tmp_path, device="cpu", llm=llm)
    with pytest.raises(ValueError, match="unknown task"):
        scorer.score(sample_df.head(4), task="nope")
