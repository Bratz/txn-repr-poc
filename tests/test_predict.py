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
