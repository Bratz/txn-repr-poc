"""The v2 precomputed-feature path on the Layer-4 decoder (Option B enabler).

Feeding a (B, D_enc) tensor record must equal encoding a batch dict, so an externally
computed entity representation h_USR can be scored by the frozen LLM (C5).
"""

import torch

from decoder.multimodal_decoder import DecoderConfig, MockLLM, MultimodalDecoder
from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack


def _decoder(schema, sample_df):
    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(
        sample_df, schema, EncoderConfig(hidden=16, layers=1, heads=2, ff_mult=2),
        party_epochs=1)
    enc.freeze()
    llm = MockLLM(vocab_size=64, hidden=16, num_layers=1, num_heads=2)
    dec = MultimodalDecoder(enc, llm, DecoderConfig(n_tasks=1, phi_mode="prompt"))
    return enc, vocabs, dec


def test_tensor_record_equals_encoded_batch(schema, sample_df):
    enc, vocabs, dec = _decoder(schema, sample_df)
    batch = vocabs.encode(sample_df.head(6))
    task = torch.zeros(6, dtype=torch.long)
    instr = torch.randint(0, 64, (6, 4))
    f = enc.encode(batch)                                   # (6, D_enc)

    z_dict, _ = dec.build_inputs(batch, task, instr)
    z_tensor, _ = dec.build_inputs(f, task, instr)
    assert z_dict.shape == z_tensor.shape
    assert torch.allclose(z_dict, z_tensor, atol=1e-5)      # tensor path == encoding the batch


def test_predict_proba_accepts_tensor_record(schema, sample_df):
    enc, vocabs, dec = _decoder(schema, sample_df)
    f = enc.encode(vocabs.encode(sample_df.head(5)))
    p = dec.predict_proba(f, torch.zeros(5, dtype=torch.long),
                          torch.randint(0, 64, (5, 4)), [0, 1])
    assert p.shape == (5, 2)
    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-4)
