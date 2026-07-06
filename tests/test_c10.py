"""C10 read-out plumbing: token outputs align with the encoder's row embedding."""

import numpy as np
import torch

from run_c10 import embed_all_tokens


def test_embed_all_tokens_first_token_is_fx():
    from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    from run_seq import embed_all_rows

    pay, _, accs = build_dataset(IndiaConfig(num_accounts=30, num_payments=200, seed=5))
    schema = build_schema(pay, accs)
    cfg = EncoderConfig(hidden=32, layers=1, heads=2, ff_mult=2, epochs=1)
    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(pay, schema, cfg, party_epochs=1)
    enc.freeze()
    full = vocabs.encode(pay)

    toks = embed_all_tokens(enc, full, len(pay), "cpu", batch_size=64)
    assert toks.shape[0] == len(pay) and toks.shape[2] == cfg.hidden
    assert toks.shape[1] >= 3                      # row token + several column tokens

    e = embed_all_rows(enc, full, len(pay), "cpu")
    assert np.allclose(toks[:, 0].numpy(), e.numpy(), atol=1e-5)   # [:,0] == f(x)

    # the wide read-outs really differ from the row token (bandwidth is additional)
    mean = toks[:, 1:].mean(1).numpy()
    assert not np.allclose(mean, e.numpy(), atol=1e-3)
