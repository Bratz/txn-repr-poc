"""Unit tests for the v2 Layer-3b history encoder (encoder/history_encoder.py)."""

import torch

from encoder.history_encoder import HistoryConfig, HistoryEncoder

RECON = {"amount_level": 8, "Ccy": 5, "SttlmMtd": 4}


def _batch(B, L, D):
    e_seq = torch.randn(B, L, D)
    pad = torch.zeros(B, L, dtype=torch.bool)
    pad[0, -1] = True                                   # one padded tail position
    batch = {
        "dt": torch.rand(B, L) * 10,
        "dow": torch.randint(0, 7, (B, L)),
        "dom": torch.randint(0, 31, (B, L)),
        "month": torch.randint(0, 12, (B, L)),
        "pad_mask": pad,
    }
    targets = {k: torch.randint(0, v, (B, L)) for k, v in RECON.items()}
    return e_seq, batch, targets


def _enc(D=16):
    return HistoryEncoder(RECON, HistoryConfig(hidden=D, layers=2, heads=2, ff_mult=2))


def test_forward_shapes():
    D = 16
    hist = _enc(D)
    e_seq, batch, _ = _batch(3, 5, D)
    h_usr, h_events = hist.forward(e_seq, batch)
    assert h_usr.shape == (3, D)
    assert h_events.shape == (3, 5, D)


def test_event_mask_excludes_pad_and_is_nonempty():
    hist = _enc()
    _, batch, _ = _batch(4, 6, 16)
    m = hist.sample_event_mask(batch["pad_mask"])
    assert not (m & batch["pad_mask"]).any()            # never masks a pad slot
    valid_rows = (~batch["pad_mask"]).any(dim=1)
    assert (m.any(dim=1) | ~valid_rows).all()           # >= 1 masked per non-empty row


def test_composite_loss_backprops():
    hist = _enc()
    e_seq, batch, targets = _batch(4, 6, 16)
    loss, parts = hist.composite_loss(e_seq, batch, targets)
    assert loss.requires_grad and float(loss) > 0
    assert set(parts) == {"mask", "triplet"}
    loss.backward()
    assert hist.usr_token.grad is not None


def test_encode_is_detached_and_freeze():
    hist = _enc()
    e_seq, batch, _ = _batch(2, 4, 16)
    h = hist.encode(e_seq, batch)
    assert h.shape == (2, 16) and not h.requires_grad
    hist.freeze()
    assert all(not p.requires_grad for p in hist.parameters())
