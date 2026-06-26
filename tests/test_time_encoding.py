"""Unit tests for the v2 event-time encoding (encoders/time_encoding.py)."""

import torch

from encoders.time_encoding import TimeEncoding


def test_shape_and_grad():
    te = TimeEncoding(d=32)
    B, L = 4, 6
    dt = torch.rand(B, L) * 30
    dow = torch.randint(0, 7, (B, L))
    dom = torch.randint(0, 31, (B, L))
    month = torch.randint(0, 12, (B, L))
    out = te(dt, dow, dom, month)
    assert out.shape == (B, L, 32)
    out.sum().backward()
    assert te.dow.weight.grad is not None


def test_inter_arrival_changes_output():
    # different gaps must produce different time codes (same calendar slot)
    te = TimeEncoding(d=16)
    z = torch.zeros(1, 2, dtype=torch.long)
    a = te(torch.tensor([[1.0, 1.0]]), z, z, z)
    b = te(torch.tensor([[1.0, 100.0]]), z, z, z)
    assert torch.allclose(a[0, 0], b[0, 0])          # gap 1 == gap 1
    assert not torch.allclose(a[0, 1], b[0, 1])      # gap 1 != gap 100


def test_negative_gap_clamped():
    te = TimeEncoding(d=8)
    z = torch.zeros(1, 1, dtype=torch.long)
    neg = te(torch.tensor([[-5.0]]), z, z, z)
    zero = te(torch.tensor([[0.0]]), z, z, z)
    assert torch.allclose(neg, zero)                 # log1p(clamp(<0)) == log1p(0)
