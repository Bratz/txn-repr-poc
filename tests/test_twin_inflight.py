"""The payment-twin in-flight model: step encoder + reused history-encoder backbone."""

import numpy as np
import torch

from encoder.history_encoder import HistoryConfig
from run_twin import Inflight, StepEmbedder, collate_steps


def test_step_embedder_shapes():
    se = StepEmbedder(n_steps=7, n_exc=10, n_dir=2, d=16)
    step = torch.randint(0, 7, (4, 5))
    exc = torch.randint(0, 10, (4, 5))
    direction = torch.randint(0, 2, (4,))
    out = se(step, exc, direction)
    assert out.shape == (4, 5, 16)


def test_collate_pads_and_builds_time():
    batch = [
        {"step": np.array([0, 1, 2]), "exc": np.array([0, 0, 3]),
         "t": np.array([1.0, 3.0, 8.0], np.float32), "direction": 0, "target": 3},
        {"step": np.array([0]), "exc": np.array([0]),
         "t": np.array([1.0], np.float32), "direction": 1, "target": 0},
    ]
    b = collate_steps(batch, "cpu")
    assert b["step"].shape == (2, 3) and b["pad_mask"][1].tolist() == [False, True, True]
    assert (b["dt"] >= 0).all() and b["dt"][0, 0] == 0          # first gap is 0
    assert b["target"].tolist() == [3, 0]


def test_inflight_forward_predicts_next_class():
    hcfg = HistoryConfig(hidden=16, layers=2, heads=2, ff_mult=2)
    model = Inflight(n_steps=7, n_exc=10, n_dir=2, n_next=10, hcfg=hcfg)
    batch = [
        {"step": np.array([0, 1, 2]), "exc": np.array([0, 0, 0]),
         "t": np.array([1.0, 3.0, 8.0], np.float32), "direction": 0, "target": 5},
        {"step": np.array([0, 1]), "exc": np.array([0, 2]),
         "t": np.array([1.0, 4.0], np.float32), "direction": 1, "target": 0},
    ]
    logits = model(collate_steps(batch, "cpu"))
    assert logits.shape == (2, 10)
    # trains a step: loss is finite and backprops
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([5, 0]))
    loss.backward()
    assert model.head.weight.grad is not None and model.steps.step.weight.grad is not None
