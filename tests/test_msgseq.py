"""Tests for the message-lifecycle sequence helpers (run_msgseq.py)."""

import pandas as pd

from run_msgseq import TRACKER_CLASS, message_prefix_sequences, tracker_labels


def _msg():
    rows = [
        # E0: happy chain -> settled (ACSC)
        {"end_to_end_id": "E0", "seq": 0, "msg_type": "pain.001", "t_offset_min": 0.0},
        {"end_to_end_id": "E0", "seq": 1, "msg_type": "pain.002", "t_offset_min": 1.0},
        {"end_to_end_id": "E0", "seq": 2, "msg_type": "pacs.008", "t_offset_min": 2.0},
        {"end_to_end_id": "E0", "seq": 3, "msg_type": "pacs.009", "t_offset_min": 2.5},
        {"end_to_end_id": "E0", "seq": 4, "msg_type": "pacs.002", "t_offset_min": 3.0, "tx_sts": "ACSC"},
        {"end_to_end_id": "E0", "seq": 5, "msg_type": "camt.054", "t_offset_min": 4.0},
        # E1: held (ACSP/G002), no cover
        {"end_to_end_id": "E1", "seq": 0, "msg_type": "pain.001", "t_offset_min": 0.0},
        {"end_to_end_id": "E1", "seq": 1, "msg_type": "pain.002", "t_offset_min": 1.0},
        {"end_to_end_id": "E1", "seq": 2, "msg_type": "pacs.008", "t_offset_min": 2.0},
        {"end_to_end_id": "E1", "seq": 3, "msg_type": "pacs.002", "t_offset_min": 3.0, "tx_sts": "ACSP"},
        # E2: pre-submission reject -> no pacs.008, no pacs.002 -> dropped (no label, too short)
        {"end_to_end_id": "E2", "seq": 0, "msg_type": "pain.001", "t_offset_min": 0.0},
        {"end_to_end_id": "E2", "seq": 1, "msg_type": "pain.002", "t_offset_min": 1.0, "tx_sts": "RJCT"},
    ]
    return pd.DataFrame(rows)


def test_prefix_excludes_outcome_messages_and_keeps_order():
    msg = _msg()
    by = {s["actor"]: s for s in message_prefix_sequences(msg, min_len=2)}
    assert [msg["msg_type"].iloc[p] for p in by["E0"]["pos"]] == ["pain.001", "pacs.008", "pacs.009"]
    assert [msg["msg_type"].iloc[p] for p in by["E1"]["pos"]] == ["pain.001", "pacs.008"]
    assert "E2" not in by                                       # only pain.001 kept -> < min_len
    assert by["E0"]["dt"][0] == 0.0                             # first gap is zero


def test_tracker_labels_from_pacs002_status():
    msg = _msg()
    seqs = message_prefix_sequences(msg, min_len=2)
    kept, y = tracker_labels(msg, seqs)
    lab = {k["actor"]: int(c) for k, c in zip(kept, y)}
    assert lab == {"E0": TRACKER_CLASS["ACSC"], "E1": TRACKER_CLASS["ACSP"]}
