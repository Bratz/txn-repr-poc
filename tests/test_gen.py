"""C8 generative tier: band arithmetic, label plumbing, and rollout invariants."""

import numpy as np
import pandas as pd
import torch

from data.next_event import windowed_examples
from encoders.quantizer import AdaptiveQuantizer
from run_gen import (AMOUNT_GROUPS, BAND_DAYS, NEXT_BANDS, amount_group_of,
                     decode_amount, gap_band_of)


def test_gap_band_of():
    assert list(gap_band_of([0, 1, 3, 9, 20, 40])) == [0, 1, 2, 3, 4, 5]
    assert NEXT_BANDS[0] == "0d" and NEXT_BANDS[-1] == ">30d"
    assert len(BAND_DAYS) == len(NEXT_BANDS) == 6


def test_amount_group_decode_roundtrip():
    rng = np.random.default_rng(0)
    amts = np.exp(rng.uniform(3, 13, 500))
    ccys = np.array(["INR"] * 500, object)
    q = AdaptiveQuantizer().fit(amts, ccys)
    groups = amount_group_of(q, amts, ccys)
    assert groups.min() >= 0 and groups.max() < AMOUNT_GROUPS
    # decoding a group lands back in the same group for that currency
    for g in sorted(set(groups.tolist()))[:5]:
        x = decode_amount(q, g, "INR")
        assert amount_group_of(q, [x], ["INR"])[0] == g
    # unseen currency falls back to the global grid without raising
    assert decode_amount(q, 3, "XXX") > 0


def test_next_pos_labels():
    df = pd.DataFrame({
        "DbtrAcct_Id": ["A"] * 6,
        "CdtrAcct_Id": ["P1", "P2", "P1", "P1", "P2", "P1"],
        "IntrBkSttlmAmt": [10.0, 20, 30, 40, 50, 60],
        "IntrBkSttlmDt": pd.date_range("2023-01-01", periods=6, freq="10D").astype(str),
    })
    seqs, lab = windowed_examples(df, horizon_days=35, min_history=3, max_len=8)
    real = lab[lab["next_pos"] >= 0]
    # each labelled example's next_pos row carries exactly the labelled amount
    for _, r in real.iterrows():
        assert df["IntrBkSttlmAmt"].iloc[int(r["next_pos"])] == r["next_amount"]
    # censored tail negative (window extends past horizon? here it does not) -> if
    # present it must be -1
    assert (lab[lab["gap_days"].isna()]["next_pos"] == -1).all()


def test_rollout_invariants():
    from data.synth_india_rails import IndiaConfig, build_dataset, build_schema
    from encoder.tabular_encoder import EncoderConfig, build_pretraining_stack
    from encoder.tabular_encoder import pretrain as enc_pretrain
    from run_gen import example_fields, fit_field_heads, simulate
    from run_seq import embed_all_rows, encode_histories, small_history_encoder

    pay, _, accs = build_dataset(IndiaConfig(num_accounts=40, num_payments=500,
                                             seed=29, cadence_frac=0.6))
    schema = build_schema(pay, accs)
    cfg = EncoderConfig(hidden=64, layers=2, heads=2, ff_mult=2, epochs=1)
    torch.manual_seed(0)
    enc, _, vocabs = build_pretraining_stack(pay, schema, cfg, party_epochs=1)
    enc_pretrain(enc, vocabs.encode(pay), cfg, batch_size=128)
    enc.freeze()
    e_all = embed_all_rows(enc, vocabs.encode(pay), len(pay), "cpu")

    seqs, lab = windowed_examples(pay)
    q = AdaptiveQuantizer().fit(pay["IntrBkSttlmAmt"].to_numpy(), pay["Ccy"].to_numpy())
    recon = {"Ccy": vocabs.core_size("Ccy")}
    full = vocabs.encode(pay)
    hist, _ = small_history_encoder(e_all, recon, {"Ccy": full["core"]["Ccy"]},
                                    seqs, "cpu", epochs=1)
    H = encode_histories(hist, e_all, seqs, "cpu").cpu().numpy()
    m, y, naive, _ = example_fields(pay, seqs, lab, q)
    heads = fit_field_heads(H[m], y)

    actor = lab["actor"].iloc[0]
    roll = simulate(pay[pay["DbtrAcct_Id"] == actor], enc, vocabs, hist, heads, q,
                    steps=4, device="cpu")
    assert len(roll) == 4
    dates = pd.to_datetime(roll["IntrBkSttlmDt"])
    assert dates.is_monotonic_increasing                      # never goes backwards
    # strictly increasing whenever the predicted band is not same-day
    d = dates.diff().dropna().dt.days.to_numpy()
    bands = roll["gap_band"].tolist()[1:]
    for gap, b in zip(d, bands):
        if b != "0d":
            assert gap > 0
    # amounts decode onto the actor currency's grid (or the global fallback)
    ccy = str(pay[pay["DbtrAcct_Id"] == actor]["Ccy"].iloc[-1])
    grid = q.grids_.get(ccy, q.grids_["__GLOBAL__"])
    for a in roll["IntrBkSttlmAmt"]:
        assert np.isclose(grid, a).any()
    assert {"rail_pred", "conf_gap", "conf_amount", "conf_rail"} <= set(roll.columns)
