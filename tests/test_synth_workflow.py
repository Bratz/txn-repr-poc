"""Tests for the payment-twin workflow-log generator (data/synth_workflow.py)."""

from data.synth_workflow import (
    EXCEPTION_CODES, TERMINAL_STATUS, WfConfig, build_schema, build_workflow_dataset,
)


def test_emits_payment_and_event_tables():
    pay, evt, accs = build_workflow_dataset(WfConfig(num_accounts=120, num_payments=1500, seed=1))
    assert len(pay) == 1500 and len(evt) > len(pay)
    for col in ("payment_id", "direction", "terminal_status", "time_to_settle_min",
                "DbtrAcct_Id", "IntrBkSttlmAmt"):
        assert col in pay.columns
    for c in EXCEPTION_CODES:
        assert f"exc_{c}" in pay.columns
    assert set(pay["terminal_status"]).issubset(set(TERMINAL_STATUS))
    assert set(pay["direction"]) == {"outward", "inward"}
    assert set(evt.columns) >= {"payment_id", "seq", "step", "outcome", "excode", "t_min"}


def test_repaired_payments_take_longer_than_stp():
    pay, _, _ = build_workflow_dataset(WfConfig(num_accounts=120, num_payments=2500, seed=2))
    m = pay.groupby("terminal_status")["time_to_settle_min"].mean()
    # a repair adds real delay, so REPAIRED should settle slower than clean STP
    assert m["REPAIRED"] > m["STP"]


def test_event_log_references_payments_and_orders():
    pay, evt, _ = build_workflow_dataset(WfConfig(num_accounts=80, num_payments=800, seed=3))
    assert set(evt["payment_id"]).issubset(set(pay["payment_id"]))
    one = evt[evt.payment_id == evt.payment_id.iloc[0]]
    assert list(one["seq"]) == sorted(one["seq"])              # chronological
    assert (one["t_min"].to_numpy()[1:] >= one["t_min"].to_numpy()[:-1]).all()


def test_schema_twin_block():
    pay, _, accs = build_workflow_dataset(WfConfig(num_accounts=80, num_payments=600, seed=4))
    s = build_schema(pay, accs)
    t = s["twin"]
    assert set(t["directions"]) == {"outward", "inward"}
    assert t["status_column"] == "terminal_status" and t["eta_column"] == "time_to_settle_min"
    assert len(t["exc_columns"]) == len(EXCEPTION_CODES)
