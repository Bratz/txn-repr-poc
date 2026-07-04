"""
eFRM channel source - the digital-banking (channel/device/session/audit) view of a payment,
modelled as an ADDITIONAL SOURCE beside the interbank ISO view (multi-source, like
data/iso_lifecycle): its own event stream + a per-transaction channel-context frame, fused at
the entity/embedding level. It is NEVER join-widened into the pacs row - the encoder schema,
serving bundle and existing tests are untouched.

Source of truth: the bank's eFRM request schema (eFRM.xlsx, ~84 attributes). EFRM_ATTRS below
records the DISPOSITION of every attribute group:
  feature    genuinely new signal for the TFM (channel/device/session/audit axes)
  duplicate  already carried by the ISO view (amount, ccy, parties, countries, ...)
  label      an OUTCOME (responseFlag/ErrCode) - never an intake feature (leakage)
  key        join/entity key (trace ids, userId, sessionId) - groups sequences, not content
  park       different domain (card) - out of scope until card rails are
  skip       extensibility placeholders with no semantics

Synthetic behaviour (documented choices):
  * each customer (UltmtDbtr_Id) owns 1-2 devices; a small fraction of logins use a NEW device
  * an ATO (account-takeover) episode = new device + credential change + payee-add shortly
    before an outward payment (the classic change-then-drain pattern) -> atoFlag=1
  * the ATO signal lives ONLY in this source - by construction the ISO row cannot see it.
    That is the point: run_efrm.py measures the fusion lift, i.e. what the second source buys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Disposition registry (exact attribute names from the eFRM sheet)
# --------------------------------------------------------------------------- #
EFRM_ATTRS = {
    # Channel
    "userId": "key", "groupId": "duplicate", "customerId": "duplicate",
    "channelId": "feature", "deviceOs": "feature", "deviceAgent": "feature",
    "deviceId": "feature", "deviceType": "feature", "ipAddress": "feature",
    "deviceCountryCode": "feature",
    # Transaction
    "txnCategory": "duplicate", "txnCode": "duplicate", "txnName": "duplicate",
    "txnPurpose": "feature", "txnInitatedTime": "feature", "custCategory": "feature",
    "txnRemarks": "skip", "txnAmount": "duplicate", "txnCurrency": "duplicate",
    "responseFlag": "label", "responseErrCode": "label", "responseErrDesc": "label",
    "transactionTraceId": "key", "transactionReference": "key",
    "transactionHostReference": "key",
    # Account
    "accountNumber": "duplicate", "accountBBAN": "duplicate", "accountName": "duplicate",
    "accountType": "feature", "accountStatus": "feature",
    # Payee
    "bankName": "duplicate", "bankCode": "duplicate", "payee.accountNumber": "duplicate",
    "payee.accountBBAN": "duplicate", "payee.accountName": "duplicate",
    "bankCountryCode": "duplicate", "type": "duplicate", "mode": "feature",
    "nickName": "feature", "mmID": "duplicate", "payeeCategory": "feature",
    "payeeCountry": "duplicate",
    # Audit (credential-change events - ATO precursors)
    "tPinChange": "feature", "mPinChange": "feature", "passwordChange": "feature",
    "mobileNumberChange": "feature", "emailIdChange": "feature", "alertChange": "feature",
    "securityQuestionsChange": "feature", "securityImageChange": "feature",
    "securityPhraseChange": "feature", "passwordChangedRecently": "feature",
    "mobileChangedRecently": "feature", "emailChangedRecently": "feature",
    # Card - separate instrument domain
    "cardType": "park", "cardNumber": "park", "pinChange": "park", "cardStatus": "park",
    # Session
    "sessionId": "key", "sessionStartTime": "feature", "lastLoginTimeStamp": "feature",
    "failedLogins_1hr": "feature",
    # Device integrity
    "deviceFingerprint": "feature", "appVersion": "feature",
    # deviceTrustLevel is ANOTHER MODEL'S OUTPUT - circularity risk; take only with lineage.
    "deviceTrustLevel": "park",
    "isEmulator": "feature", "isRootedOrJailbroken": "feature",
    # Extensibility placeholders
    **{f"stringField{i}": "skip" for i in range(1, 6)},
    **{f"DateField{i}": "skip" for i in range(1, 6)},
    **{f"binaryField{i}": "skip" for i in range(1, 4)},
    **{f"AmountField{i}": "skip" for i in range(1, 4)},
}

CHANNELS = ["IB", "MB"]
DEVICE_OS = {"IB": ["Windows", "macOS"], "MB": ["Android", "iOS"]}
EVENT_TYPES = ["login", "txn", "payee_add", "password_change", "mobile_change", "email_change"]


def build_channel_events(pay: pd.DataFrame, seed: int = 23, ato_frac: float = 0.02,
                         new_device_frac: float = 0.05, geo_noise_frac: float = 0.02):
    """Synthesise the eFRM view for OUTWARD payments of an India dataset.

    Returns (ctx, events):
      ctx     one row per outward payment_id - the channel context at transaction time
              (features per EFRM_ATTRS) + atoFlag (label; the drain payment of an episode)
      events  the raw channel event stream (login / payee_add / credential changes / txn),
              keyed by userId & deviceId, minutes-timestamped around the payment
    """
    rng = np.random.default_rng(seed)
    out = pay[pay["direction"] == "outward"].reset_index(drop=True)
    user_dev: dict = {}
    ctx_rows, evt_rows = [], []
    for r in out.itertuples():
        user = str(r.UltmtDbtr_Id)
        base = user_dev.setdefault(user, f"DEV-{abs(hash(user)) % 10**8:08d}")
        ato = rng.random() < ato_frac
        new_dev = ato or (rng.random() < new_device_frac)
        device = f"DEV-{rng.integers(10**7, 10**8):08d}" if new_dev else base
        channel = str(rng.choice(CHANNELS))
        geo_mismatch = ato or (rng.random() < geo_noise_frac)
        failed = int(rng.poisson(2.0)) + 1 if ato else int(rng.poisson(0.1))
        mins_since_login = float(rng.uniform(0.5, 4.0) if ato else rng.uniform(1.0, 45.0))
        cred_recent = ato or (rng.random() < 0.01)
        payee_recent = ato or (rng.random() < 0.03)
        t0 = 0.0                                              # txn time = episode anchor
        ctx_rows.append({
            "payment_id": r.payment_id, "userId": user, "deviceId": device,
            "channelId": channel, "deviceOs": str(rng.choice(DEVICE_OS[channel])),
            "deviceType": "Mobile" if channel == "MB" else "Desktop",
            "deviceCountryCode": ("XX" if geo_mismatch else str(r.Dbtr_Ctry)),
            "deviceIsNew": int(new_dev), "geoMismatch": int(geo_mismatch),
            "isEmulator": int(ato and rng.random() < 0.3),
            "failedLogins_1hr": failed, "minsSinceLogin": round(mins_since_login, 1),
            "credChangedRecently": int(cred_recent), "payeeAddedRecently": int(payee_recent),
            "accountStatus": "Dormant" if (ato and rng.random() < 0.2) else "Active",
            "atoFlag": int(ato),
        })
        evts = [("login", t0 - mins_since_login), ("txn", t0)]
        if payee_recent:
            evts.append(("payee_add", t0 - float(rng.uniform(5, 120))))
        if cred_recent:
            evts.append((str(rng.choice(["password_change", "mobile_change",
                                         "email_change"])), t0 - float(rng.uniform(10, 300))))
        for etype, t in evts:
            evt_rows.append({"payment_id": r.payment_id, "userId": user, "deviceId": device,
                             "sessionId": f"S-{r.payment_id}", "event_type": etype,
                             "t_offset_min": round(t, 1)})
    ctx = pd.DataFrame(ctx_rows)
    events = pd.DataFrame(evt_rows).sort_values(
        ["payment_id", "t_offset_min"]).reset_index(drop=True)
    return ctx, events


# numeric/binary context features the fusion runner consumes (linear-probe friendly).
CTX_FEATURES = ["deviceIsNew", "geoMismatch", "isEmulator", "failedLogins_1hr",
                "minsSinceLogin", "credChangedRecently", "payeeAddedRecently"]


def ctx_matrix(ctx: pd.DataFrame) -> np.ndarray:
    X = ctx[CTX_FEATURES].to_numpy(dtype=float)
    X[:, CTX_FEATURES.index("minsSinceLogin")] = np.log1p(X[:, CTX_FEATURES.index("minsSinceLogin")])
    return X
