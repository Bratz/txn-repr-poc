# TODO — PRAGMA-alignment: close the input-model gaps (2026-07-06)

Close the four gaps between the twin's hybrid input (pacs.008 spine + pain.001 enrichment +
lifecycle stream) and PRAGMA's input contract. ALL ITEMS ADDITIVE — dismantling analysis at
the bottom. Order = predicted payoff per effort.

- [x] **C11 — entity-level fusion (the big one).** DONE except serving arg (deferred to
  post-verdict, measure-first — stated amendment); verdict = runbook K. Score the CURRENT payment conditioned on
  the ACCOUNT's full history (all prior payments' lifecycle events + profile), PRAGMA-style.
  1. Generator first: `IndiaConfig.recall_momentum=0.0` (default OFF) — an account with past
     recalls/returns gets an elevated hazard for future ones (the signal must exist to find;
     `acct_heat`/`exception_momentum` is the pattern to extend). New seeded corpora only when
     the knob is ON; every existing number stays reproducible.
  2. Entity timeline builder: per-account time-ordered stream of ALL its messages
     (visibility-filtered) across payments — reuse `message_prefix_sequences` + history
     encoder; h_entity via `encode_histories`.
  3. Fusion: in-flight heads read [prefix features ⊕ h_entity] (the origination-fusion
     pattern from run_impute, applied at the in-flight grain).
  4. Pre-register: with momentum ON, return/cancel PR-AUC (entity-fused vs prefix-only)
     >= +5pp on held-out accounts; with momentum OFF, expect ~0 (the honest null control).
  5. Serving: `predict_stream(msg_df, account_history=None)` — OPTIONAL new argument;
     payment-only path unchanged (stateless contract preserved).
- [x] **C12 — life-long events (cheapest).** DONE; smoke null as pre-stated; verdict =
  runbook L (default + momentum-ON readings). Timed milestones on the [USR]/profile side:
  first_payment / first_crossborder / first_recall / account_age, encoded as
  log-time-since-milestone at the evaluation point (PRAGMA's life-long items). Measure on
  the DOCUMENTED weak spot: cold-start / young-account rail+risk accuracy (accounts with
  < N events), with vs without tenure features. Threshold: any significant lift on the
  young-account slice without degrading the full slice.
- [x] **C13 — sub-day time resolution (surgical).** DONE — folded into C11's entity
  timeline (fractional-day dt from t_offset_min; test pins the 90-min case). Payment-date
  sequences untouched as planned; msgseq was already minute-granular. `seq_from_dates` casts to datetime64[D];
  message streams carry `t_offset_min` that the history encoder never sees. Add fractional-
  day dt ONLY where minute data exists (message-timeline sequences: msgseq, in-flight, C11
  entity stream) via an OPT-IN parameter. Payment-date sequences (C3/C4/velocity/C7/C8)
  UNTOUCHED — their measured numbers were earned with day-granular dt and stay comparable.
- [ ] **Key–value tokenizer — deferred, trigger-based.** Not scheduled. Trigger: a measured
  ceiling attributable to schema-forcing (e.g. C11's entity stream underperforming with
  null-padded message rows). Until a number demands it, the fixed-schema spine carries C1.

## Dismantling analysis (explicit)

- **Nothing is removed.** C11 = new fusion input + optional API argument; C12 = new profile
  features; C13 = opt-in resolution; tokenizer = deferred.
- **Two near-misses guarded:** (1) generator momentum knobs DEFAULT OFF so all seeded corpora
  and measured claims (C1–C10) remain byte-reproducible; (2) C13 is opt-in per sequence
  family so no previously measured dt-dependent number silently shifts.
- **Serving contracts:** predict_stream stays valid without account history (stateless path
  preserved); /score/ask, velocity, forecast, liquidity untouched.
- **Pending GPU verdicts (runs F/G/H/I) unaffected** — none of these items touch run_india
  --paper-serving, run_c7, run_gen, or run_c9 paths.

---

# TODO — ISO 20022 lifecycle + predict-at-initiation (2026-07-01)

Model the training data as the ISO message lifecycle (multi-source) and add lifecycle
predictions; drop UPI for now.

- [x] **Lifecycle model.** `data/iso_lifecycle.py` — 5 message types (pain.001/pain.002/
  pacs.008/pacs.002/camt.054) with real field ownership, TxSts + ISO reason codes, emission
  driven by where the workflow halted. Message timestamps (`t_offset_min`). Replaces
  `enrichment_stages.py`. `data/synth_india_rails.build_messages` writes a 3rd table.
- [x] **Drop UPI (reversible).** `IndiaConfig.domestic_rails=("RTGS","NEFT","IMPS")` + `allow`
  filter in `rails.eligible_rails/choose_rail`. Registry untouched; opt back in per run.
  KNOWN: limit_exceeded prevalence falls to ~0.1% (IMPS 5L is the only cap now).
- [x] **A/B/C predict-at-initiation.** `run_impute.py` — STP outcome / reject reason / ETA
  from the pain.001 view vs complete pacs.008, + imputation. `reject_reason` label added.
- [x] **Mix pain.001 into pretraining.** `run_seq.frozen_embeddings(extra=)` opt-in.
- [x] **D streaming outcome.** booked-prediction PR-AUC rises as the message prefix grows
  (mean-pooled per-message embeddings; sequence encoder deferred — prefixes are short).
- [x] Tests: `tests/test_iso_lifecycle.py`; UPI-path tests opt in explicitly. 173 pass.
- [ ] Full (non-smoke) run for real A/B/C/D numbers (smoke encoder is near-chance).
- [ ] OPTIONAL: restore limit_exceeded strength if the twin binary task must stay learnable.

# TODO — post-review hardening

Plan of record for the senior code-review follow-up (2026-06-30). Findings came from a
4-way parallel review (core repr / decoder+v2 / data+Layer-1 / orchestrators+serving+tests).
These are robustness/integrity fixes — they HARDEN the experiment, they do not change paper
hyperparameters or unfreeze anything (fidelity preserved).

## This PR (P0s + top P1s)

- [x] **P0-1 Party encoder UNK fallback.** unseen attr → per-field [MASK] row (graceful UNK),
  no NaN/OOB. `encoders/party_encoder.py`.
- [x] **P0-2 Rail-routing metric integrity.** `is_mis_routed` added to the generator; routing
  now reports **per-class**, **domestic-only**, and **clean (mis-routed-excluded)** accuracy.
  `data/synth_india_rails.py`, `run_india.py`.
- [x] **P1-3 Shared categorical canonicaliser.** `encoders/coerce.canon_categorical` used at
  build AND encode in `column_assembler` + `party_encoder`; `ColumnVocabs.unk_rate()` diagnostic.
- [x] **P1-4 Freeze = deterministic.** `_freeze_base` calls `llm.eval()`; `MultimodalDecoder.train()`
  overridden to keep encoder+LLM in eval; `predict_proba` / `history_encoder.encode` save/restore mode.
- [x] **P1-5 Seed numpy.** `run_india/run_twin/run_seq/run_golden` seed `np.random` in `main()`.

## Backlog (from review)

- [x] P1 strict=False decoder load — now checks unexpected/missing trio keys (`predict.py`).
- [x] P1 XML: type-dispatched source (string/bytes/path); symmetric UltmtDbtr + `_idstr`
  canonical ids (no "12345.0"); guarded `float(amt.text)`; explicit `_first_id` priority paths.
- [x] P1 serve_india: no-eligible-rail fallback (drop instrument constraint) + clean
  missing-bundle/input errors.
- [x] P2 freeze() blocking later `.train()` — done via `MultimodalDecoder.train()` override.
- [x] P2 serve round-trip reuses same probes (tautology) — fixed to compare a 2nd loaded scorer.
- [x] P1 held-out leak: `history_encoder.pretrain` now asserts `e_all`/targets aligned.
- [x] P1 quantizer: rejects non-finite amounts (fit + transform).
- [x] P1 (latent) decoder: documented the all-rows-equal-length (no-pad) mask invariant;
  replaced the dead prefix-pad branch with `assert logits.shape[1] == z.shape[1]`.
- [x] P2 cdist zero-distance: triplet loss uses stable squared-distance + eps sqrt.
- [x] P2 CoLES B<2: composite_loss skips the triplet term for singleton batches.
- [x] P2 tests: twin intake_eval covered (+ quantizer / triplet / alignment-assert tests).
- [ ] P2 (minor, deferred): run_india run_inflight direct test; legacy single-task ckpt load;
  run_gpu tiny/single-class split path. Low-risk, exercised indirectly.

## Review (filled on completion)

Done in this PR: all 5 P0/P1 items above. 6 new tests (tests/test_review_hardening.py +
a freeze-eval test in test_multimodal_decoder.py). Full suite 158 green, no regressions.
Fidelity preserved: no paper hyperparameter changes, nothing unfrozen — the freeze
invariant is now *stronger* (frozen base stays eval through `.train()`). Backlog items
remain for a follow-up PR.
