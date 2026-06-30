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

## Backlog (from review, not this PR)

- [ ] P1 strict=False decoder load swallows missing trio keys (`predict.py:253`).
- [ ] P1 held-out leak: assert `e_all` aligned to reset-index df (`history_encoder.pretrain`).
- [ ] P1 XML: parser source-dispatch on long string/bytes; writer drops UltmtDbtr & emits
  "12345.0" for numeric IDs; guarded `float(amt.text)`; explicit `_first_id` paths.
- [ ] P1 serve_india: no-eligible-rail fallback + missing-bundle/file errors.
- [ ] P1 quantizer: reject non-finite amounts.
- [ ] P1 (latent) decoder all-ones attention/target masks; dead prefix-pad logit code.
- [ ] P2 tests: intake_eval/run_inflight (twin+india), no-eligible-rail, legacy ckpt load,
  run_gpu tiny/single-class split; serve round-trip reuses same probes (tautology).
- [ ] P2 cdist zero-distance eps; freeze() block later .train(); CoLES B<2 no-op.

## Review (filled on completion)

Done in this PR: all 5 P0/P1 items above. 6 new tests (tests/test_review_hardening.py +
a freeze-eval test in test_multimodal_decoder.py). Full suite 158 green, no regressions.
Fidelity preserved: no paper hyperparameter changes, nothing unfrozen — the freeze
invariant is now *stronger* (frozen base stays eval through `.train()`). Backlog items
remain for a follow-up PR.
