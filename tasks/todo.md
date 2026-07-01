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
