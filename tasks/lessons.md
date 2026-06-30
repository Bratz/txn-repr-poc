# Lessons

Recurring patterns to prevent repeat mistakes (per CLAUDE.md self-improvement loop).
Reviewed at session start.

## Tooling / workflow

- **Bash tool is POSIX sh, not PowerShell.** `@'…'` / `@"…"@` here-strings are PowerShell
  only; in the Bash tool they leave a stray `@` as the commit subject. Use a real heredoc:
  `git commit -F - <<'EOF' … EOF`.
- **Background runs: use `python -u`.** Buffered stdout is lost if the process is killed at a
  session boundary; unbuffered keeps partial progress in the log file.

## Data / modelling correctness

- **Label leakage hides in "consequence" columns.** A feature that is a deterministic
  function of the label (here: `SttlmMtd`, and later caught in review: `Ccy`, counterparty
  country, near-deterministic `identifier_type`) silently inflates a reported metric. Before
  reporting any classification accuracy, ask "what in the feature set recovers the label?" and
  report **per-class** / on the non-trivial slice.
- **A label must match the task.** Over-cap/below-min *injected* rows store the *attempted*
  (ineligible) rail; that's correct for exception generation but corrupts the routing label.
  Keep generation-intent and task-label separate (`is_mis_routed`).
- **Categorical encoding is dtype-sensitive.** `.astype(str)` makes `123` (int) and `123.0`
  (float) different keys → silent UNK at serve time. Always canonicalise categoricals through
  ONE helper used at both vocab-build and encode time; reserve a UNK slot everywhere (the
  party encoder had none); log serve-time UNK rate.
- **Freeze ≠ deterministic.** Killing gradients does not stop dropout. Frozen submodules
  (encoder, LLM) must also be `.eval()`; `predict`/`encode` helpers that flip a module to eval
  must save and restore the prior `training` mode, or they silently disable dropout for the
  rest of a training loop.
- **Seed every RNG you use.** Seeding torch + the split RNG isn't enough if training also
  calls the *global* numpy RNG (`np.random.permutation`). Unseeded => non-reproducible
  results.json, which voids a falsifiable-claim POC.

## Verification

- **Reproduce the failure before asserting it.** The int↔float UNK hazard was confirmed with
  a 3-line repro before recommending a fix — cheaper than a wrong assertion.
- **A round-trip test that reuses the same object is a tautology.** Reload a *second*,
  independent instance to actually exercise serialization.
