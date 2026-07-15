# NOTES — design & decision journal

Newest entries first within each phase.

## Phase 0 — Foundations

- **Interview fallback.** The interactive question UI aborted twice in the build
  session; user said to continue. Adopted the brief's defaults (GitHub Pages, English,
  live training headline, asides on, no trims) — recorded in PLAN.md as revisable.
- **Fetch route.** Egress policy here blocks `gist.githubusercontent.com` (CONNECT 403)
  but allows `gist.github.com` git clone and `raw.githubusercontent.com`. Fetch script
  tries raw first (works elsewhere), falls back to clone at the pinned revision
  `14fb038816c7aae0bb9342c2dbf1a51dd134a5ff`.
- **Close-read findings that shape the engines** (also in PLAN.md): softmax max-subtract
  is a raw-float constant (no grad through max); `-x` is `x * -1`; `a/b` with Value `b`
  is `a * b**-1` (pow node) but `/number` is a constant multiply; `sum()` starts at
  int 0 (wrapped constant); grads zeroed inside the Adam loop; `n = min(block_size,
  len(tokens)-1)` — a 15-char name yields exactly n=16.
- **Golden doc subset.** 1000 training steps use `docs[step % 32033]` = the first 1000
  shuffled docs; `docs.json` commits the first 1024 plus the total count. The full
  names list ships separately in `src/data/` for chapter 1 (built from input.txt).
