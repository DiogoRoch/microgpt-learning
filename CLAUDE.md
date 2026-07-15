# CLAUDE.md — microgpt-learning

Interactive learning experience for Karpathy's microgpt (see PLAN.md for the full plan).
`reference/microgpt.py` (fetched, gitignored) is the source of truth and outranks
everything written here or in PLAN.md.

## Commands

```bash
bash tools/fetch_reference.sh   # fetch reference/microgpt.py (pinned gist revision)
python3 tools/golden.py         # regenerate golden/ dumps (full 1000-step run; ~minutes)
                                #   GOLDEN_MAX_STEPS=3 for a fast partial run while iterating
                                #   (never commit partial dumps)
npm run dev                     # Vite dev server
npm run typecheck               # tsc --noEmit
npm test                        # Vitest: engine + parity suite (loads golden/)
npm run golden                  # alias: fetch reference + regenerate golden dumps
npm run train                   # tools/train_checkpoint.ts → src/data run artifacts
npm run build                   # production build (static)
npm run lint                    # eslint
```

## Working agreements (from the build brief — binding)

- **Engine changes are only done with parity tests green.** If a change to
  `src/engine/*` breaks parity, fix the engine — never loosen a tolerance to make a
  test pass without a written justification in NOTES.md.
- The fetched file outranks the brief; the brief outranks defaults. Consequential
  ambiguity (scope, pedagogy, visual identity) → ask the user; local implementation
  details → decide and note in NOTES.md.
- **Never fake a number.** Every value on screen comes from the in-repo engine or the
  shipped precomputed run. Never hardcode, never approximate silently.
- Never teach a variant the file doesn't implement (no GeLU, no layernorm, no biases,
  no explicit causal mask) except inside clearly-marked "in the wild" asides.
- UI uses the file's own variable names (`tok_emb`, `attn_logits`, `x_residual`,
  `m_hat`, …) — one shared vocabulary between app and file.
- Engine stays strictly UI-free. Small components. Conventional commits per milestone.
- Keep CLAUDE.md and PLAN.md current as decisions evolve; NOTES.md is the running
  design/decision journal.
- Attribute Karpathy prominently (app footer + README), linking the gist.
- If `tools/fetch_reference.sh`'s pinned revision is bumped: regenerate golden/, run the
  full parity suite, and re-read the diff of the reference file before anything else.

## Gotchas discovered so far

- This build environment's egress proxy blocks `gist.githubusercontent.com` but allows
  `gist.github.com` (git clone) and `raw.githubusercontent.com` — hence the fallback in
  the fetch script.
- `golden.py` instrumentation must never call `random.*`: the RNG stream must stay
  byte-identical to the pristine reference (the 20 sampled names in `final.json` prove
  it).
- Python `sum()` over `Value`s wraps the int 0 via `__radd__`; softmax's max-subtraction
  uses raw `.data` (no gradient through max). Mirror exactly — see PLAN.md "Engine
  semantics".
