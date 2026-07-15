# PLAN — MicroGPT, Explained Interactively

An interactive, visually excellent learning experience teaching the transformer
architecture through [Andrej Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
— a ~200-line, pure-Python, dependency-free GPT.

**Success criterion (the "emma test"):** a motivated beginner who plays through the whole
experience genuinely understands every line of `microgpt.py` — not vague intuition, the
actual mechanics of this file.

Source of truth: `reference/microgpt.py`, pinned at gist revision `14fb0388` (200 lines,
2026-02-16). The file outranks the brief; the brief outranks defaults.

## Interview answers / assumptions

The interactive question channel was unavailable in the build session (two aborted
attempts); the user said to continue, so the brief's stated defaults were adopted.
**Every one of these is revisable — say the word and it changes:**

1. **Deployment: GitHub Pages.** Vite `base` read from an env var (`VITE_BASE`) so it
   also runs at root locally and on any static host; a GitHub Actions workflow deploys on
   push to the default branch.
2. **Language: English** for all copy.
3. **Live in-browser training: headline feature** (chapter 9 trains the real model in a
   Web Worker; the precomputed run ships too, for instant time-travel).
4. **"In the wild" asides: yes**, as clearly-marked, collapsed-by-default panels. The
   chapter-5 mask-equivalence toggle is core content, not an aside (brief mandates it).
5. **Scope trims: none.** Full 12-chapter v1 with quizzes and playground.
6. **Original-artifact must-haves: none specified**; the brief is treated as complete.

## Architecture

As specified in the brief §5/§6, adopted without pushback except where noted:

- Vite + React 18 + TypeScript strict; React Router with lazy chapter routes.
- Tailwind (v4, CSS-first config) + CSS custom-property design tokens.
- Hand-rolled SVG/canvas viz; d3 only for scales/shape helpers.
- Zustand for cross-chapter state (checkpoint, trained/untrained toggle, running example,
  chapter progress for the minimap).
- KaTeX for math asides. Code panel uses a hand-rolled, test-guarded Python
  tokenizer instead of Shiki (decision + rationale in NOTES.md: one known 200-line
  file, zero bundle cost, full control of line anatomy for sync/minimap).
- Vitest for engine + parity tests. Web Worker for training/batch inference.
- No backend; static build output.

```
reference/microgpt.py        # fetched, gitignored — the source of truth
tools/fetch_reference.sh     # pinned-revision fetch (raw URL, git-clone fallback)
tools/golden.py              # instrumented copy of the reference → golden/*.json
tools/train_checkpoint.ts    # Node: trains the TS engine → src/data artifacts
golden/                      # committed ground-truth dumps (see "Golden parity")
src/engine/                  # UI-agnostic, fully typed, parity-tested
  graph.ts                   #   Value-equivalent scalar autograd (exact semantics)
  tensor.ts                  #   fast Float32Array tape-autograd twin
  model.ts                   #   config, state dict (Python key names), incremental
                             #   forward w/ KV cache, doc loss, Adam, sampling
  rng.ts                     #   seeded PRNG + gaussian (for reproducible browser runs)
  trace.ts                   #   record named intermediates per (step,pos,layer,head)
src/viz/                     # reusable primitives (brief §8)
src/chapters/                # ch00-map … ch11-playground
src/data/                    # names list, precomputed run (base64-f32 snapshots)
src/worker/                  # training/inference worker + typed message protocol
```

### Engine semantics that must mirror the file exactly

Established by close reading; parity tests pin each one:

- `softmax` subtracts `max_val` as a **raw float** (`val.data`), so no gradient flows
  through the max — it is a constant shift, not a graph op.
- `a - b` desugars to `a + (b * -1)`; `x / v` (Value denominator) desugars to
  `x * v**-1` (a pow node); `x / 2` (number denominator) is a constant multiply.
- `sum()` over Values starts at int 0 → wrapped as a constant node via `__radd__`.
- `backward()` = DFS post-order topo sort, `self.grad = 1`, then reversed-topo
  `child.grad += local_grad * v.grad`. Local grads are captured **at construction**.
  (TS uses an iterative DFS — same order, no recursion limit.)
- rmsnorm: `(ms + 1e-5) ** -0.5`, no learnable scale; applied after the embedding sum,
  and pre-norm inside the block; **no final norm** before `lm_head`.
- Attention is incremental (KV-cache form): causality is implicit because future keys
  don't exist yet. No mask anywhere. `attn_logits` scaled by `1/head_dim**0.5`.
- Per-doc loss: `[BOS] + chars + [BOS]`, `n = min(block_size, len(tokens)-1)`,
  mean of per-position `-log(probs[target])`.
- Adam: `lr=0.01, β1=0.85, β2=0.99, eps=1e-8`, bias correction with `step+1`, linear
  decay `lr_t = lr·(1 − step/1000)`, one doc per step (`docs[step % len(docs)]`),
  `p.grad = 0` inside the update loop.
- Inference: `softmax([l / temperature ...])` with temperature 0.5, weighted sampling,
  stop on BOS, ≤ block_size tokens, 20 samples.

## Golden parity (brief §7)

`tools/golden.py` is a line-marked instrumented copy of the reference. Instrumentation
never calls `random.*`, so the RNG stream is byte-identical to the pristine file — the
final 20 sampled names must match a pristine run exactly (encoded as a meta-test).

Dumps (all JSON, committed):
- `meta.json` — gist revision, doc counts, timing, python version.
- `tokenizer.json` — `uchars`, `vocab_size`, BOS id.
- `docs.json` — shuffled-order head (first 1024 docs; training uses only the first 1000
  since 1000 steps < 32,033 docs), plus total count.
- `init_weights.json` — full f64 state_dict at init.
- `step0_trace.json` — every named intermediate for step 0, all positions:
  `tok_emb, pos_emb, x_emb_sum, x_emb_norm`, per-layer `x_ln_attn, q, k, v`, per-head
  `attn_logits, attn_weights, head_out`, `x_attn, x_wo, x_after_attn, x_ln_mlp, fc1,
  relu, fc2, x_after_mlp`, `logits, probs, loss_t`.
- `step{0,1,2}_state.json` — tokens, loss, full grad vector after `backward()`, and
  `m, v, params` after the Adam update.
- `losses.json` — all 1000 per-step losses (f64).
- `forced_decode_{init,step3,final}.json` — fixed token sequence in → per-position
  logits out, at three weight states (sidesteps RNG parity for inference).
- `final.json` — final-step loss, final params, the 20 sampled names.

Tolerances: scalar TS engine vs golden ≤ 1e-9 (both are f64; only libm ulp differences).
f32 tensor engine ≤ 1e-4 relative on forward/grads (f32 has ~7 significant digits;
intermediates here are O(1) so 1e-4 leaves headroom — justified in NOTES.md with
measured values). Full-training final loss compared loosely (measured drift documented).

Sanity checks (tests + surfaced in the app): initial loss ≈ ln 27 ≈ 3.2958; 4,192
params; pos-0 attention weights exactly `[1.0]`.

## Design direction

**Concept: "the annotated file."** The subject is a single Python file you can hold in
your head; the design makes that literal. A calm, paper-like reading canvas carries the
narrative, and the file itself is always present as a dark, monospaced artifact — the
code panel and the signature element:

- **Signature element — the file as navigation.** A minimap of all 200 lines of
  `microgpt.py` runs down the screen edge: each chapter's lines fill in as you complete
  it, hovering shows the code, clicking jumps to the chapter that teaches those lines.
  Finishing the course = watching the entire file light up. "You now understand this
  whole file" becomes a visible, literal state.
- **Palette tokens** (light canvas, dark code — deliberately not cream/terracotta, not
  black/acid-green):
  - `--paper  #FAF9F7` warm near-white canvas
  - `--ink    #16181D` text; also the code panel / minimap background
  - `--pos    #E8590C` vermilion — positive values in all data viz
  - `--neg    #1971C2` cerulean — negative values, and interactive affordances
  - `--hot    #FFC94D` amber — the "current" thing: active code line, current token,
    scrubber position
  - `--muted  #6B7280` secondary text/grid
  The signed pair (vermilion/cerulean) is colorblind-safe and used *only* for signed
  data, so numbers are readable at a glance everywhere (VectorChips, heatmaps, grads).
- **Type pairing:** IBM Plex Serif (display/headings) + IBM Plex Sans (body/UI) +
  IBM Plex Mono (code, all numbers). One superfamily designed to harmonize — apt for a
  project about reading code as literature — self-hosted via @fontsource (no CDN).
- **Motion** animates data transformations only (a vector flowing through rmsnorm,
  probability mass sharpening as temperature drops). `prefers-reduced-motion` swaps
  animation for instant state + step buttons. Keyboard operable, visible focus.

## Chapters

Brief §10 adopted as written: 0 big picture · 1 data/tokenizer · 2 autograd ·
3 parameters · 4 embeddings · 5 attention (centerpiece, incl. mask-equivalence morph) ·
6 MLP/residual stream · 7 loss · 8 backward/Adam · 9 training (live worker + scrubbable
precomputed run) · 10 inference/temperature · 11 playground. Every chapter:
hook → interactive core → CodePanel with the exact lines → recap + 2–3 PredictReveal
questions. Running example threaded throughout: **emma**.

## Phases

Per brief §12. Each phase ends with `npm run typecheck && npm test && npm run build`
green, dev-server exercise of the new work, and a conventional commit.

- **Phase 0 — Foundations** ✅ done — golden dumps committed, tokenizer parity green.
- **Phase 1 — Engines** ✅ done — full parity suite green, cross-engine agreement,
  0.33 s measured training time.
- **Phase 2 — Shell** ✅ done — navigable skeleton, minimap, CodePanel line-sync,
  primitives on real golden data.
- **Phase 3 — Chapters 0–2** ✅ · **Phase 4 — Chapters 3–7** ✅ ·
  **Phase 5 — Chapters 8–11** ✅ — all twelve chapters live, four pillars self-reviewed
  per phase (screenshots in docs/journal/).
- **Phase 6 — Ship** ✅ done — perf + a11y + mobile pass, README, Pages workflow.

## Performance budget

Training ≪ 30 s in-worker (measure and report actual; expected seconds), 60 fps
interactions, traces recorded on demand, lazy-loaded chapters, precomputed-run payload
< 1.5 MB (41 snapshots × 4,192 f32 ≈ 690 KB raw before base64/gzip — fits).
