# MicroGPT, Explained Interactively

An interactive learning experience that teaches the transformer architecture through
[Andrej Karpathy's **microgpt**](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
— a ~200-line, pure-Python, dependency-free GPT that trains on 32,033 names and invents
new ones.

Play through twelve chapters and come out understanding **every line of microgpt.py** —
not a vague intuition of "attention", but the actual mechanics of this file: how a name
becomes tokens, how the computation graph carries gradients, why the loss starts at
exactly ln 27, why there is **no causal mask anywhere in the code**, and what temperature
really does to sampling.

## The four pillars

1. **Every number on screen is real.** Produced by the in-repo engine or the shipped
   precomputed run — never hardcoded, never faked.
2. **Code-synced.** Each chapter shows the actual lines of microgpt.py; interactions
   highlight the exact line they embody. The UI speaks the file's own vocabulary
   (`tok_emb`, `attn_logits`, `x_residual`, `m_hat`, …).
3. **Manipulable.** Scrub any process, hover any value, edit one of the 4,192 weights
   and watch the ripple, train the whole model live in a Web Worker (~1 s).
4. **Layered depth.** The main flow stays beginner-friendly; math and "in the wild"
   bridges to real GPTs live in expandable asides.

## Correctness: golden-value parity

`tools/golden.py` is a mechanically instrumented copy of the reference file (every added
line marked, no `random.*` calls added — the RNG stream is byte-identical, proven by the
20 final samples matching a pristine run). It dumps ground truth to `golden/`:
tokenizer, shuffled doc order, initial weights, every named intermediate of step 0, full
gradient vectors and Adam state for steps 0–2, per-step losses for all 1000 steps,
forced decodes at three weight states, final weights and samples.

Two TypeScript engines are tested against those dumps on every `npm test`:

- `src/engine/graph.ts` — a scalar autograd twin of `class Value` (construction-time
  local grads, iterative topo-sort backward, Python operator desugaring). Matches at
  **atol/rtol 1e-9**.
- `src/engine/tensor.ts` + `model.ts` — the fast Float32Array tape engine that powers
  the app. Matches forward/backward/Adam within f32 tolerance (measured max deviation
  ~1e-7); its full 1000-step training run tracks the Python loss curve within **4.2e-7**
  at every step and trains in **~0.3 s** (the pure-Python file takes ~262 s).

Sanity facts are pinned as tests and surfaced in the app: 4,192 parameters, initial
loss = ln 27 exactly on a zero-weight model, attention weights at position 0 exactly
`[1.0]`.

## Development

```bash
npm install
npm run golden      # fetch reference/microgpt.py (pinned gist rev) + regenerate golden/ (~4 min, needs python3)
npm run train       # train the TS engine → src/data artifacts (run.json, names.json, facts.json)
npm run dev         # dev server
npm test            # engine + parity suite (fetches the reference file if missing)
npm run typecheck   # tsc
npm run lint        # eslint
npm run build       # static production build (dist/)
```

`golden/` and `src/data/` artifacts are committed, so after `npm install` you can go
straight to `npm run dev` — the fetch script pulls `reference/microgpt.py` automatically
(it is intentionally not committed; the gist has no license, so we fetch and attribute
rather than redistribute).

## Architecture

```
reference/microgpt.py     # fetched source of truth (gitignored, pinned revision)
tools/                    # fetch script, instrumented golden.py, checkpoint trainer
golden/                   # committed ground-truth dumps from the Python reference
src/engine/               # UI-agnostic, parity-tested engines (scalar + f32 tensor)
src/worker/               # training worker + typed message protocol
src/viz/                  # VectorChips, MatrixHeatmap, AttnMatrix, BarDistribution,
                          # GraphView, LossCurve — hand-rolled SVG/DOM primitives
src/components/           # CodePanel (line-sync), Minimap, StepPlayer, PredictReveal…
src/chapters/             # ch00 – ch11
src/data/                 # names list, precomputed run (41 snapshots), slim payloads
```

The signature element: **the file itself is the navigation.** A minimap of all 200
lines runs down the edge; each chapter's lines light up as you complete it, and clicking
any region jumps to the chapter that teaches it.

## Deployment

Pushes to the default branch deploy to GitHub Pages via
`.github/workflows/deploy.yml` (build with `VITE_BASE=/microgpt-learning/`, SPA
fallback via `404.html`). Any static host works: `VITE_BASE=/ npm run build` and serve
`dist/`.

> [!IMPORTANT]
> **GitHub Pages source must be set to "GitHub Actions"**, *not* "Deploy from a
> branch". Under **Settings → Pages → Build and deployment → Source**, choose
> **GitHub Actions**. If Pages is left on "Deploy from a branch" (`main` / root), it
> serves the repository's raw `index.html` — whose `<script src="/src/main.tsx">` is
> Vite's dev-only entry and does not exist as a served file — so the site loads a
> **blank page**. The `deploy.yml` build artifact is only served when the source is
> "GitHub Actions".

## Attribution

**microgpt.py is by [Andrej Karpathy](https://github.com/karpathy)** —
[the gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). This
project re-implements it for the browser purely as an instrument for teaching that
file, and is parity-tested against it. The names dataset comes from Karpathy's
[makemore](https://github.com/karpathy/makemore).
