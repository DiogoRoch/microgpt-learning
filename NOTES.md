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

- **RNG-identity proof passed.** Pristine reference run vs instrumented golden.py:
  all 20 sampled names identical (kamon, ann, karai, …) and final loss 2.6497
  matches. Full 1000-step pure-Python run: 262 s on this machine.
- **Golden facts:** first shuffled doc "yuheng" (n=7, step-1 loss 3.3660 ≈ ln 27);
  emma → [26,4,12,12,0,26]; 3,728 of 4,192 grads nonzero at step 0 (unused wte/wpe
  rows get none — good chapter-8 material); golden/ totals ~1.4 MB.
- **Single tsconfig** instead of project references: tests import engine sources
  directly and `composite` adds friction with `noEmit`; one strict config covers
  src + tools + tests.

## Phase 1 — Engines

- **Training speed:** the f32 TS engine trains the full 1000 steps in **0.33 s**
  single-threaded under Node (0.3 ms/step) vs 262 s for the pure-Python reference —
  ~800×. Live in-browser training is comfortably a headline feature; chapter 9 can
  re-run the whole thing on every click if it wants.
- **Measured parity headroom** (declared tolerance ⇢ measured): scalar vs golden
  1e-9 ⇢ passes with f64-identical op order (libm-ulp level); tensor forward/grads
  atol 1e-5 + rtol 2e-4 ⇢ measured maxAbs 5.3e-8 (grads), 1.9e-7 (params after 3 Adam
  steps); full-run trajectory gate 1e-4 ⇢ measured max |Δloss| 4.2e-7, final |Δ|
  8.7e-8; shipped final snapshot vs Python forced decode ⇢ maxAbs 2.4e-6.
- **Graph-shape fidelity matters:** vsum mirrors Python `sum()`'s `x0 + 0` first node
  (via `__radd__`), division by a Value is a pow node, by a number a constant
  multiply, softmax max-subtract is a constant. With those mirrored, scalar parity
  needs no tolerance slack at all.
- **Adam buffers kept in f64** (params/grads f32): matches the reference's float m/v
  and keeps v (≈ grad², O(1e-10)) out of f32 denormal territory.
- **run.json ships both loss curves** (TS + Python) so chapter 9 can overlay "your
  browser's run" on "the file's run" — honest numbers, no hand-waving.
- **Worker protocol** supports golden-init replay ("the exact run the file does") and
  fresh seeded inits; chunked loop yields every 10 steps so stop messages land.

## Phase 2 — Shell

- **Screenshots:** docs/journal/phase2-home.png (chapter frame, minimap, code panel),
  docs/journal/phase2-gallery.png (all primitives on real golden data + live code sync).
- **Custom highlighter instead of Shiki** (brief said push back with reasons): we render
  ONE known 200-line file; a 90-line tokenizer gives zero bundle cost and full control
  over line anatomy for sync/minimap. Two tests guard it: verbatim reconstruction of
  the file from tokens, and pinned classifications of known lines.
- **The file is 200 lines, not 199** — `wc -l` says 199 because the last line has no
  trailing newline. TOTAL_LINES = 200 everywhere; a test asserts it.
- **File map invariants tested:** every non-blank line has exactly one owning chapter,
  ranges never overlap, all 12 slugs unique. The minimap is generated from the real
  source (indentation + line length drawn per line), so it *is* the file.
- **Minimap placed left** (book-spine metaphor); code panel right and sticky; the
  narrative column between them. Below xl, the code panel hides (chapters keep inline
  excerpts; mobile pass comes in Phase 6).
- **BrowserRouter + configurable base** (VITE_BASE) for GitHub Pages; deploy workflow
  will copy index.html → 404.html for SPA fallback.

## Phase 3 — Chapters 0–2

- **Screenshots:** docs/journal/phase3-ch{0,1,2}.png.
- **Four-pillar self-review:** (1) real numbers — ch0 cards read facts.json from the
  golden run, ch1 builds the vocab live from all 32,033 names with the file's exact
  recipe, ch2 runs the actual scalar engine; (2) code-sync — flow-map hover, vocab
  hover, and the backward stepper drive the panel (line 69 on seed, 70–72 while
  stepping, op lines on node hover); (3) manipulable — round-trip input feeds the
  global running example, expression sandbox differentiates arbitrary input;
  (4) layered depth — math aside derives the chain rule, wild asides bridge BPE and
  PyTorch.
- **Completion = quizzes.** A chapter auto-completes when all its PredictReveals are
  revealed (QuizProvider tracks qids); Recap offers a manual fallback when a chapter
  has no quizzes. Progress persists in localStorage and lights the minimap.
- **Sandbox parser** builds graphs through the engine's own desugaring ops so division
  really creates a pow node, unary minus really multiplies by −1 — the sandbox can't
  drift from Value semantics. Parser has its own test file (7 tests).
- **GraphView never mutates nodes**: backward stepping uses partialGrads() into a Map,
  so scrubbing back and forth is pure. planBackward() skips leaf no-ops so every step
  shown does real work.
- **facts.json** (tiny) added for light chapters; heavy artifacts (names.json 250 KB,
  run.json 916 KB) only load inside the routes that teach them.

## Phase 4 — Chapters 3–7

- **Screenshots:** docs/journal/phase4-ch{3,4,5,6,7}.png.
- **Live-model plumbing:** useModel.ts caches Models per snapshot; useTrace(word, step)
  re-runs the real f32 engine on the user's example on every render-relevant change
  (a full traced forward is ~microseconds). CompareToggle flips the app-wide
  checkpoint (store.checkpointStep) so every viz in a chapter re-renders from the
  other weights.
- **Ch4's perturbation lab** clones the model via flatParams → edit one wte cell →
  Model.fromFlat → full re-trace. The per-position Σ|Δp| ripple bar makes causality
  visible (positions before the edit are exactly 0) — deliberately foreshadows ch5.
- **Ch5 mask-equivalence** renders the SAME trace numbers in both framings; the
  "textbook" upper triangle is drawn as −∞ cells over numbers that were never
  computed. Copy stresses inference-form vs batched-training-form.
- **Honest-numbers moment:** the trained model shows 63/64 dead ReLU neurons at BOS
  pos 0 in ch6 — real trace data, kept (with the dead-neuron count computed live)
  rather than smoothed over.
- **Ch3 treemap** rotates labels on narrow blocks (attention 256-cell matrices).
- All five chapters: hook → interactive core → exact-lines CodePanel sync → recap +
  3 PredictReveals, per the brief's chapter contract.

## Phase 5 — Chapters 8–11

- **Screenshots:** docs/journal/phase5-ch{8,9-live,10,11}.png. The ch9 screenshot is
  from an automated Playwright run that actually clicked "train" — captured mid-proof.
- **Live training verified in a real browser:** 1000 steps in 1.41 s (dev mode,
  flat-out pace) inside Chromium via the worker. A "watchable" pace (~20 s) is the
  default so the curve draws itself and samples visibly evolve (step 5:
  "kvirqguvkqhdckuh" → step 1000: "kana, canan, cedela"). tools/test_training.mjs
  automates this check.
- **Worker protocol grew paceMs + lossesChunk** so progress messages carry the full
  loss segment (smooth curve, no stair-steps) and humans can watch a run that would
  otherwise finish in a blink.
- **Ch8 shows the real optimizer numbers**: grad/m/v/m̂/lr_t/updated-p for selected
  real parameters across golden steps 0–2, including a parameter whose grad is 0 at
  step 0 (the 'a' embedding — absent from "yuheng"). The gradient-field heatmaps make
  the 464 zero-grads visibly structural (unused wte/wpe rows), not numerical.
- **Ch10's novelty check** marks generated names against the full 32,033-name set —
  typically ~2/3 genuinely invented at T=0.5, and the copy explains why collisions
  aren't memorization (each name seen ≤ once).
- **Ch11 playground**: any prefix, any snapshot (LossCurve doubles as checkpoint
  scrubber), toggleable stage chips for every named intermediate, 4-head attention
  grid, name-list export via Blob download, and the full annotated file as the
  closing screen with a conquered-lines counter.

## Phase 6 — Ship

- **Perf pass.** Landing page ships ~61 KB gz JS + 7 KB CSS + fonts; heavy data is
  route-lazy: run.json (676 KB gz — base64 f32 resists compression; 41 snapshots is
  the honest budget for full time-travel, within the brief's 1.5 MB data target),
  names (108 KB gz, ch1/ch10 only), KaTeX (78 KB gz + fonts, math chapters only).
  Ch8's ~1 MB of golden step files was replaced by a 44 KB derived payload
  (adam_steps.json: step-0 grads at 6 s.f. + featured-param optimizer records) —
  full precision remains in golden/ for the tests.
- **Prod-build verification:** tools/test_training.mjs against `vite preview` —
  1000 live steps in 1.57 s in Chromium, no page errors.
- **Mobile pass.** Root cause of overflow: below xl the chapter grid's implicit
  column sized to max-content. Fixed with an explicit minmax(0,1fr) base track,
  overflow-x-auto wrappers on the wide SVGs (mask-equivalence, free-play grids), and
  body overflow-x: clip as a backstop. Code panel now follows the article below xl
  (50 vh, own scroll) instead of disappearing.
- **A11y pass.** Skip link; header/main/footer landmarks; focus-visible amber
  outlines throughout; PredictReveal options are a radiogroup; steppers accept
  ←/→/space with reduced-motion hiding autoplay; every SVG has role=img + aria-label;
  the minimap is aria-labeled with progress and its 200 hover targets stay
  mouse-only because the header chapter nav is the keyboard-equivalent route;
  gist-link contrast bumped. Live-training progress is textual (step counter +
  samples), not color-only.
- **Deploy:** .github/workflows/deploy.yml — npm ci → test (fetches the pinned
  reference) → build with VITE_BASE=/<repo>/ → 404.html SPA fallback → Pages.
  README documents setup, commands, architecture, parity story, attribution.
