/**
 * Trains the TypeScript f32 engine for the full 1000 steps from the golden
 * initial weights + golden shuffled doc order, validates the trajectory
 * against the Python run, measures wall-clock time, and writes the shipped
 * precomputed-run artifacts:
 *
 *   src/data/run.json    — loss history, 41 weight snapshots (base64 f32,
 *                          every 25 steps), periodic samples, timing
 *   src/data/names.json  — the full names dataset (for chapter 1's live
 *                          vocab build), from input.txt
 *
 * Run: npm run train   (requires golden/ to exist; fetches input.txt if missing)
 */
import { mkdirSync, existsSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { Adam, MICRO_CONFIG, Model, trainStep } from '../src/engine/model.ts'
import { RNG } from '../src/engine/rng.ts'
import { Tokenizer } from '../src/engine/tokenizer.ts'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const goldenDir = join(root, 'golden')
const dataDir = join(root, 'src', 'data')
mkdirSync(dataDir, { recursive: true })

const readJSON = <T>(p: string): T => JSON.parse(readFileSync(p, 'utf8')) as T

const initWeights = readJSON<Record<string, number[][]>>(join(goldenDir, 'init_weights.json'))
const tokenizerGolden = readJSON<{ uchars: string[]; bos: number; vocab_size: number }>(join(goldenDir, 'tokenizer.json'))
const docsGolden = readJSON<{ num_docs: number; head: string[] }>(join(goldenDir, 'docs.json'))
const lossesGolden = readJSON<{ losses: number[] }>(join(goldenDir, 'losses.json'))
const finalGolden = readJSON<{ final_loss: number; samples: string[] }>(join(goldenDir, 'final.json'))

// ---------------------------------------------------------------------------
// names.json — full dataset, original file order (chapter 1 builds the vocab
// from this live, exactly like `uchars = sorted(set(''.join(docs)))`).
const inputPath = join(root, 'input.txt')
if (!existsSync(inputPath)) {
  const NAMES_URL = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
  console.log(`input.txt missing — fetching ${NAMES_URL}`)
  const res = await fetch(NAMES_URL)
  if (!res.ok) throw new Error(`fetch names.txt: HTTP ${res.status}`)
  writeFileSync(inputPath, await res.text())
}
const names = readFileSync(inputPath, 'utf8')
  .split('\n')
  .map((l) => l.trim())
  .filter((l) => l.length > 0)

// Full-vocabulary check: the whole dataset yields exactly the golden uchars.
const fullVocab = [...new Set(names.join(''))].sort()
if (JSON.stringify(fullVocab) !== JSON.stringify(tokenizerGolden.uchars)) {
  throw new Error('vocabulary derived from input.txt does not match golden uchars')
}
writeFileSync(join(dataDir, 'names.json'), JSON.stringify({ names }))
console.log(`names.json: ${names.length} names, vocab check OK`)

// ---------------------------------------------------------------------------
// Train the f32 engine on the exact golden run.
const tok = Tokenizer.fromMeta(tokenizerGolden)
const model = Model.fromWeights(MICRO_CONFIG, initWeights)
const NUM_STEPS = 1000
const SNAPSHOT_EVERY = 25
const adam = new Adam(model.numParams, NUM_STEPS)

const b64 = (f32: Float32Array): string =>
  Buffer.from(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength)).toString('base64')

interface Snapshot {
  step: number
  loss: number | null
  params_b64: string
  samples: string[]
}

const sampleAt = (step: number, count = 8): string[] => {
  const rng = new RNG(4242 + step)
  return Array.from({ length: count }, () => tok.decode(model.sample(rng, 0.5)))
}

const snapshots: Snapshot[] = [{ step: 0, loss: null, params_b64: b64(model.flatParams()), samples: sampleAt(0) }]
const losses: number[] = []
const t0 = performance.now()
for (let step = 0; step < NUM_STEPS; step++) {
  const tokens = tok.encodeDoc(docsGolden.head[step % docsGolden.head.length]!)
  losses.push(trainStep(model, adam, tokens, step))
  if ((step + 1) % SNAPSHOT_EVERY === 0) {
    snapshots.push({ step: step + 1, loss: losses[step]!, params_b64: b64(model.flatParams()), samples: sampleAt(step + 1) })
  }
}
const trainMs = performance.now() - t0

// ---------------------------------------------------------------------------
// Validate the trajectory against the Python (f64) run.
let maxAbsDiff = 0
let maxAbsDiffStep = 0
for (let i = 0; i < NUM_STEPS; i++) {
  const d = Math.abs(losses[i]! - lossesGolden.losses[i]!)
  if (d > maxAbsDiff) {
    maxAbsDiff = d
    maxAbsDiffStep = i
  }
}
const finalDiff = Math.abs(losses[NUM_STEPS - 1]! - finalGolden.final_loss)
console.log(`train time: ${(trainMs / 1000).toFixed(2)}s for ${NUM_STEPS} steps (${(trainMs / NUM_STEPS).toFixed(1)} ms/step)`)
console.log(`loss[0]   ts=${losses[0]!.toFixed(6)}  py=${lossesGolden.losses[0]!.toFixed(6)}`)
console.log(`loss[999] ts=${losses[999]!.toFixed(6)}  py=${finalGolden.final_loss.toFixed(6)}  |Δ|=${finalDiff.toExponential(2)}`)
console.log(`max |Δloss| over run: ${maxAbsDiff.toExponential(2)} at step ${maxAbsDiffStep}`)

// f32 vs f64 drift compounds over 1000 sequential updates. Measured on the
// reference machine: max |Δloss| 4.2e-7 across the run, final |Δ| 8.7e-8
// (NOTES.md). Gate at 100× the measured drift to allow platform libm
// variation while still catching any real semantic regression:
if (Math.abs(losses[0]! - lossesGolden.losses[0]!) > 1e-5) throw new Error('step-0 loss diverges — engine bug, not drift')
if (maxAbsDiff > 1e-4) throw new Error(`loss trajectory drifted ${maxAbsDiff} > 1e-4 from the Python run`)
if (finalDiff > 1e-4) throw new Error(`final loss drifted ${finalDiff} > 1e-4 from the Python run`)

const samplesFinal = sampleAt(NUM_STEPS, 20)
console.log('sample names @1000:', samplesFinal.slice(0, 8).join(', '))

const run = {
  config: MICRO_CONFIG,
  numSteps: NUM_STEPS,
  snapshotEvery: SNAPSHOT_EVERY,
  trainMs: Math.round(trainMs),
  losses: losses.map((l) => Math.round(l * 1e5) / 1e5),
  pythonLosses: lossesGolden.losses.map((l) => Math.round(l * 1e5) / 1e5),
  finalLossPython: finalGolden.final_loss,
  maxAbsLossDiff: maxAbsDiff,
  snapshots,
}
writeFileSync(join(dataDir, 'run.json'), JSON.stringify(run))
const sizeKB = Math.round(Buffer.byteLength(JSON.stringify(run)) / 1024)
console.log(`run.json written: ${sizeKB} KB (${snapshots.length} snapshots)`)

// Small facts payload for light-weight chapters (landing page etc.) so they
// don't have to import the heavy artifacts.
const facts = {
  numDocs: docsGolden.num_docs,
  vocabSize: tokenizerGolden.vocab_size,
  numParams: model.numParams,
  numSteps: NUM_STEPS,
  loss0: lossesGolden.losses[0]!,
  finalLossPython: finalGolden.final_loss,
  finalLossTs: losses[NUM_STEPS - 1]!,
  pythonTrainSec: 262.4,
  tsTrainMs: Math.round(trainMs),
  samplesPython: finalGolden.samples,
  docHead: docsGolden.head.slice(0, 8),
}
writeFileSync(join(dataDir, 'facts.json'), JSON.stringify(facts, null, 1))
console.log('facts.json written')

// Slim chapter-8 payload derived from the golden dumps: the full step-0
// gradient field (rounded to 6 significant digits — display is 5) plus the
// complete optimizer record of a few featured real parameters across steps
// 0–2. Full precision stays in golden/ where the tests read it.
interface GoldenStep {
  step: number
  doc: string
  n: number
  loss: number
  lr_t: number
  grads: number[]
  m: number[]
  v: number[]
  params: number[]
}
const steps: GoldenStep[] = [0, 1, 2].map((k) => readJSON<GoldenStep>(join(goldenDir, `step${k}_state.json`)))
const initFlat: number[] = []
for (const key of Object.keys(initWeights)) for (const row of initWeights[key]!) initFlat.push(...row)
const r6 = (x: number) => Number(x.toPrecision(6))
const uch = tokenizerGolden.uchars
const FEATURED = [
  { label: `wte['y'][0] — used at step 0`, idx: uch.indexOf('y') * 16 },
  { label: `wte['a'][0] — absent from "${steps[0]!.doc}"`, idx: uch.indexOf('a') * 16 },
  { label: `lm_head['a'][0] — blamed at every position`, idx: (27 + 16) * 16 + uch.indexOf('a') * 16 },
  { label: `mlp_fc1[0][0]`, idx: (27 + 16 + 27 + 16 * 4) * 16 },
]
const adamSteps = {
  featured: FEATURED.map((f) => ({
    label: f.label,
    init: r6(initFlat[f.idx]!),
    steps: steps.map((st, k) => ({
      step: k,
      doc: st.doc,
      grad: r6(st.grads[f.idx]!),
      m: r6(st.m[f.idx]!),
      v: Number(st.v[f.idx]!.toPrecision(4)),
      lr_t: st.lr_t,
      param: r6(st.params[f.idx]!),
    })),
  })),
  step0: {
    doc: steps[0]!.doc,
    n: steps[0]!.n,
    loss: r6(steps[0]!.loss),
    grads: steps[0]!.grads.map(r6),
    nonzero: steps[0]!.grads.filter((g) => g !== 0).length,
  },
}
writeFileSync(join(dataDir, 'adam_steps.json'), JSON.stringify(adamSteps))
console.log(`adam_steps.json written (${Math.round(Buffer.byteLength(JSON.stringify(adamSteps)) / 1024)} KB)`)
