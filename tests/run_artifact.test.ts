/**
 * The shipped precomputed run must hold up against Python ground truth:
 * decoding the final snapshot and forced-decoding emma should land near the
 * golden forced decode of the fully-trained Python model. The two trainings
 * are f32-vs-f64 twins, so tolerances here bound the ACCUMULATED drift of
 * 1000 optimizer steps (measured: loss drift ≤ 4.2e-7; logits drift ~1e-4).
 */
import { describe, expect, it } from 'vitest'
import { decodeF32, modelAtSnapshot, snapshotForStep, type RunData } from '../src/data/loadRun.ts'
import { Tokenizer } from '../src/engine/tokenizer.ts'
import { expectClose, maxDeviation } from './helpers.ts'
import runJson from '../src/data/run.json'
import forcedDecodeFinal from '../golden/forced_decode_final.json'
import tokenizerGolden from '../golden/tokenizer.json'
import step0State from '../golden/step0_state.json'

const run = runJson as unknown as RunData

describe('shipped precomputed run artifact', () => {
  it('has the expected shape', () => {
    expect(run.numSteps).toBe(1000)
    expect(run.losses).toHaveLength(1000)
    expect(run.pythonLosses).toHaveLength(1000)
    expect(run.snapshots).toHaveLength(41)
    expect(run.snapshots[0]!.step).toBe(0)
    expect(run.snapshots.at(-1)!.step).toBe(1000)
  })

  it('snapshots decode to 4192 params', () => {
    for (const s of [run.snapshots[0]!, run.snapshots.at(-1)!]) {
      expect(decodeF32(s.params_b64)).toHaveLength(4192)
    }
  })

  it('snapshot scrubbing picks the right checkpoint', () => {
    expect(snapshotForStep(run, 0).step).toBe(0)
    expect(snapshotForStep(run, 24).step).toBe(0)
    expect(snapshotForStep(run, 25).step).toBe(25)
    expect(snapshotForStep(run, 999).step).toBe(975)
    expect(snapshotForStep(run, 1000).step).toBe(1000)
  })

  it('final snapshot reproduces the trained Python model (forced decode emma)', () => {
    const model = modelAtSnapshot(run, run.snapshots.at(-1)!)
    const logits = model.forcedDecode(forcedDecodeFinal.tokens)
    let worst = { maxAbs: 0, maxRel: 0 }
    for (let p = 0; p < logits.length; p++) {
      const dev = maxDeviation(logits[p]!, forcedDecodeFinal.logits[p]!)
      worst = { maxAbs: Math.max(worst.maxAbs, dev.maxAbs), maxRel: Math.max(worst.maxRel, dev.maxRel) }
      expectClose(logits[p]!, forcedDecodeFinal.logits[p]!, { atol: 5e-3, rtol: 5e-3, label: `final pos ${p}` })
    }
    // eslint-disable-next-line no-console
    console.log(`[run artifact] final-snapshot logits vs python: maxAbs=${worst.maxAbs.toExponential(2)}`)
  })

  it('the step-0 snapshot is the untrained init (loss ≈ ln 27 on doc 0)', () => {
    const model = modelAtSnapshot(run, run.snapshots[0]!)
    const tok = Tokenizer.fromMeta(tokenizerGolden)
    const { loss } = model.docLoss(tok.encodeDoc(step0State.doc))
    expect(Math.abs(loss.data - step0State.loss)).toBeLessThan(1e-5)
  })
})
