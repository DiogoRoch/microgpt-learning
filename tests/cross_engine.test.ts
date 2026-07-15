/**
 * Cross-engine agreement: the scalar graph engine (gradients accumulated node
 * by node through the recorded computation graph) and the vectorized engine
 * (hand-written closed-form backward per op) must agree on the same weights.
 * Both engines see IDENTICAL f32-rounded weights so the only differences are
 * f32 round-off in the tensor engine's intermediate stores.
 */
import { describe, expect, it } from 'vitest'
import { docLoss, flattenParams, loadScalarStateDict, MICRO_CONFIG, stateDictKeys } from '../src/engine/model_scalar.ts'
import { Model } from '../src/engine/model.ts'
import { Tokenizer } from '../src/engine/tokenizer.ts'
import { expectClose } from './helpers.ts'
import initWeights from '../golden/init_weights.json'
import tokenizerGolden from '../golden/tokenizer.json'

const cfg = MICRO_CONFIG
const TOL = { atol: 1e-5, rtol: 2e-4 }

/** Round golden f64 weights through f32 so both engines start identical. */
function f32Weights(): Record<string, number[][]> {
  const src = initWeights as Record<string, number[][]>
  const out: Record<string, number[][]> = {}
  for (const key of stateDictKeys(cfg.nLayer)) {
    out[key] = src[key]!.map((row) => [...Float32Array.from(row)])
  }
  return out
}

describe('scalar ⇄ tensor cross-engine agreement', () => {
  const weights = f32Weights()
  const tok = Tokenizer.fromMeta(tokenizerGolden)
  const tokens = tok.encodeDoc('emma')

  it('forward and backward agree on emma', () => {
    const sd = loadScalarStateDict(weights, cfg)
    const scalarLoss = docLoss(sd, cfg, tokens)
    scalarLoss.loss.backward()
    const scalarParams = flattenParams(sd, cfg)

    const model = Model.fromWeights(cfg, weights)
    const { tape, loss } = model.docLoss(tokens)
    tape.backward(loss)

    expectClose([loss.data], [scalarLoss.loss.data], { ...TOL, label: 'loss' })
    expectClose(model.flatGrads(), scalarParams.map((p) => p.grad), { ...TOL, label: 'grads' })
  })

  it('per-position probabilities agree on emma', () => {
    const sd = loadScalarStateDict(weights, cfg)
    const scalar = docLoss(sd, cfg, tokens)
    const model = Model.fromWeights(cfg, weights)
    const { perPosLoss } = model.docLoss(tokens)
    expect(perPosLoss).toHaveLength(scalar.perPosition.length)
    for (let p = 0; p < perPosLoss.length; p++) {
      expectClose([perPosLoss[p]!], [scalar.perPosition[p]!.lossT.data], { ...TOL, label: `loss_t pos ${p}` })
    }
  })
})
