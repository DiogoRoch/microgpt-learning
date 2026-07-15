/**
 * Vectorized f32 engine (tensor.ts + model.ts) vs golden dumps.
 *
 * Storage is f32 (~7 significant digits); intermediates here are O(0.01–5),
 * and error compounds mildly through the ~10 chained ops of one forward pass
 * and across 3 Adam steps. Measured max deviations (see NOTES.md) sit well
 * below atol 1e-5 + rtol 2e-4 for forward/grads and atol 1e-5 + rtol 1e-3
 * after repeated optimizer steps; Adam's v gets an atol floor because its
 * entries are O(grad²) ≈ 1e-10.
 */
import { describe, expect, it } from 'vitest'
import { Adam, MICRO_CONFIG, Model } from '../src/engine/model.ts'
import type { GptCallTrace } from '../src/engine/trace.ts'
import { expectClose, maxDeviation } from './helpers.ts'
import initWeights from '../golden/init_weights.json'
import finalGolden from '../golden/final.json'
import step0Trace from '../golden/step0_trace.json'
import step0State from '../golden/step0_state.json'
import step1State from '../golden/step1_state.json'
import step2State from '../golden/step2_state.json'
import forcedDecodeInit from '../golden/forced_decode_init.json'
import forcedDecodeStep3 from '../golden/forced_decode_step3.json'
import forcedDecodeFinal from '../golden/forced_decode_final.json'

const cfg = MICRO_CONFIG
const FWD = { atol: 1e-5, rtol: 2e-4 }
const OPT = { atol: 1e-5, rtol: 1e-3 }
const weights = initWeights as Record<string, number[][]>
const finalWeights = (finalGolden as { state_dict: Record<string, number[][]> }).state_dict

describe('tensor engine ⇄ golden step-0 forward trace', () => {
  const model = Model.fromWeights(cfg, weights)
  const trace: GptCallTrace[] = []
  const { loss, n, perPosLoss } = model.docLoss(step0Trace.tokens, trace)

  it('has 4192 params and walks the same document', () => {
    expect(model.numParams).toBe(4192)
    expect(n).toBe(step0Trace.n)
  })

  it('matches every named intermediate at every position', () => {
    const vecFields = ['tok_emb', 'pos_emb', 'x_emb_sum', 'x_emb_norm', 'logits'] as const
    const layerFields = [
      'x_ln_attn', 'q', 'k', 'v', 'x_attn', 'x_wo', 'x_after_attn',
      'x_ln_mlp', 'fc1', 'relu', 'fc2', 'x_after_mlp',
    ] as const
    const headFields = ['attn_logits', 'attn_weights', 'head_out'] as const
    for (let p = 0; p < step0Trace.gpt_calls.length; p++) {
      const got = trace[p]!
      const want = step0Trace.gpt_calls[p]!
      for (const f of vecFields) expectClose(got[f], want[f], { ...FWD, label: `pos ${p} ${f}` })
      for (let li = 0; li < want.layers.length; li++) {
        for (const f of layerFields) {
          expectClose(got.layers[li]![f], want.layers[li]![f], { ...FWD, label: `pos ${p} layer ${li} ${f}` })
        }
        for (let h = 0; h < want.layers[li]!.heads.length; h++) {
          for (const f of headFields) {
            expectClose(got.layers[li]!.heads[h]![f], want.layers[li]!.heads[h]![f], {
              ...FWD, label: `pos ${p} layer ${li} head ${h} ${f}`,
            })
          }
        }
      }
    }
  })

  it('matches per-position and document losses', () => {
    expectClose(perPosLoss, step0Trace.positions.map((p) => p.loss_t), { ...FWD, label: 'per-position losses' })
    expectClose([loss.data], [step0Trace.loss], { ...FWD, label: 'doc loss' })
  })
})

describe('tensor engine ⇄ golden backward + 3 Adam steps', () => {
  it('reproduces gradients and Adam m/v/params for steps 0–2', () => {
    const model = Model.fromWeights(cfg, weights)
    const adam = new Adam(model.numParams, 1000)
    const states = [step0State, step1State, step2State]
    for (let step = 0; step < 3; step++) {
      const want = states[step]!
      const { tape, loss } = model.docLoss(want.tokens)
      expectClose([loss.data], [want.loss], { ...FWD, label: `step ${step} loss` })
      tape.backward(loss)
      const grads = model.flatGrads()
      const gradDev = maxDeviation(grads, want.grads)
      expectClose(grads, want.grads, { ...FWD, label: `step ${step} grads` })
      const lrT = adam.step(model, step)
      expectClose([lrT], [want.lr_t], { atol: 1e-12, rtol: 0, label: `step ${step} lr_t` })
      expectClose(adam.m, want.m, { ...OPT, label: `step ${step} adam m` })
      expectClose(adam.v, want.v, { atol: 1e-12, rtol: 2e-3, label: `step ${step} adam v` })
      const paramDev = maxDeviation(model.flatParams(), want.params)
      expectClose(model.flatParams(), want.params, { ...OPT, label: `step ${step} params` })
      // eslint-disable-next-line no-console
      console.log(
        `[tensor step ${step}] grads maxAbs=${gradDev.maxAbs.toExponential(2)} ` +
          `params maxAbs=${paramDev.maxAbs.toExponential(2)}`,
      )
    }
  })
})

describe('tensor engine ⇄ golden forced decodes at three weight states', () => {
  it('matches with init weights', () => {
    const model = Model.fromWeights(cfg, weights)
    const logits = model.forcedDecode(forcedDecodeInit.tokens)
    for (let p = 0; p < logits.length; p++) {
      expectClose(logits[p]!, forcedDecodeInit.logits[p]!, { ...FWD, label: `init pos ${p}` })
    }
  })

  it('matches after 3 training steps', () => {
    const model = Model.fromWeights(cfg, weights)
    const adam = new Adam(model.numParams, 1000)
    for (const [step, st] of [step0State, step1State, step2State].entries()) {
      const { tape, loss } = model.docLoss(st.tokens)
      tape.backward(loss)
      adam.step(model, step)
    }
    const logits = model.forcedDecode(forcedDecodeStep3.tokens)
    for (let p = 0; p < logits.length; p++) {
      expectClose(logits[p]!, forcedDecodeStep3.logits[p]!, { ...OPT, label: `step3 pos ${p}` })
    }
  })

  it('matches with the fully-trained weights', () => {
    const model = Model.fromWeights(cfg, finalWeights)
    const logits = model.forcedDecode(forcedDecodeFinal.tokens)
    for (let p = 0; p < logits.length; p++) {
      expectClose(logits[p]!, forcedDecodeFinal.logits[p]!, { ...FWD, label: `final pos ${p}` })
    }
  })
})
