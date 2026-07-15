/**
 * Scalar engine (graph.ts + model_scalar.ts) vs golden dumps. Both sides are
 * f64 with identical op order, so tolerances are essentially libm ulp noise:
 * atol/rtol 1e-9 (brief: "scalar engine matches within ~1e-9").
 */
import { describe, expect, it } from 'vitest'
import {
  MICRO_CONFIG, adamStep, docLoss, flattenParams, gpt, initAdam, loadScalarStateDict, emptyKV,
} from '../src/engine/model_scalar.ts'
import type { DocTrace } from '../src/engine/trace.ts'
import { expectClose } from './helpers.ts'
import initWeights from '../golden/init_weights.json'
import step0Trace from '../golden/step0_trace.json'
import step0State from '../golden/step0_state.json'
import step1State from '../golden/step1_state.json'
import step2State from '../golden/step2_state.json'
import forcedDecodeInit from '../golden/forced_decode_init.json'

const cfg = MICRO_CONFIG
const TOL = { atol: 1e-9, rtol: 1e-9 }
const weights = initWeights as Record<string, number[][]>

function freshTrace(): DocTrace {
  return { tokens: [], n: 0, gpt_calls: [], positions: [], loss: 0 }
}

describe('scalar engine ⇄ golden step-0 forward trace', () => {
  const sd = loadScalarStateDict(weights, cfg)
  const trace = freshTrace()
  const { loss, n } = docLoss(sd, cfg, step0Trace.tokens, trace)

  it('walks the same document', () => {
    expect(n).toBe(step0Trace.n)
    expect(trace.gpt_calls).toHaveLength(step0Trace.gpt_calls.length)
  })

  it('matches every named intermediate at every position', () => {
    const vecFields = ['tok_emb', 'pos_emb', 'x_emb_sum', 'x_emb_norm', 'logits'] as const
    const layerFields = [
      'x_ln_attn', 'q', 'k', 'v', 'x_attn', 'x_wo', 'x_after_attn',
      'x_ln_mlp', 'fc1', 'relu', 'fc2', 'x_after_mlp',
    ] as const
    const headFields = ['attn_logits', 'attn_weights', 'head_out'] as const
    for (let p = 0; p < step0Trace.gpt_calls.length; p++) {
      const got = trace.gpt_calls[p]!
      const want = step0Trace.gpt_calls[p]!
      for (const f of vecFields) {
        expectClose(got[f], want[f], { ...TOL, label: `pos ${p} ${f}` })
      }
      for (let li = 0; li < want.layers.length; li++) {
        for (const f of layerFields) {
          expectClose(got.layers[li]![f], want.layers[li]![f], { ...TOL, label: `pos ${p} layer ${li} ${f}` })
        }
        for (let h = 0; h < want.layers[li]!.heads.length; h++) {
          for (const f of headFields) {
            expectClose(got.layers[li]!.heads[h]![f], want.layers[li]!.heads[h]![f], {
              ...TOL, label: `pos ${p} layer ${li} head ${h} ${f}`,
            })
          }
        }
      }
    }
  })

  it('matches probs and per-position losses', () => {
    for (let p = 0; p < step0Trace.positions.length; p++) {
      const got = trace.positions[p]!
      const want = step0Trace.positions[p]!
      expect(got.token_id).toBe(want.token_id)
      expect(got.target_id).toBe(want.target_id)
      expectClose(got.probs, want.probs, { ...TOL, label: `pos ${p} probs` })
      expectClose([got.loss_t], [want.loss_t], { ...TOL, label: `pos ${p} loss_t` })
    }
  })

  it('matches the document loss', () => {
    expectClose([loss.data], [step0Trace.loss], { ...TOL, label: 'doc loss' })
  })
})

describe('scalar engine ⇄ golden backward + 3 Adam steps', () => {
  it('reproduces the full gradient vector and Adam m/v/params for steps 0–2', () => {
    const sd = loadScalarStateDict(weights, cfg)
    const params = flattenParams(sd, cfg)
    expect(params).toHaveLength(4192)
    const adam = initAdam(params.length, 1000)
    const states = [step0State, step1State, step2State]
    for (let step = 0; step < 3; step++) {
      const want = states[step]!
      const { loss } = docLoss(sd, cfg, want.tokens)
      expectClose([loss.data], [want.loss], { atol: 1e-9, rtol: 1e-9, label: `step ${step} loss` })
      loss.backward()
      expectClose(params.map((p) => p.grad), want.grads, { atol: 1e-9, rtol: 1e-9, label: `step ${step} grads` })
      const lrT = adamStep(params, adam, step)
      expectClose([lrT], [want.lr_t], { atol: 1e-12, rtol: 0, label: `step ${step} lr_t` })
      expectClose(adam.m, want.m, { atol: 1e-9, rtol: 1e-9, label: `step ${step} adam m` })
      expectClose(adam.v, want.v, { atol: 1e-12, rtol: 1e-9, label: `step ${step} adam v` })
      expectClose(params.map((p) => p.data), want.params, { atol: 1e-9, rtol: 1e-9, label: `step ${step} params` })
    }
  })
})

describe('scalar engine ⇄ golden forced decode (init weights)', () => {
  it('reproduces per-position logits for emma', () => {
    const sd = loadScalarStateDict(weights, cfg)
    const kv = emptyKV(cfg)
    for (let pos = 0; pos < forcedDecodeInit.tokens.length; pos++) {
      const logits = gpt(sd, cfg, forcedDecodeInit.tokens[pos]!, pos, kv)
      expectClose(logits.map((l) => l.data), forcedDecodeInit.logits[pos]!, {
        ...TOL, label: `forced decode pos ${pos}`,
      })
    }
  })
})
