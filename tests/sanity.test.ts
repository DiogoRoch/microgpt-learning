/**
 * The teaching facts the app itself surfaces — encoded as tests so the app
 * can never drift from them (brief §7).
 */
import { describe, expect, it } from 'vitest'
import { docLoss, loadScalarStateDict, MICRO_CONFIG, stateDictKeys } from '../src/engine/model_scalar.ts'
import { Model } from '../src/engine/model.ts'
import { Tokenizer } from '../src/engine/tokenizer.ts'
import type { GptCallTrace, DocTrace } from '../src/engine/trace.ts'
import initWeights from '../golden/init_weights.json'
import step0Trace from '../golden/step0_trace.json'
import lossesGolden from '../golden/losses.json'
import metaGolden from '../golden/meta.json'
import tokenizerGolden from '../golden/tokenizer.json'
import finalGolden from '../golden/final.json'

const cfg = MICRO_CONFIG

describe('sanity facts the app teaches', () => {
  it('the model has exactly 4,192 parameters', () => {
    expect(metaGolden.num_params).toBe(4192)
    expect(Model.fromWeights(cfg, initWeights as Record<string, number[][]>).numParams).toBe(4192)
    // 27·16 (wte) + 16·16 (wpe) + 27·16 (lm_head) + 4·16·16 (attn) + 64·16 + 16·64 (mlp)
    expect(432 + 256 + 432 + 1024 + 1024 + 1024).toBe(4192)
  })

  it('initial loss ≈ ln(vocab_size): uniform guessing over 27 tokens', () => {
    expect(Math.abs(lossesGolden.losses[0]! - Math.log(27))).toBeLessThan(0.15)
    // With all-zero weights the logits are exactly zero → probs exactly uniform
    // → loss exactly ln(27). The whole pipeline reproduces the cold-start fact.
    const zeros: Record<string, number[][]> = {}
    for (const key of stateDictKeys(cfg.nLayer)) {
      const src = (initWeights as Record<string, number[][]>)[key]!
      zeros[key] = src.map((row) => row.map(() => 0))
    }
    const sd = loadScalarStateDict(zeros, cfg)
    const tok = Tokenizer.fromMeta(tokenizerGolden)
    const { loss } = docLoss(sd, cfg, tok.encodeDoc('emma'))
    expect(loss.data).toBeCloseTo(Math.log(27), 12)
  })

  it('attention weights at position 0 are exactly [1.0] in golden and both engines', () => {
    for (const head of step0Trace.gpt_calls[0]!.layers[0]!.heads) {
      expect(head.attn_weights).toEqual([1.0])
    }
    const model = Model.fromWeights(cfg, initWeights as Record<string, number[][]>)
    const trace: GptCallTrace[] = []
    model.docLoss(step0Trace.tokens, trace)
    for (const head of trace[0]!.layers[0]!.heads) {
      expect(head.attn_weights).toEqual([1.0])
    }
    const sd = loadScalarStateDict(initWeights as Record<string, number[][]>, cfg)
    const scalarTrace: DocTrace = { tokens: [], n: 0, gpt_calls: [], positions: [], loss: 0 }
    docLoss(sd, cfg, step0Trace.tokens, scalarTrace)
    for (const head of scalarTrace.gpt_calls[0]!.layers[0]!.heads) {
      expect(head.attn_weights).toEqual([1.0])
    }
  })

  it('training reached a name-like final loss and produced 20 samples', () => {
    expect(lossesGolden.losses).toHaveLength(1000)
    expect(finalGolden.final_loss).toBeLessThan(2.8)
    expect(finalGolden.samples).toHaveLength(20)
  })
})
