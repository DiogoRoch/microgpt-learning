/**
 * The full microgpt model on the vectorized engine (tensor.ts). Same
 * structure, same state_dict key names, same incremental KV-cache forward as
 * the file — just Float32Arrays instead of one Value per number. This is the
 * engine that trains live in the browser and powers full-model interactions.
 */

import {
  Mat, Scal, Tape, Vec,
  addVec, attnLogits, concatVecs, crossEntropy, linear, meanScals, relu,
  rmsnorm, rowVec, softmaxProbs, softmaxVec, weightedSum,
} from './tensor.ts'
import type { Config } from './model_scalar.ts'
import { MICRO_CONFIG, stateDictKeys } from './model_scalar.ts'
import type { RNG } from './rng.ts'
import type { GptCallTrace, LayerTrace } from './trace.ts'

export type { Config }
export { MICRO_CONFIG }

export interface KV {
  keys: Vec[][]
  values: Vec[][]
}

const matShape = (key: string, cfg: Config): [number, number] => {
  if (key === 'wte' || key === 'lm_head') return [cfg.vocabSize, cfg.nEmbd]
  if (key === 'wpe') return [cfg.blockSize, cfg.nEmbd]
  if (key.endsWith('mlp_fc1')) return [4 * cfg.nEmbd, cfg.nEmbd]
  if (key.endsWith('mlp_fc2')) return [cfg.nEmbd, 4 * cfg.nEmbd]
  return [cfg.nEmbd, cfg.nEmbd] // attn_wq/wk/wv/wo
}

export class Model {
  readonly cfg: Config
  readonly sd: Record<string, Mat>
  readonly keys: string[]
  readonly numParams: number

  private constructor(cfg: Config, sd: Record<string, Mat>) {
    this.cfg = cfg
    this.sd = sd
    this.keys = stateDictKeys(cfg.nLayer)
    this.numParams = this.keys.reduce((acc, k) => acc + sd[k]!.data.length, 0)
  }

  /** Gaussian init, std 0.08, in state_dict key order (like the file). */
  static init(cfg: Config, rng: RNG, std = 0.08): Model {
    const sd: Record<string, Mat> = {}
    for (const key of stateDictKeys(cfg.nLayer)) {
      const [nout, nin] = matShape(key, cfg)
      const m = new Mat(nout, nin)
      for (let i = 0; i < m.data.length; i++) m.data[i] = rng.gauss(0, std)
      sd[key] = m
    }
    return new Model(cfg, sd)
  }

  /** Load exact weights (golden dumps, snapshots). */
  static fromWeights(cfg: Config, weights: Record<string, number[][]>): Model {
    const sd: Record<string, Mat> = {}
    for (const key of stateDictKeys(cfg.nLayer)) {
      const rows = weights[key]
      if (!rows) throw new Error(`missing state_dict key ${key}`)
      sd[key] = Mat.fromRows(rows)
    }
    return new Model(cfg, sd)
  }

  /** Load from a flat f32 param vector (snapshot format), in param order. */
  static fromFlat(cfg: Config, flat: Float32Array): Model {
    const sd: Record<string, Mat> = {}
    let off = 0
    for (const key of stateDictKeys(cfg.nLayer)) {
      const [nout, nin] = matShape(key, cfg)
      const m = new Mat(nout, nin)
      m.data.set(flat.subarray(off, off + nout * nin))
      off += nout * nin
      sd[key] = m
    }
    if (off !== flat.length) throw new Error(`flat params length ${flat.length}, expected ${off}`)
    return new Model(cfg, sd)
  }

  /** Flat copy of all params, in the file's param order. */
  flatParams(): Float32Array {
    const out = new Float32Array(this.numParams)
    let off = 0
    for (const key of this.keys) {
      out.set(this.sd[key]!.data, off)
      off += this.sd[key]!.data.length
    }
    return out
  }

  /** Flat copy of all gradients, same order. */
  flatGrads(): Float32Array {
    const out = new Float32Array(this.numParams)
    let off = 0
    for (const key of this.keys) {
      out.set(this.sd[key]!.grad, off)
      off += this.sd[key]!.grad.length
    }
    return out
  }

  zeroGrads(): void {
    for (const key of this.keys) this.sd[key]!.zeroGrad()
  }

  emptyKV(): KV {
    return {
      keys: Array.from({ length: this.cfg.nLayer }, () => []),
      values: Array.from({ length: this.cfg.nLayer }, () => []),
    }
  }

  /**
   * One incremental forward pass — gpt(token_id, pos_id, keys, values).
   * Appends k,v for this position; attends over everything cached so far.
   */
  forward(tape: Tape, tokenId: number, posId: number, kv: KV, trace?: GptCallTrace[]): Vec {
    const { cfg, sd } = this
    const tokEmb = rowVec(tape, sd['wte']!, tokenId)
    const posEmb = rowVec(tape, sd['wpe']!, posId)
    const xEmbSum = addVec(tape, tokEmb, posEmb)
    let x = rmsnorm(tape, xEmbSum)
    let rec: GptCallTrace | undefined
    if (trace) {
      rec = {
        token_id: tokenId, pos_id: posId,
        tok_emb: [...tokEmb.data], pos_emb: [...posEmb.data],
        x_emb_sum: [...xEmbSum.data], x_emb_norm: [...x.data],
        layers: [], logits: [],
      }
      trace.push(rec)
    }

    for (let li = 0; li < cfg.nLayer; li++) {
      // 1) Multi-head Attention block
      let xResidual = x
      x = rmsnorm(tape, x)
      const q = linear(tape, sd[`layer${li}.attn_wq`]!, x)
      const k = linear(tape, sd[`layer${li}.attn_wk`]!, x)
      const v = linear(tape, sd[`layer${li}.attn_wv`]!, x)
      kv.keys[li]!.push(k)
      kv.values[li]!.push(v)
      let lrec: LayerTrace | undefined
      if (rec) {
        lrec = {
          x_ln_attn: [...x.data], q: [...q.data], k: [...k.data], v: [...v.data], heads: [],
          x_attn: [], x_wo: [], x_after_attn: [], x_ln_mlp: [], fc1: [], relu: [], fc2: [], x_after_mlp: [],
        }
        rec.layers.push(lrec)
      }
      const headOuts: Vec[] = []
      for (let h = 0; h < cfg.nHead; h++) {
        const hs = h * cfg.headDim
        const logits = attnLogits(tape, q, kv.keys[li]!, hs, cfg.headDim)
        const weights = softmaxVec(tape, logits)
        const headOut = weightedSum(tape, weights, kv.values[li]!, hs, cfg.headDim)
        headOuts.push(headOut)
        lrec?.heads.push({
          attn_logits: [...logits.data], attn_weights: [...weights.data], head_out: [...headOut.data],
        })
      }
      const xAttn = concatVecs(tape, headOuts)
      const xw = linear(tape, sd[`layer${li}.attn_wo`]!, xAttn)
      if (lrec) {
        lrec.x_attn = [...xAttn.data]
        lrec.x_wo = [...xw.data]
      }
      x = addVec(tape, xw, xResidual)
      if (lrec) lrec.x_after_attn = [...x.data]
      // 2) MLP block
      xResidual = x
      x = rmsnorm(tape, x)
      if (lrec) lrec.x_ln_mlp = [...x.data]
      x = linear(tape, sd[`layer${li}.mlp_fc1`]!, x)
      if (lrec) lrec.fc1 = [...x.data]
      x = relu(tape, x)
      if (lrec) lrec.relu = [...x.data]
      x = linear(tape, sd[`layer${li}.mlp_fc2`]!, x)
      if (lrec) lrec.fc2 = [...x.data]
      x = addVec(tape, x, xResidual)
      if (lrec) lrec.x_after_mlp = [...x.data]
    }

    const logits = linear(tape, sd['lm_head']!, x)
    if (rec) rec.logits = [...logits.data]
    return logits
  }

  /**
   * Whole-document average next-token loss on a fresh tape:
   * n = min(block_size, len(tokens)-1); mean of -log p(target).
   */
  docLoss(tokens: readonly number[], trace?: GptCallTrace[]): { tape: Tape; loss: Scal; n: number; perPosLoss: number[] } {
    const tape = new Tape()
    const n = Math.min(this.cfg.blockSize, tokens.length - 1)
    const kv = this.emptyKV()
    const losses: Scal[] = []
    for (let posId = 0; posId < n; posId++) {
      const logits = this.forward(tape, tokens[posId]!, posId, kv, trace)
      losses.push(crossEntropy(tape, logits, tokens[posId + 1]!))
    }
    const loss = meanScals(tape, losses)
    return { tape, loss, n, perPosLoss: losses.map((l) => l.data) }
  }

  /** Inference-only forward for a fixed token sequence → per-position logits. */
  forcedDecode(tokens: readonly number[]): number[][] {
    const tape = new Tape()
    const kv = this.emptyKV()
    const out: number[][] = []
    const steps = Math.min(tokens.length, this.cfg.blockSize)
    for (let posId = 0; posId < steps; posId++) {
      out.push([...this.forward(tape, tokens[posId]!, posId, kv).data])
    }
    return out
  }

  /**
   * Autoregressive sampling, exactly like the file's inference loop:
   * start at BOS, softmax(logits / temperature), weighted draw, stop on BOS.
   * Returns sampled token ids (without the surrounding BOS).
   */
  sample(rng: RNG, temperature = 0.5, bos = this.cfg.vocabSize - 1): number[] {
    const tape = new Tape()
    const kv = this.emptyKV()
    let tokenId = bos
    const out: number[] = []
    for (let posId = 0; posId < this.cfg.blockSize; posId++) {
      const logits = this.forward(tape, tokenId, posId, kv)
      const probs = softmaxProbs(logits.data, temperature)
      tokenId = rng.choiceWeighted(probs)
      if (tokenId === bos) break
      out.push(tokenId)
    }
    return out
  }
}

/** Adam with bias correction + linear LR decay; buffers in f64 for stability. */
export class Adam {
  readonly learningRate: number
  readonly beta1: number
  readonly beta2: number
  readonly eps: number
  readonly numSteps: number
  readonly m: Float64Array
  readonly v: Float64Array

  constructor(numParams: number, numSteps = 1000, learningRate = 0.01, beta1 = 0.85, beta2 = 0.99, eps = 1e-8) {
    this.learningRate = learningRate
    this.beta1 = beta1
    this.beta2 = beta2
    this.eps = eps
    this.numSteps = numSteps
    this.m = new Float64Array(numParams)
    this.v = new Float64Array(numParams)
  }

  /** One update over all mats (file param order); zeroes grads. Returns lr_t. */
  step(model: Model, step: number): number {
    const lrT = this.learningRate * (1 - step / this.numSteps)
    let i = 0
    for (const key of model.keys) {
      const mat = model.sd[key]!
      const { data, grad } = mat
      for (let j = 0; j < data.length; j++, i++) {
        const g = grad[j]!
        this.m[i] = this.beta1 * this.m[i]! + (1 - this.beta1) * g
        this.v[i] = this.beta2 * this.v[i]! + (1 - this.beta2) * g * g
        const mHat = this.m[i]! / (1 - this.beta1 ** (step + 1))
        const vHat = this.v[i]! / (1 - this.beta2 ** (step + 1))
        data[j] = data[j]! - (lrT * mHat) / (vHat ** 0.5 + this.eps)
        grad[j] = 0
      }
    }
    return lrT
  }
}

/** One full training step: forward doc, backward, Adam update. Returns loss. */
export function trainStep(model: Model, adam: Adam, tokens: readonly number[], step: number): number {
  const { tape, loss } = model.docLoss(tokens)
  tape.backward(loss)
  adam.step(model, step)
  return loss.data
}
