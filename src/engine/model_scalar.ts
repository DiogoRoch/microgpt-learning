/**
 * The full microgpt model on the scalar graph engine — a line-for-line mirror
 * of microgpt.py's `linear` / `softmax` / `rmsnorm` / `gpt` / training-loop
 * math, building real computation graphs of V nodes.
 *
 * This is the pedagogical engine (chapter 2's live graphs, worked examples)
 * and the f64 parity reference. It is ~1000× too slow to train the full model
 * interactively — that's tensor.ts's job. Both are parity-tested against
 * golden/ and against each other.
 */

import { V, vsum } from './graph.ts'
import type { DocTrace, GptCallTrace, LayerTrace } from './trace.ts'

export interface Config {
  nLayer: number
  nEmbd: number
  blockSize: number
  nHead: number
  headDim: number
  vocabSize: number
}

/** The file's hyperparameters with the names-dataset vocab. */
export const MICRO_CONFIG: Config = {
  nLayer: 1,
  nEmbd: 16,
  blockSize: 16,
  nHead: 4,
  headDim: 4, // n_embd // n_head
  vocabSize: 27,
}

/** state_dict key order — identical to the Python insertion order. */
export function stateDictKeys(nLayer: number): string[] {
  const keys = ['wte', 'wpe', 'lm_head']
  for (let i = 0; i < nLayer; i++) {
    keys.push(
      `layer${i}.attn_wq`,
      `layer${i}.attn_wk`,
      `layer${i}.attn_wv`,
      `layer${i}.attn_wo`,
      `layer${i}.mlp_fc1`,
      `layer${i}.mlp_fc2`,
    )
  }
  return keys
}

export type ScalarStateDict = Record<string, V[][]>

/** Load weights (e.g. golden init_weights.json) into V matrices. */
export function loadScalarStateDict(weights: Record<string, number[][]>, cfg: Config): ScalarStateDict {
  const sd: ScalarStateDict = {}
  for (const key of stateDictKeys(cfg.nLayer)) {
    const mat = weights[key]
    if (!mat) throw new Error(`missing state_dict key ${key}`)
    sd[key] = mat.map((row) => row.map((x) => new V(x)))
  }
  return sd
}

/** Flatten params exactly like `[p for mat in state_dict.values() for row in mat for p in row]`. */
export function flattenParams(sd: ScalarStateDict, cfg: Config): V[] {
  const params: V[] = []
  for (const key of stateDictKeys(cfg.nLayer)) {
    for (const row of sd[key]!) for (const p of row) params.push(p)
  }
  return params
}

/** linear(x, w): [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w] */
export function linear(x: readonly V[], w: readonly V[][]): V[] {
  return w.map((wo) => vsum(wo.map((wi, i) => wi.mul(x[i]!))))
}

/** Numerically-stable softmax; the max is subtracted as a RAW FLOAT (no grad). */
export function softmax(logits: readonly V[]): V[] {
  const maxVal = Math.max(...logits.map((v) => v.data))
  const exps = logits.map((v) => v.sub(maxVal).exp())
  const total = vsum(exps)
  return exps.map((e) => e.div(total))
}

/** rmsnorm without learnable scale: x · (mean(x²) + 1e-5)^-0.5 */
export function rmsnorm(x: readonly V[]): V[] {
  const ms = vsum(x.map((xi) => xi.mul(xi))).div(x.length)
  const scale = ms.add(1e-5).pow(-0.5)
  return x.map((xi) => xi.mul(scale))
}

/** keys[layer][t] and values[layer][t] — the growing KV cache. */
export interface ScalarKV {
  keys: V[][][]
  values: V[][][]
}

export function emptyKV(cfg: Config): ScalarKV {
  return {
    keys: Array.from({ length: cfg.nLayer }, () => []),
    values: Array.from({ length: cfg.nLayer }, () => []),
  }
}

const data = (xs: readonly V[]): number[] => xs.map((x) => x.data)

/**
 * One incremental forward pass — microgpt.py's `gpt(token_id, pos_id, keys, values)`.
 * Appends this position's k and v to the cache; attention sees only the
 * positions that already exist (causality with no mask).
 */
export function gpt(
  sd: ScalarStateDict,
  cfg: Config,
  tokenId: number,
  posId: number,
  kv: ScalarKV,
  trace?: GptCallTrace[],
): V[] {
  const tokEmb = sd['wte']![tokenId]!
  const posEmb = sd['wpe']![posId]!
  let x = tokEmb.map((t, i) => t.add(posEmb[i]!))
  const xEmbSum = x
  x = rmsnorm(x) // not redundant: the residual stream bypasses it in backward
  let rec: GptCallTrace | undefined
  if (trace) {
    rec = {
      token_id: tokenId,
      pos_id: posId,
      tok_emb: data(tokEmb),
      pos_emb: data(posEmb),
      x_emb_sum: data(xEmbSum),
      x_emb_norm: data(x),
      layers: [],
      logits: [],
    }
    trace.push(rec)
  }

  for (let li = 0; li < cfg.nLayer; li++) {
    // 1) Multi-head Attention block
    let xResidual = x
    x = rmsnorm(x)
    const q = linear(x, sd[`layer${li}.attn_wq`]!)
    const k = linear(x, sd[`layer${li}.attn_wk`]!)
    const v = linear(x, sd[`layer${li}.attn_wv`]!)
    kv.keys[li]!.push(k)
    kv.values[li]!.push(v)
    let lrec: LayerTrace | undefined
    if (rec) {
      lrec = {
        x_ln_attn: data(x), q: data(q), k: data(k), v: data(v), heads: [],
        x_attn: [], x_wo: [], x_after_attn: [], x_ln_mlp: [], fc1: [], relu: [], fc2: [], x_after_mlp: [],
      }
      rec.layers.push(lrec)
    }
    const xAttn: V[] = []
    for (let h = 0; h < cfg.nHead; h++) {
      const hs = h * cfg.headDim
      const qH = q.slice(hs, hs + cfg.headDim)
      const kH = kv.keys[li]!.map((ki) => ki.slice(hs, hs + cfg.headDim))
      const vH = kv.values[li]!.map((vi) => vi.slice(hs, hs + cfg.headDim))
      const attnLogits = kH.map((kt) => vsum(qH.map((qj, j) => qj.mul(kt[j]!))).div(cfg.headDim ** 0.5))
      const attnWeights = softmax(attnLogits)
      const headOut = Array.from({ length: cfg.headDim }, (_, j) =>
        vsum(attnWeights.map((w, t) => w.mul(vH[t]![j]!))),
      )
      xAttn.push(...headOut)
      lrec?.heads.push({ attn_logits: data(attnLogits), attn_weights: data(attnWeights), head_out: data(headOut) })
    }
    x = linear(xAttn, sd[`layer${li}.attn_wo`]!)
    if (lrec) {
      lrec.x_attn = data(xAttn)
      lrec.x_wo = data(x)
    }
    x = x.map((a, i) => a.add(xResidual[i]!))
    if (lrec) lrec.x_after_attn = data(x)
    // 2) MLP block
    xResidual = x
    x = rmsnorm(x)
    if (lrec) lrec.x_ln_mlp = data(x)
    x = linear(x, sd[`layer${li}.mlp_fc1`]!)
    if (lrec) lrec.fc1 = data(x)
    x = x.map((xi) => xi.relu())
    if (lrec) lrec.relu = data(x)
    x = linear(x, sd[`layer${li}.mlp_fc2`]!)
    if (lrec) lrec.fc2 = data(x)
    x = x.map((a, i) => a.add(xResidual[i]!))
    if (lrec) lrec.x_after_mlp = data(x)
  }

  const logits = linear(x, sd['lm_head']!)
  if (rec) rec.logits = data(logits)
  return logits
}

export interface ScalarDocLoss {
  loss: V
  perPosition: { tokenId: number; targetId: number; probs: V[]; lossT: V }[]
  n: number
}

/**
 * Forward a whole document and build the average next-token loss, exactly as
 * the training loop does: n = min(block_size, len(tokens)-1), per-position
 * -log(probs[target]), then (1/n) * sum.
 */
export function docLoss(sd: ScalarStateDict, cfg: Config, tokens: readonly number[], trace?: DocTrace): ScalarDocLoss {
  const n = Math.min(cfg.blockSize, tokens.length - 1)
  const kv = emptyKV(cfg)
  const perPosition: ScalarDocLoss['perPosition'] = []
  const losses: V[] = []
  for (let posId = 0; posId < n; posId++) {
    const tokenId = tokens[posId]!
    const targetId = tokens[posId + 1]!
    const logits = gpt(sd, cfg, tokenId, posId, kv, trace?.gpt_calls)
    const probs = softmax(logits)
    const lossT = probs[targetId]!.log().neg()
    losses.push(lossT)
    perPosition.push({ tokenId, targetId, probs, lossT })
    trace?.positions.push({ token_id: tokenId, target_id: targetId, probs: data(probs), loss_t: lossT.data })
  }
  const loss = vsum(losses).mul(1 / n) // Python: (1/n) * sum(losses) via __rmul__
  if (trace) {
    trace.tokens = [...tokens]
    trace.n = n
    trace.loss = loss.data
  }
  return { loss, perPosition, n }
}

/** Adam hyperparameters and buffers, exactly as in the file. */
export interface AdamState {
  learningRate: number
  beta1: number
  beta2: number
  eps: number
  numSteps: number
  m: Float64Array
  v: Float64Array
}

export function initAdam(numParams: number, numSteps = 1000): AdamState {
  return {
    learningRate: 0.01,
    beta1: 0.85,
    beta2: 0.99,
    eps: 1e-8,
    numSteps,
    m: new Float64Array(numParams),
    v: new Float64Array(numParams),
  }
}

/** One Adam update with bias correction and linear LR decay; zeroes grads. */
export function adamStep(params: readonly V[], adam: AdamState, step: number): number {
  const lrT = adam.learningRate * (1 - step / adam.numSteps)
  for (let i = 0; i < params.length; i++) {
    const p = params[i]!
    adam.m[i] = adam.beta1 * adam.m[i]! + (1 - adam.beta1) * p.grad
    adam.v[i] = adam.beta2 * adam.v[i]! + (1 - adam.beta2) * p.grad ** 2
    const mHat = adam.m[i]! / (1 - adam.beta1 ** (step + 1))
    const vHat = adam.v[i]! / (1 - adam.beta2 ** (step + 1))
    p.data -= (lrT * mHat) / (vHat ** 0.5 + adam.eps)
    p.grad = 0
  }
  return lrT
}
