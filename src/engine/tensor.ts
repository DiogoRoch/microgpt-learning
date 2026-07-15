/**
 * Vectorized engine: Float32Array storage, tape-based reverse-mode autograd
 * with a hand-written backward per op. Implements the SAME semantics as the
 * scalar graph engine (and therefore microgpt.py) — the derivatives here are
 * the closed forms of the gradients the scalar graph accumulates node by
 * node. A cross-engine test asserts agreement.
 *
 * Tape correctness invariant: every op creates a fresh output and never
 * mutates its inputs, so reverse creation order is a valid topological order
 * for backprop. Ops that read a GROWING list (the KV cache) snapshot it at
 * op time.
 *
 * Precision: storage is f32; arithmetic runs in f64 (plain JS numbers) and
 * rounds on store. Optimizer buffers live in f64 (see model.ts).
 */

/** A vector with value and gradient storage. */
export class Vec {
  data: Float32Array
  grad: Float32Array
  constructor(n: number) {
    this.data = new Float32Array(n)
    this.grad = new Float32Array(n)
  }
  static from(xs: ArrayLike<number>): Vec {
    const v = new Vec(xs.length)
    v.data.set(xs)
    return v
  }
  get length(): number {
    return this.data.length
  }
}

/** A scalar with value and gradient (loss nodes). */
export class Scal {
  data = 0
  grad = 0
}

/** A parameter matrix, row-major [nout × nin], same orientation as the file. */
export class Mat {
  readonly nout: number
  readonly nin: number
  data: Float32Array
  grad: Float32Array
  constructor(nout: number, nin: number) {
    this.nout = nout
    this.nin = nin
    this.data = new Float32Array(nout * nin)
    this.grad = new Float32Array(nout * nin)
  }
  static fromRows(rows: number[][]): Mat {
    const m = new Mat(rows.length, rows[0]?.length ?? 0)
    for (let o = 0; o < m.nout; o++) m.data.set(rows[o]!, o * m.nin)
    return m
  }
  row(o: number): Float32Array {
    return this.data.subarray(o * this.nin, (o + 1) * this.nin)
  }
  zeroGrad(): void {
    this.grad.fill(0)
  }
}

/** Records backward closures in forward order; backward() runs them reversed. */
export class Tape {
  private backs: Array<() => void> = []
  push(back: () => void): void {
    this.backs.push(back)
  }
  /** Seed d(seed)/d(seed) = 1 and run the whole tape backwards. */
  backward(seed: Scal): void {
    seed.grad = 1
    for (let i = this.backs.length - 1; i >= 0; i--) this.backs[i]!()
  }
  get size(): number {
    return this.backs.length
  }
}

/** Copy of one matrix row as a Vec; grads flow back into the matrix (embedding lookup). */
export function rowVec(tape: Tape, w: Mat, row: number): Vec {
  const out = new Vec(w.nin)
  out.data.set(w.row(row))
  tape.push(() => {
    const base = row * w.nin
    for (let i = 0; i < w.nin; i++) w.grad[base + i]! += out.grad[i]!
  })
  return out
}

/** Elementwise a + b (the residual add, the embedding sum). */
export function addVec(tape: Tape, a: Vec, b: Vec): Vec {
  const n = a.length
  const out = new Vec(n)
  for (let i = 0; i < n; i++) out.data[i] = a.data[i]! + b.data[i]!
  tape.push(() => {
    for (let i = 0; i < n; i++) {
      const g = out.grad[i]!
      a.grad[i]! += g
      b.grad[i]! += g
    }
  })
  return out
}

/** rmsnorm(x) = x · (mean(x²) + 1e-5)^-0.5, no learnable scale. */
export function rmsnorm(tape: Tape, x: Vec): Vec {
  const n = x.length
  let ms = 0
  for (let i = 0; i < n; i++) ms += x.data[i]! * x.data[i]!
  ms /= n
  const scale = (ms + 1e-5) ** -0.5
  const out = new Vec(n)
  for (let i = 0; i < n; i++) out.data[i] = x.data[i]! * scale
  tape.push(() => {
    // out_i = x_i·s with s = (ms+ε)^-1/2:
    //   dL/dx_i = s·g_i + (Σ_j g_j·x_j) · ds/dms · 2x_i/n,  ds/dms = -s³/2
    let gDotX = 0
    for (let j = 0; j < n; j++) gDotX += out.grad[j]! * x.data[j]!
    const dScale = -0.5 * scale ** 3
    const common = (gDotX * dScale * 2) / n
    for (let i = 0; i < n; i++) x.grad[i]! += scale * out.grad[i]! + common * x.data[i]!
  })
  return out
}

/** y = W·x  (microgpt's linear(x, w): y_o = Σ_i w[o,i]·x_i). */
export function linear(tape: Tape, w: Mat, x: Vec): Vec {
  const out = new Vec(w.nout)
  for (let o = 0; o < w.nout; o++) {
    const base = o * w.nin
    let acc = 0
    for (let i = 0; i < w.nin; i++) acc += w.data[base + i]! * x.data[i]!
    out.data[o] = acc
  }
  tape.push(() => {
    for (let o = 0; o < w.nout; o++) {
      const g = out.grad[o]!
      if (g === 0) continue
      const base = o * w.nin
      for (let i = 0; i < w.nin; i++) {
        w.grad[base + i]! += g * x.data[i]!
        x.grad[i]! += g * w.data[base + i]!
      }
    }
  })
  return out
}

/** Elementwise ReLU. */
export function relu(tape: Tape, x: Vec): Vec {
  const n = x.length
  const out = new Vec(n)
  for (let i = 0; i < n; i++) out.data[i] = Math.max(0, x.data[i]!)
  tape.push(() => {
    for (let i = 0; i < n; i++) if (x.data[i]! > 0) x.grad[i]! += out.grad[i]!
  })
  return out
}

/**
 * Per-head attention logits over the growing key list:
 *   logit_t = (Σ_j q[hs+j] · k_t[hs+j]) / √head_dim
 * Snapshots `keys` — the caller keeps appending to it at later positions.
 */
export function attnLogits(tape: Tape, q: Vec, keys: readonly Vec[], hs: number, headDim: number): Vec {
  const ks = keys.slice()
  const T = ks.length
  const invSqrt = 1 / headDim ** 0.5
  const out = new Vec(T)
  for (let t = 0; t < T; t++) {
    const k = ks[t]!
    let acc = 0
    for (let j = 0; j < headDim; j++) acc += q.data[hs + j]! * k.data[hs + j]!
    out.data[t] = acc * invSqrt
  }
  tape.push(() => {
    for (let t = 0; t < T; t++) {
      const g = out.grad[t]! * invSqrt
      if (g === 0) continue
      const k = ks[t]!
      for (let j = 0; j < headDim; j++) {
        q.grad[hs + j]! += g * k.data[hs + j]!
        k.grad[hs + j]! += g * q.data[hs + j]!
      }
    }
  })
  return out
}

/**
 * Numerically-stable softmax over a Vec (attention weights). Max subtraction
 * is a constant shift, exactly like the file. Backward is the closed-form
 * softmax jacobian: dL/dx_i = p_i (g_i - Σ_j g_j p_j).
 */
export function softmaxVec(tape: Tape, x: Vec): Vec {
  const n = x.length
  const out = new Vec(n)
  let max = -Infinity
  for (let i = 0; i < n; i++) max = Math.max(max, x.data[i]!)
  let total = 0
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x.data[i]! - max)
    out.data[i] = e
    total += e
  }
  for (let i = 0; i < n; i++) out.data[i] = out.data[i]! / total
  tape.push(() => {
    let gDotP = 0
    for (let j = 0; j < n; j++) gDotP += out.grad[j]! * out.data[j]!
    for (let i = 0; i < n; i++) x.grad[i]! += out.data[i]! * (out.grad[i]! - gDotP)
  })
  return out
}

/**
 * Per-head weighted sum of the growing value list:
 *   out_j = Σ_t w_t · v_t[hs+j]   (j = 0..headDim-1)
 */
export function weightedSum(tape: Tape, w: Vec, values: readonly Vec[], hs: number, headDim: number): Vec {
  const vs = values.slice()
  const T = vs.length
  const out = new Vec(headDim)
  for (let j = 0; j < headDim; j++) {
    let acc = 0
    for (let t = 0; t < T; t++) acc += w.data[t]! * vs[t]!.data[hs + j]!
    out.data[j] = acc
  }
  tape.push(() => {
    for (let t = 0; t < T; t++) {
      const v = vs[t]!
      let gw = 0
      for (let j = 0; j < headDim; j++) {
        const g = out.grad[j]!
        gw += g * v.data[hs + j]!
        v.grad[hs + j]! += g * w.data[t]!
      }
      w.grad[t]! += gw
    }
  })
  return out
}

/** Concatenate head outputs back into one n_embd vector. */
export function concatVecs(tape: Tape, parts: readonly Vec[]): Vec {
  let n = 0
  for (const p of parts) n += p.length
  const out = new Vec(n)
  let off = 0
  for (const p of parts) {
    out.data.set(p.data, off)
    off += p.length
  }
  tape.push(() => {
    let off2 = 0
    for (const p of parts) {
      for (let i = 0; i < p.length; i++) p.grad[i]! += out.grad[off2 + i]!
      off2 += p.length
    }
  })
  return out
}

/** Softmax probabilities WITHOUT tape (inference-only, e.g. sampling). */
export function softmaxProbs(logits: ArrayLike<number>, temperature = 1): Float64Array {
  const n = logits.length
  const out = new Float64Array(n)
  let max = -Infinity
  for (let i = 0; i < n; i++) max = Math.max(max, (logits[i] as number) / temperature)
  let total = 0
  for (let i = 0; i < n; i++) {
    const e = Math.exp((logits[i] as number) / temperature - max)
    out[i] = e
    total += e
  }
  for (let i = 0; i < n; i++) out[i] = out[i]! / total
  return out
}

/**
 * Cross-entropy of one position: loss = -log softmax(logits)[target],
 * computed stably as logsumexp(z) - z_target with z = logits - max.
 * Backward: dL/dlogit_i = (p_i - 1[i = target]) · g.
 */
export function crossEntropy(tape: Tape, logits: Vec, target: number): Scal {
  const n = logits.length
  let max = -Infinity
  for (let i = 0; i < n; i++) max = Math.max(max, logits.data[i]!)
  let total = 0
  const probs = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    const e = Math.exp(logits.data[i]! - max)
    probs[i] = e
    total += e
  }
  for (let i = 0; i < n; i++) probs[i] = probs[i]! / total
  const out = new Scal()
  out.data = Math.log(total) - (logits.data[target]! - max)
  tape.push(() => {
    const g = out.grad
    if (g === 0) return
    for (let i = 0; i < n; i++) logits.grad[i]! += (probs[i]! - (i === target ? 1 : 0)) * g
  })
  return out
}

/** Mean of scalar losses: (1/n) · Σ loss_t. */
export function meanScals(tape: Tape, xs: readonly Scal[]): Scal {
  const n = xs.length
  const out = new Scal()
  let acc = 0
  for (const x of xs) acc += x.data
  out.data = acc / n
  tape.push(() => {
    for (const x of xs) x.grad += out.grad / n
  })
  return out
}
