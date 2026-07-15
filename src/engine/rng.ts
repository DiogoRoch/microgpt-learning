/**
 * Seeded PRNG for reproducible in-browser runs (fresh weight inits, sampling).
 *
 * Deliberately NOT Python's Mersenne Twister: replicating CPython's RNG is a
 * trap (gauss caches a spare value, shuffle/choices consume the stream in
 * version-specific ways). Everything that must match the reference numerically
 * is loaded from golden/ dumps instead; this RNG only needs to be seeded and
 * stable across browsers.
 */

/** sfc32 — small fast counter PRNG, good quality, trivially seedable. */
export class RNG {
  private a: number
  private b: number
  private c: number
  private d: number
  private spareGauss: number | null = null

  constructor(seed: number) {
    // splitmix32 to spread one 32-bit seed into four state words
    let s = seed >>> 0
    const next = () => {
      s = (s + 0x9e3779b9) >>> 0
      let z = s
      z = Math.imul(z ^ (z >>> 16), 0x21f0aaad)
      z = Math.imul(z ^ (z >>> 15), 0x735a2d97)
      return (z ^ (z >>> 15)) >>> 0
    }
    this.a = next()
    this.b = next()
    this.c = next()
    this.d = next()
    for (let i = 0; i < 12; i++) this.random() // warm up
  }

  /** uniform in [0, 1) */
  random(): number {
    this.a >>>= 0
    this.b >>>= 0
    this.c >>>= 0
    this.d >>>= 0
    const t = (this.a + this.b) | 0
    this.a = this.b ^ (this.b >>> 9)
    this.b = (this.c + (this.c << 3)) | 0
    this.c = (this.c << 21) | (this.c >>> 11)
    this.d = (this.d + 1) | 0
    const r = (t + this.d) | 0
    this.c = (this.c + r) | 0
    return (r >>> 0) / 4294967296
  }

  /** gaussian via Box–Muller (polar form), with the spare cached like Python's gauss */
  gauss(mu = 0, sigma = 1): number {
    if (this.spareGauss !== null) {
      const z = this.spareGauss
      this.spareGauss = null
      return mu + sigma * z
    }
    let u = 0
    let v = 0
    let s = 0
    do {
      u = this.random() * 2 - 1
      v = this.random() * 2 - 1
      s = u * u + v * v
    } while (s >= 1 || s === 0)
    const f = Math.sqrt((-2 * Math.log(s)) / s)
    this.spareGauss = v * f
    return mu + sigma * (u * f)
  }

  /** weighted choice — the sampling step's random.choices(range(n), weights)[0] */
  choiceWeighted(weights: ArrayLike<number>): number {
    let total = 0
    for (let i = 0; i < weights.length; i++) total += weights[i] as number
    let r = this.random() * total
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i] as number
      if (r <= 0) return i
    }
    return weights.length - 1
  }
}
