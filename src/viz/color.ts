/**
 * Data-color conventions (PLAN.md): vermilion = positive, cerulean = negative,
 * everywhere a signed number is drawn. Attention weights (0..1) use a
 * sequential cerulean ramp. Intensity uses sqrt so small values stay visible.
 */

const POS = [232, 89, 12] as const // #E8590C
const NEG = [25, 113, 194] as const // #1971C2
const PAPER = [250, 249, 247] as const // #FAF9F7

function mix(from: readonly number[], to: readonly number[], t: number): string {
  const r = Math.round(from[0]! + (to[0]! - from[0]!) * t)
  const g = Math.round(from[1]! + (to[1]! - from[1]!) * t)
  const b = Math.round(from[2]! + (to[2]! - from[2]!) * t)
  return `rgb(${r},${g},${b})`
}

/** Signed value → fill color. vmax is the color-scale max (defaults sensible). */
export function signedColor(v: number, vmax: number): string {
  if (vmax <= 0 || v === 0 || Number.isNaN(v)) return mix(PAPER, [22, 24, 29], 0.04)
  const t = Math.sqrt(Math.min(1, Math.abs(v) / vmax))
  return mix(PAPER, v > 0 ? POS : NEG, t)
}

/** 0..1 → paper→cerulean ramp (attention weights, probabilities-as-cells). */
export function seqColor(t: number): string {
  return mix(PAPER, NEG, Math.sqrt(Math.max(0, Math.min(1, t))))
}

/** Max |v| of a vector/matrix, for shared color scales. */
export function absMax(xs: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < xs.length; i++) m = Math.max(m, Math.abs(xs[i] as number))
  return m
}

/** Format a number the way the app does everywhere: short but honest. */
export function fmt(v: number, digits = 4): string {
  if (v === 0) return '0'
  const a = Math.abs(v)
  if (a >= 1e4 || a < 1e-4) return v.toExponential(2)
  return v.toFixed(digits)
}
