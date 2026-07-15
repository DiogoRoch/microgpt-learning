import { expect } from 'vitest'

/** |a - e| <= atol + rtol·|e|, elementwise, with a useful failure message. */
export function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  { atol, rtol, label }: { atol: number; rtol: number; label: string },
): void {
  expect(actual.length, `${label}: length`).toBe(expected.length)
  let worstIdx = -1
  let worstExcess = 0
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i] as number
    const e = expected[i] as number
    const excess = Math.abs(a - e) - (atol + rtol * Math.abs(e))
    if (excess > worstExcess) {
      worstExcess = excess
      worstIdx = i
    }
  }
  if (worstIdx >= 0) {
    const a = actual[worstIdx] as number
    const e = expected[worstIdx] as number
    expect.fail(
      `${label}: [${worstIdx}] actual=${a} expected=${e} |Δ|=${Math.abs(a - e)} ` +
        `exceeds atol=${atol} + rtol=${rtol}·|expected|`,
    )
  }
}

/** Max absolute and relative deviation — for measuring/documenting tolerances. */
export function maxDeviation(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
): { maxAbs: number; maxRel: number } {
  let maxAbs = 0
  let maxRel = 0
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i] as number
    const e = expected[i] as number
    const abs = Math.abs(a - e)
    maxAbs = Math.max(maxAbs, abs)
    if (Math.abs(e) > 1e-12) maxRel = Math.max(maxRel, abs / Math.abs(e))
  }
  return { maxAbs, maxRel }
}
