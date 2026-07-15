import { describe, expect, it } from 'vitest'
import { parseExpr } from '../src/chapters/exprParser.ts'
import { partialGrads, planBackward } from '../src/viz/GraphView.tsx'

const grads = (src: string, values: Record<string, number>) => {
  const p = parseExpr(src, values)
  if ('error' in p) throw new Error(p.error)
  p.root.backward()
  return { p, g: Object.fromEntries([...p.vars].map(([k, v]) => [k, v.grad])), out: p.root.data }
}

describe('expression sandbox parser (builds real V graphs)', () => {
  it('computes the micrograd classic: d = a*b + c**2', () => {
    const { g, out } = grads('a*b + c**2', { a: 2, b: -3, c: 4 })
    expect(out).toBe(10)
    expect(g).toEqual({ a: -3, b: 2, c: 8 })
  })

  it('accumulates shared-node gradients: y = x*x + x', () => {
    const { g, out } = grads('x*x + x', { x: 3 })
    expect(out).toBe(12)
    expect(g['x']).toBe(7) // 2x + 1
  })

  it('mirrors Python desugaring: division builds a pow node', () => {
    const { g, out } = grads('a/b', { a: 1, b: 4 })
    expect(out).toBe(0.25)
    expect(g['a']).toBeCloseTo(0.25, 12)
    expect(g['b']).toBeCloseTo(-1 / 16, 12) // via b**-1 with local grad -1*b^-2
  })

  it('relu kills gradient for negative inputs', () => {
    const { g } = grads('relu(x) + x', { x: -2 })
    expect(g['x']).toBe(1) // only the direct path contributes
  })

  it('log/exp round trip', () => {
    const { g, out } = grads('log(exp(x))', { x: 1.5 })
    expect(out).toBeCloseTo(1.5, 12)
    expect(g['x']).toBeCloseTo(1, 12)
  })

  it('rejects what Value rejects', () => {
    expect(parseExpr('a ** b', { a: 2, b: 3 })).toHaveProperty('error')
    expect(parseExpr('log(0 - 1)', {})).toHaveProperty('error')
    expect(parseExpr('a +* b', {})).toHaveProperty('error')
  })

  it('partialGrads at full depth equals a real backward()', () => {
    const p = parseExpr('a*b + relu(c) - 1', { a: 2, b: -3, c: 4 })
    if ('error' in p) throw new Error(p.error)
    const topo = p.root.topo()
    const plan = planBackward(topo, () => 'n')
    const partial = partialGrads(topo, plan.rawProcessed[plan.rawProcessed.length - 1]!)
    p.root.backward()
    for (const [, v] of p.vars) {
      expect(partial.get(v)).toBeCloseTo(v.grad, 12)
    }
  })
})
