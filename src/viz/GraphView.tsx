/**
 * GraphView — draws a real V computation graph (small expressions only; the
 * full model's graph has ~10⁵ nodes: run it, never draw it — brief §8).
 *
 * Forward values live in the nodes. Backward is shown as a step-through of
 * the actual algorithm: reversed topo order, child.grad += local_grad ·
 * v.grad, with the running grad totals drawn beneath each node. Gradients
 * are computed into a Map — the nodes themselves are never mutated.
 */
import { useMemo } from 'react'
import type { V } from '../engine/graph.ts'
import { fmt } from './color.ts'

const OP_LABEL: Record<string, string> = {
  const: '', add: '+', mul: '×', pow: '^', log: 'log', exp: 'exp', relu: 'relu',
}
/** the line of microgpt.py where each op stores its local grads */
export const OP_LINE: Record<string, number[]> = {
  add: [39, 40, 41], mul: [43, 44, 45], pow: [47], log: [48], exp: [49], relu: [50],
}

export interface BackwardPlan {
  /** captions[k] describes step k; step 0 is the seed */
  captions: string[]
  /** how many raw reversed-topo entries are processed after step k */
  rawProcessed: number[]
  /** node highlighted at step k */
  focus: (V | null)[]
}

/** Plan the meaningful backward steps (skipping leaf no-ops). */
export function planBackward(topo: V[], nameOf: (v: V) => string): BackwardPlan {
  const captions = ['seed: set the output node’s grad to 1 (∂out/∂out = 1)']
  const rawProcessed = [0]
  const focus: (V | null)[] = [topo[topo.length - 1] ?? null]
  for (let raw = 0; raw < topo.length; raw++) {
    const v = topo[topo.length - 1 - raw]!
    if (v.children.length === 0) continue
    const parts = v.children.map(
      (c, i) => `${nameOf(c)}.grad += ${fmt(v.localGrads[i]!, 3)} × ${nameOf(v)}.grad`,
    )
    captions.push(`process ${nameOf(v)}: ${parts.join(' ;  ')}`)
    rawProcessed.push(raw + 1)
    focus.push(v)
  }
  return { captions, rawProcessed, focus }
}

/** Grad state after processing `raw` entries of reversed topo (pure). */
export function partialGrads(topo: V[], raw: number): Map<V, number> {
  const g = new Map<V, number>()
  for (const v of topo) g.set(v, 0)
  const root = topo[topo.length - 1]
  if (root) g.set(root, 1)
  for (let s = 0; s < raw; s++) {
    const v = topo[topo.length - 1 - s]!
    for (let i = 0; i < v.children.length; i++) {
      const c = v.children[i]!
      g.set(c, g.get(c)! + v.localGrads[i]! * g.get(v)!)
    }
  }
  return g
}

export interface GraphViewProps {
  root: V
  /** names for leaves (variables); ops get their symbol */
  names?: Map<V, string>
  /** grads to display (from partialGrads); omit to hide grads entirely */
  grads?: Map<V, number> | null
  /** node to spotlight (current backward step) */
  focus?: V | null
  onHoverNode?: (v: V | null) => void
}

const NODE_W = 96
const NODE_H = 52
const COL_GAP = 60
const ROW_GAP = 18

export function GraphView({ root, names, grads, focus, onHoverNode }: GraphViewProps) {
  const layout = useMemo(() => {
    const topo = root.topo()
    const depth = new Map<V, number>()
    for (const v of topo) {
      depth.set(v, v.children.length === 0 ? 0 : 1 + Math.max(...v.children.map((c) => depth.get(c)!)))
    }
    const cols = new Map<number, V[]>()
    for (const v of topo) {
      const d = depth.get(v)!
      if (!cols.has(d)) cols.set(d, [])
      cols.get(d)!.push(v)
    }
    const pos = new Map<V, { x: number; y: number }>()
    const nCols = Math.max(...cols.keys()) + 1
    let maxRows = 0
    for (const [d, vs] of cols) {
      maxRows = Math.max(maxRows, vs.length)
      vs.forEach((v, i) => {
        pos.set(v, { x: d * (NODE_W + COL_GAP) + 8, y: i * (NODE_H + ROW_GAP) + 8 })
      })
    }
    // center shorter columns vertically
    const H = maxRows * (NODE_H + ROW_GAP)
    for (const [, vs] of cols) {
      const colH = vs.length * (NODE_H + ROW_GAP)
      const off = (H - colH) / 2
      for (const v of vs) pos.get(v)!.y += off
    }
    return { topo, pos, width: nCols * (NODE_W + COL_GAP) - COL_GAP + 16, height: H + 8 }
  }, [root])

  const nameOf = (v: V) => names?.get(v) ?? (v.op === 'const' ? fmt(v.data, 3) : OP_LABEL[v.op]!)

  return (
    <div className="overflow-x-auto">
      <svg
        width={layout.width}
        height={layout.height}
        viewBox={`0 0 ${layout.width} ${layout.height}`}
        role="img"
        aria-label="computation graph"
      >
        {/* edges */}
        {layout.topo.map((v) =>
          v.children.map((c, i) => {
            const a = layout.pos.get(c)!
            const b = layout.pos.get(v)!
            const x1 = a.x + NODE_W
            const y1 = a.y + NODE_H / 2
            const x2 = b.x
            const y2 = b.y + NODE_H / 2
            const isFocus = focus === v
            return (
              <g key={`${v.id}-${i}`}>
                <path
                  d={`M ${x1} ${y1} C ${x1 + COL_GAP / 2} ${y1}, ${x2 - COL_GAP / 2} ${y2}, ${x2} ${y2}`}
                  fill="none"
                  stroke={isFocus ? 'var(--hot)' : 'rgba(22,24,29,0.25)'}
                  strokeWidth={isFocus ? 2.5 : 1.5}
                />
                <text
                  x={(x1 + x2) / 2}
                  y={(y1 + y2) / 2 - 4}
                  textAnchor="middle"
                  fontSize={9}
                  className="font-mono"
                  fill={isFocus ? 'var(--ink)' : 'var(--muted)'}
                >
                  {fmt(v.localGrads[i]!, 2)}
                </text>
              </g>
            )
          }),
        )}
        {/* nodes */}
        {layout.topo.map((v) => {
          const p = layout.pos.get(v)!
          const isFocus = focus === v
          const g = grads?.get(v)
          const isLeafVar = v.children.length === 0 && names?.has(v)
          return (
            <g
              key={v.id}
              transform={`translate(${p.x},${p.y})`}
              onMouseEnter={onHoverNode ? () => onHoverNode(v) : undefined}
              onMouseLeave={onHoverNode ? () => onHoverNode(null) : undefined}
            >
              <rect
                width={NODE_W}
                height={NODE_H}
                rx={8}
                fill={isLeafVar ? 'rgba(25,113,194,0.08)' : 'white'}
                stroke={isFocus ? 'var(--hot)' : 'rgba(22,24,29,0.3)'}
                strokeWidth={isFocus ? 2.5 : 1}
              />
              <text x={8} y={16} fontSize={11} className="font-mono" fontWeight={600} fill="var(--ink)">
                {nameOf(v)}
              </text>
              <text x={NODE_W - 8} y={16} fontSize={10} textAnchor="end" className="font-mono" fill="var(--muted)">
                {v.op !== 'const' ? 'data' : ''}
              </text>
              <text x={8} y={32} fontSize={12} className="font-mono" fill="var(--ink)">
                {fmt(v.data, 3)}
              </text>
              {grads && (
                <text
                  x={8}
                  y={46}
                  fontSize={11}
                  className="font-mono"
                  fill={g !== 0 ? 'var(--pos)' : 'var(--muted)'}
                >
                  grad {fmt(g ?? 0, 3)}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
