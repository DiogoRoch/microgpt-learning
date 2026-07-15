/**
 * The growing lower-triangular attention view: row = query position, columns
 * = the key positions that EXISTED at that moment. The upper triangle isn't
 * masked out — it simply never existed. That absence is the whole lesson.
 */
import { useState } from 'react'
import { fmt, seqColor } from './color.ts'

export interface AttnMatrixProps {
  /** weights[pos] has pos+1 entries (softmax over existing keys) */
  weights: ReadonlyArray<ArrayLike<number>>
  /** token label per position (chars; BOS = '·') */
  tokens: readonly string[]
  cellSize?: number
  /** only draw rows ≤ this position (for stepping); default all */
  uptoRow?: number
  /** row highlighted as "current" */
  activeRow?: number | null
  showValues?: boolean
}

export function AttnMatrix({ weights, tokens, cellSize = 30, uptoRow, activeRow, showValues = false }: AttnMatrixProps) {
  const [hover, setHover] = useState<[number, number] | null>(null)
  const T = uptoRow != null ? Math.min(uptoRow + 1, weights.length) : weights.length
  const n = weights.length
  const pad = 22
  const w = pad + n * cellSize
  const h = pad + n * cellSize
  const cells = []
  for (let r = 0; r < T; r++) {
    const row = weights[r]!
    for (let c = 0; c <= r; c++) {
      const v = row[c] as number
      cells.push(
        <g key={`${r}-${c}`}>
          <rect
            x={pad + c * cellSize}
            y={pad + r * cellSize}
            width={cellSize - 1}
            height={cellSize - 1}
            rx={2}
            fill={seqColor(v)}
            opacity={activeRow != null && activeRow !== r ? 0.35 : 1}
            onMouseEnter={() => setHover([r, c])}
            onMouseLeave={() => setHover(null)}
          />
          {showValues && (
            <text
              x={pad + c * cellSize + (cellSize - 1) / 2}
              y={pad + r * cellSize + (cellSize - 1) / 2 + 3}
              textAnchor="middle"
              className="pointer-events-none font-mono"
              fontSize={9}
              fill={v > 0.45 ? 'var(--paper)' : 'var(--ink)'}
            >
              {v >= 0.995 ? '1' : (v as number).toFixed(2).replace(/^0/, '')}
            </text>
          )}
        </g>,
      )
    }
  }
  const labels = []
  for (let i = 0; i < n; i++) {
    labels.push(
      <text
        key={`c${i}`}
        x={pad + i * cellSize + cellSize / 2}
        y={pad - 7}
        textAnchor="middle"
        fontSize={11}
        className="font-mono"
        fill={hover && hover[1] === i ? 'var(--ink)' : 'var(--muted)'}
      >
        {tokens[i]}
      </text>,
      <text
        key={`r${i}`}
        x={pad - 8}
        y={pad + i * cellSize + cellSize / 2 + 4}
        textAnchor="middle"
        fontSize={11}
        className="font-mono"
        fill={i === activeRow ? 'var(--ink)' : hover && hover[0] === i ? 'var(--ink)' : 'var(--muted)'}
      >
        {i < T ? tokens[i] : ''}
      </text>,
    )
  }
  return (
    <figure className="inline-block">
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="img" aria-label="attention weights, growing lower-triangular">
        {labels}
        {cells}
      </svg>
      <figcaption className="mt-1 h-4 font-mono text-[11px] text-muted">
        {hover
          ? `attn_weights: query '${tokens[hover[0]]}' (pos ${hover[0]}) → key '${tokens[hover[1]]}' (pos ${hover[1]}) = ${fmt(weights[hover[0]]![hover[1]] as number)}`
          : 'row = query position · columns = keys that existed at that moment'}
      </figcaption>
    </figure>
  )
}
