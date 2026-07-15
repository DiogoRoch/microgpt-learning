/**
 * A weight matrix as a heatmap. Hover gives row/col meaning + exact value —
 * "which of the 4,192 parameters is this?" answered everywhere.
 */
import { useState } from 'react'
import { absMax, fmt, signedColor } from './color.ts'

export interface MatrixHeatmapProps {
  /** row-major values */
  data: ArrayLike<number>
  rows: number
  cols: number
  /** e.g. "wte" — used in tooltips */
  label: string
  /** meaning of a row/col, e.g. rowLabel(4) → "'e' (token 4)" */
  rowLabel?: (r: number) => string
  colLabel?: (c: number) => string
  cellSize?: number
  vmax?: number
  /** highlight one row (e.g. the current token's embedding row) */
  activeRow?: number | null
  onCellClick?: (r: number, c: number) => void
}

export function MatrixHeatmap({
  data, rows, cols, label, rowLabel, colLabel, cellSize = 10, vmax, activeRow, onCellClick,
}: MatrixHeatmapProps) {
  const [hover, setHover] = useState<[number, number] | null>(null)
  const max = vmax ?? absMax(data)
  const w = cols * cellSize
  const h = rows * cellSize
  const cells = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = data[r * cols + c] as number
      cells.push(
        <rect
          key={r * cols + c}
          x={c * cellSize}
          y={r * cellSize}
          width={cellSize - 0.5}
          height={cellSize - 0.5}
          fill={signedColor(v, max)}
          opacity={activeRow != null && activeRow !== r ? 0.25 : 1}
          onMouseEnter={() => setHover([r, c])}
          onMouseLeave={() => setHover(null)}
          onClick={onCellClick ? () => onCellClick(r, c) : undefined}
          style={{ cursor: onCellClick ? 'pointer' : 'default' }}
        />,
      )
    }
  }
  const hoverInfo = hover
    ? `${label}[${hover[0]}][${hover[1]}] = ${fmt(data[hover[0] * cols + hover[1]] as number)}` +
      (rowLabel ? ` — row: ${rowLabel(hover[0])}` : '') +
      (colLabel ? `, col: ${colLabel(hover[1])}` : '')
    : null
  return (
    <figure className="inline-block">
      <svg
        width={w}
        height={h}
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`${label}: ${rows}×${cols} weight matrix heatmap`}
      >
        {cells}
        {hover && (
          <rect
            x={hover[1] * cellSize - 0.5}
            y={hover[0] * cellSize - 0.5}
            width={cellSize}
            height={cellSize}
            fill="none"
            stroke="var(--hot)"
            strokeWidth={1.5}
            pointerEvents="none"
          />
        )}
      </svg>
      <figcaption className="mt-1 h-4 font-mono text-[11px] text-muted">
        {hoverInfo ?? `${label} · ${rows}×${cols}`}
      </figcaption>
    </figure>
  )
}
