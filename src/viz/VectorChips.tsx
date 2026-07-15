/**
 * A vector as a row of signed-color cells — the app's atom for "here is a
 * 16-dim activation". Hover any cell for index + exact value (provenance).
 */
import { absMax, fmt, signedColor } from './color.ts'

export interface VectorChipsProps {
  values: ArrayLike<number>
  /** name shown before the row and in tooltips, e.g. "tok_emb" */
  label?: string
  /** shared color scale max; defaults to this vector's own |max| */
  vmax?: number
  /** print the numbers inside cells (only sensible for short vectors) */
  showValues?: boolean
  /** cell width in px */
  cellSize?: number
  onHover?: (index: number | null) => void
  /** highlight one cell (e.g. synced with another view) */
  activeIndex?: number | null
}

export function VectorChips({
  values, label, vmax, showValues = false, cellSize = 26, onHover, activeIndex,
}: VectorChipsProps) {
  const n = values.length
  const max = vmax ?? absMax(values)
  const cells = []
  for (let i = 0; i < n; i++) {
    const v = values[i] as number
    cells.push(
      <div
        key={i}
        role="img"
        aria-label={`${label ?? 'value'}[${i}] = ${fmt(v)}`}
        title={`${label ?? ''}[${i}] = ${fmt(v)}`}
        onMouseEnter={onHover ? () => onHover(i) : undefined}
        onMouseLeave={onHover ? () => onHover(null) : undefined}
        className="flex items-center justify-center rounded-[3px] font-mono text-[10px] tabular-nums"
        style={{
          width: cellSize,
          height: cellSize,
          background: signedColor(v, max),
          outline: activeIndex === i ? '2px solid var(--hot)' : '1px solid rgba(22,24,29,0.08)',
          outlineOffset: -1,
          color: Math.abs(v) / (max || 1) > 0.55 ? 'var(--paper)' : 'var(--ink)',
        }}
      >
        {showValues ? fmt(v, 2) : ''}
      </div>,
    )
  }
  return (
    <div className="flex items-center gap-2">
      {label && <span className="w-24 shrink-0 text-right font-mono text-xs text-muted">{label}</span>}
      <div className="flex gap-[2px]">{cells}</div>
    </div>
  )
}
