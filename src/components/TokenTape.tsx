/**
 * A token sequence as a tape of chips: char on top, id below, BOS as '·'.
 * Optionally draws the next-token training arrows (input → target).
 */
export interface TokenTapeProps {
  tokens: readonly number[]
  /** id → display char */
  labelOf: (id: number) => string
  /** show ids under the chars */
  showIds?: boolean
  /** draw arrows from each position to its target (tokens[i+1]) */
  showTargets?: boolean
  activeIndex?: number | null
  onHover?: (i: number | null) => void
}

export function TokenTape({ tokens, labelOf, showIds = true, showTargets = false, activeIndex, onHover }: TokenTapeProps) {
  return (
    <div className="flex flex-wrap items-end gap-1.5 py-1">
      {tokens.map((id, i) => {
        const isBos = labelOf(id) === '·'
        const active = activeIndex === i
        return (
          <div key={i} className="flex flex-col items-center gap-0.5">
            {showTargets && (
              <span className="h-4 font-mono text-[9px] text-muted">
                {i < tokens.length - 1 ? `→'${labelOf(tokens[i + 1]!)}'` : ''}
              </span>
            )}
            <div
              onMouseEnter={onHover ? () => onHover(i) : undefined}
              onMouseLeave={onHover ? () => onHover(null) : undefined}
              className="flex h-9 w-8 items-center justify-center rounded-md border font-mono text-base"
              style={{
                borderColor: active ? 'var(--hot)' : isBos ? 'var(--neg)' : 'rgba(22,24,29,0.25)',
                borderWidth: active ? 2 : 1,
                background: isBos ? 'rgba(25,113,194,0.08)' : 'white',
                color: isBos ? 'var(--neg)' : 'var(--ink)',
              }}
              title={isBos ? 'BOS (id 26)' : `'${labelOf(id)}' (id ${id})`}
            >
              {labelOf(id)}
            </div>
            {showIds && <span className="font-mono text-[10px] tabular-nums text-muted">{id}</span>}
          </div>
        )
      })}
    </div>
  )
}
