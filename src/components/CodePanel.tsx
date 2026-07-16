/**
 * The persistent code panel: real lines of microgpt.py, custom-highlighted,
 * with a line-highlight API (via CodeSync context or the `highlight` prop).
 * Auto-scrolls to keep the highlighted line visible. Always links the gist.
 */
import { useEffect, useMemo, useRef } from 'react'
import { GIST_URL, SOURCE_LINES } from '../code/source.ts'
import { tokenizeSource, type Tok } from '../code/highlight.ts'
import { useCodeSync } from './CodeSync.tsx'

export const TOKEN_COLORS: Record<Tok['kind'], string> = {
  kw: '#82B4E8',
  builtin: '#6FCDBD',
  str: '#E8C580',
  num: '#EE9B70',
  comment: '#7A808C',
  def: '#FFC94D',
  op: '#AEB3BD',
  plain: '#E6E4E1',
}

// Tokenize the whole file once at module scope.
const ALL_TOKENS = tokenizeSource(SOURCE_LINES)

export interface CodePanelProps {
  /** inclusive line ranges to DISPLAY, e.g. a chapter's ranges; default whole file */
  ranges?: Array<[number, number]>
  /** lines to highlight; overrides CodeSync context when provided */
  highlight?: readonly number[]
  /** compact = smaller text, tighter leading (for inline embeds) */
  compact?: boolean
  title?: string
  maxHeight?: number | string
}

export function CodePanel({ ranges, highlight, compact = false, title, maxHeight }: CodePanelProps) {
  const sync = useCodeSync()
  const active = highlight ?? sync.highlight
  const activeSet = useMemo(() => new Set(active), [active])
  const scrollRef = useRef<HTMLDivElement>(null)
  const firstActive = active.length > 0 ? Math.min(...active) : null

  useEffect(() => {
    if (firstActive == null || !scrollRef.current) return
    const el = scrollRef.current.querySelector<HTMLElement>(`[data-line="${firstActive}"]`)
    el?.scrollIntoView({ block: 'nearest', behavior: 'auto' })
  }, [firstActive])

  const shown = ranges ?? [[1, SOURCE_LINES.length] as [number, number]]

  return (
    <div className="overflow-hidden rounded-lg" style={{ background: 'var(--ink)' }}>
      <div className="flex items-baseline justify-between border-b border-white/10 px-4 py-2">
        <span className="font-mono text-xs text-white/70">
          {title ?? 'microgpt.py'}
          {ranges && (
            <span className="text-white/40">
              {' '}· lines {ranges.map(([a, b]) => (a === b ? a : `${a}–${b}`)).join(', ')}
            </span>
          )}
        </span>
        <a
          href={GIST_URL}
          target="_blank"
          rel="noreferrer"
          className="font-mono text-[11px] text-white/60 underline-offset-2 hover:text-white/90 hover:underline focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
        >
          @karpathy&apos;s gist ↗
        </a>
      </div>
      <div ref={scrollRef} className="overflow-auto px-0 py-2" style={{ maxHeight: maxHeight ?? '70vh' }}>
        <pre className={compact ? 'text-[11px] leading-[1.5]' : 'text-[12.5px] leading-[1.65]'}>
          <code className="block font-mono">
            {shown.map(([a, b], ri) => (
              <div key={ri}>
                {ri > 0 && <div className="select-none px-4 text-white/25">⋯</div>}
                {SOURCE_LINES.slice(a - 1, b).map((_, i) => {
                  const lineNo = a + i
                  const isActive = activeSet.has(lineNo)
                  return (
                    <div
                      key={lineNo}
                      data-line={lineNo}
                      className="flex px-4"
                      style={
                        isActive
                          ? { background: 'rgba(255,201,77,0.13)', boxShadow: 'inset 3px 0 0 var(--hot)' }
                          : undefined
                      }
                    >
                      <span className="w-8 shrink-0 select-none text-right text-white/30" style={isActive ? { color: 'var(--hot)' } : undefined}>
                        {lineNo}
                      </span>
                      <span className="whitespace-pre pl-4">
                        {ALL_TOKENS[lineNo - 1]!.map((t, ti) => (
                          <span key={ti} style={{ color: TOKEN_COLORS[t.kind] }}>
                            {t.text}
                          </span>
                        ))}
                        {SOURCE_LINES[lineNo - 1] === '' ? ' ' : ''}
                      </span>
                    </div>
                  )
                })}
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  )
}
