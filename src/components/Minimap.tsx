/**
 * The signature element: microgpt.py itself as navigation. All 200 lines run
 * down the edge; a chapter's lines light up when you complete it; the current
 * chapter pulses amber; clicking a region jumps to the chapter that teaches
 * it. Finishing the course = watching the whole file fill in.
 */
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { CHAPTERS, TOTAL_LINES, lineOwners } from '../app/chapters.ts'
import { useAppStore } from '../app/store.ts'
import { SOURCE_LINES } from '../code/source.ts'

export function Minimap({ currentChapter }: { currentChapter: number }) {
  const completed = useAppStore((s) => s.completed)
  const navigate = useNavigate()
  const [hoverLine, setHoverLine] = useState<number | null>(null)
  const owners = useMemo(() => lineOwners(), [])

  const H = 3 // px per line
  const W = 36
  const height = TOTAL_LINES * H

  const hoverChapterId = hoverLine != null ? owners[hoverLine]! : -1
  const hoverChapter = hoverChapterId >= 0 ? CHAPTERS[hoverChapterId]! : null

  return (
    <nav aria-label="microgpt.py minimap navigation" className="sticky top-6 hidden lg:block">
      <div className="mb-2 text-center font-mono text-[9px] leading-tight text-muted">
        microgpt
        <br />
        .py
      </div>
      <svg
        width={W}
        height={height}
        viewBox={`0 0 ${W} ${height}`}
        className="rounded"
        style={{ background: 'var(--ink)' }}
        onMouseLeave={() => setHoverLine(null)}
        role="img"
        aria-label={`minimap of microgpt.py — ${completed.length} of 12 chapters complete`}
      >
        {/* hover backdrop: lights up every line range owned by the hovered chapter, so
            the whole region that a click will jump to reads as one block, not just
            the single line under the cursor. */}
        {hoverChapter?.lines.map(([a, b], idx) => (
          <rect
            key={`hoverbg-${idx}`}
            x={0}
            y={(a - 1) * H}
            width={W}
            height={(b - a + 1) * H}
            fill="rgba(25,113,194,0.28)"
          />
        ))}
        {SOURCE_LINES.map((line, i) => {
          const lineNo = i + 1
          const owner = owners[lineNo]!
          const isDone = owner >= 0 && completed.includes(owner)
          const isCurrent = owner === currentChapter
          const isHovered = owner >= 0 && owner === hoverChapterId
          const indent = line.length - line.trimStart().length
          const len = Math.min(line.trim().length, 60)
          if (len === 0) return null
          let fill = 'rgba(255,255,255,0.22)'
          if (isDone) fill = 'var(--hot)'
          else if (isCurrent) fill = 'rgba(255,201,77,0.45)'
          if (isHovered && !isDone) fill = isCurrent ? 'rgba(255,201,77,0.8)' : 'rgba(255,255,255,0.6)'
          return (
            <rect
              key={lineNo}
              x={3 + indent * 0.35}
              y={i * H + 0.6}
              width={(len / 60) * (W - 8)}
              height={H - 1.2}
              rx={0.6}
              fill={fill}
              style={{ transition: 'fill 0.1s' }}
            />
          )
        })}
        {/* hover + click layer (mouse convenience only — the header chapter nav is
            the keyboard-accessible route to the same destinations) */}
        {SOURCE_LINES.map((_, i) => {
          const lineNo = i + 1
          return (
            <rect
              key={`hit-${lineNo}`}
              x={0}
              y={i * H}
              width={W}
              height={H}
              fill="transparent"
              style={{ cursor: owners[lineNo]! >= 0 ? 'pointer' : 'default' }}
              onMouseEnter={() => setHoverLine(lineNo)}
              onClick={() => {
                const owner = owners[lineNo]!
                if (owner >= 0) navigate(`/ch/${CHAPTERS[owner]!.slug}`)
              }}
            />
          )
        })}
      </svg>
      <div className="mt-2 h-10 w-24 -translate-x-6 text-center font-mono text-[10px] leading-tight text-muted">
        {hoverChapter ? (
          <>
            L{hoverLine}: {hoverChapter.short}
          </>
        ) : (
          `${completed.length}/12`
        )}
      </div>
    </nav>
  )
}
