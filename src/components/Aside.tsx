/**
 * Layered depth (brief §4.4): the main flow stays beginner-friendly; math and
 * "in the wild" bridges live in these clearly-marked, collapsed-by-default
 * panels. Nothing inside an Aside is required to follow the story.
 */
import type { ReactNode } from 'react'

const STYLES = {
  math: { label: 'the math', border: 'var(--neg)' },
  wild: { label: 'in the wild', border: 'var(--pos)' },
} as const

export function Aside({ kind, title, children }: { kind: keyof typeof STYLES; title: string; children: ReactNode }) {
  const s = STYLES[kind]
  return (
    <details className="group my-5 rounded-lg border bg-white/40" style={{ borderColor: `color-mix(in srgb, ${s.border} 35%, transparent)` }}>
      <summary className="flex cursor-pointer list-none items-baseline gap-2 px-4 py-2.5 focus-visible:outline-2 focus-visible:outline-[var(--hot)]">
        <span
          className="rounded px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-paper"
          style={{ background: s.border }}
        >
          {s.label}
        </span>
        <span className="font-medium">{title}</span>
        <span className="ml-auto font-mono text-xs text-muted group-open:hidden">+ expand</span>
        <span className="ml-auto hidden font-mono text-xs text-muted group-open:inline">− collapse</span>
      </summary>
      <div className="border-t border-ink/10 px-4 py-3 text-[0.95rem]">{children}</div>
    </details>
  )
}
