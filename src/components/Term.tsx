/**
 * Inline glossary term: dotted underline, definition on hover/focus/tap.
 * Keyboard accessible (focusable, Escape closes).
 */
import { useId, useState } from 'react'
import { GLOSSARY } from '../app/glossary.ts'

export function Term({ t, children }: { t: string; children?: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const def = GLOSSARY[t]
  if (!def) return <>{children ?? t}</>
  return (
    <span className="relative inline-block">
      <button
        type="button"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
        onClick={() => setOpen((o) => !o)}
        className="cursor-help border-b border-dotted border-ink/50 focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
      >
        {children ?? t}
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute left-1/2 top-full z-40 mt-1 w-72 -translate-x-1/2 rounded-lg border border-ink/15 bg-paper p-3 text-sm leading-snug shadow-lg"
        >
          <span className="font-mono text-xs font-semibold">{t}</span>
          <br />
          {def}
        </span>
      )}
    </span>
  )
}
