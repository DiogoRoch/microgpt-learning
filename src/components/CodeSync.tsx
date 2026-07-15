/**
 * Code-sync context: any interactive element can point the code panel at
 * specific lines of microgpt.py ("this animation step IS this line").
 */
import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

interface CodeSync {
  /** absolute 1-based file lines currently highlighted */
  highlight: readonly number[]
  setHighlight: (lines: readonly number[]) => void
  clear: () => void
}

const Ctx = createContext<CodeSync | null>(null)

export function CodeSyncProvider({ children }: { children: ReactNode }) {
  const [highlight, setHighlightState] = useState<readonly number[]>([])
  const setHighlight = useCallback((lines: readonly number[]) => setHighlightState(lines), [])
  const clear = useCallback(() => setHighlightState([]), [])
  const value = useMemo(() => ({ highlight, setHighlight, clear }), [highlight, setHighlight, clear])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useCodeSync(): CodeSync {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useCodeSync outside CodeSyncProvider')
  return ctx
}

/** Convenience: [start, end] inclusive → array of lines. */
export function lineRange(start: number, end: number): number[] {
  return Array.from({ length: end - start + 1 }, (_, i) => start + i)
}
