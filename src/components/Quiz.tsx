/**
 * PredictReveal — the "commit a guess, then see the real value" checkpoint
 * (brief §4.3). A chapter is auto-completed when every PredictReveal in it
 * has been revealed (plus a manual mark-as-done fallback in Recap).
 */
import {
  createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode,
} from 'react'
import { useAppStore } from '../app/store.ts'

interface QuizCtx {
  register: (qid: string) => void
  reveal: (qid: string) => void
  revealed: ReadonlySet<string>
  registered: ReadonlySet<string>
}

const Ctx = createContext<QuizCtx | null>(null)

export function QuizProvider({ chapterId, children }: { chapterId: number; children: ReactNode }) {
  const [registered, setRegistered] = useState<ReadonlySet<string>>(new Set())
  const [revealed, setRevealed] = useState<ReadonlySet<string>>(new Set())
  const markCompleted = useAppStore((s) => s.markCompleted)

  const register = useCallback((qid: string) => {
    setRegistered((s) => (s.has(qid) ? s : new Set(s).add(qid)))
  }, [])
  const reveal = useCallback((qid: string) => {
    setRevealed((s) => (s.has(qid) ? s : new Set(s).add(qid)))
  }, [])

  useEffect(() => {
    if (registered.size > 0 && [...registered].every((q) => revealed.has(q))) {
      markCompleted(chapterId)
    }
  }, [registered, revealed, chapterId, markCompleted])

  const value = useMemo(() => ({ register, reveal, revealed, registered }), [register, reveal, revealed, registered])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export interface PredictRevealProps {
  /** unique within the chapter */
  qid: string
  question: ReactNode
  /** multiple-choice options; the user must pick one before revealing */
  options: string[]
  /** index of the correct option */
  answerIndex: number
  /** shown after reveal — explain from mechanism, not authority */
  explanation: ReactNode
}

export function PredictReveal({ qid, question, options, answerIndex, explanation }: PredictRevealProps) {
  const quiz = useContext(Ctx)
  const [picked, setPicked] = useState<number | null>(null)
  const [shown, setShown] = useState(false)

  useEffect(() => {
    quiz?.register(qid)
  }, [quiz, qid])

  const revealNow = () => {
    setShown(true)
    quiz?.reveal(qid)
  }

  return (
    <div className="my-6 rounded-lg border border-ink/15 bg-white/60 p-4">
      <div className="mb-3 flex items-baseline gap-2">
        <span className="rounded bg-ink px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-paper">
          predict
        </span>
        <div className="font-medium">{question}</div>
      </div>
      <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="your prediction">
        {options.map((opt, i) => {
          const isCorrect = shown && i === answerIndex
          const isWrongPick = shown && picked === i && i !== answerIndex
          return (
            <button
              key={i}
              type="button"
              role="radio"
              aria-checked={picked === i}
              disabled={shown}
              onClick={() => setPicked(i)}
              className="rounded border px-3 py-1.5 font-mono text-sm transition-colors focus-visible:outline-2 focus-visible:outline-[var(--hot)] disabled:cursor-default"
              style={{
                borderColor: isCorrect ? 'var(--pos)' : isWrongPick ? 'var(--neg)' : picked === i ? 'var(--ink)' : 'rgba(22,24,29,0.2)',
                background: isCorrect ? 'rgba(232,89,12,0.1)' : picked === i && !shown ? 'rgba(22,24,29,0.06)' : 'transparent',
                textDecoration: isWrongPick ? 'line-through' : 'none',
              }}
            >
              {opt}
              {isCorrect && ' ✓'}
            </button>
          )
        })}
      </div>
      {!shown ? (
        <button
          type="button"
          onClick={revealNow}
          disabled={picked === null}
          className="mt-3 rounded bg-ink px-3 py-1.5 font-mono text-xs text-paper hover:opacity-90 focus-visible:outline-2 focus-visible:outline-[var(--hot)] disabled:opacity-30"
        >
          {picked === null ? 'pick first, then reveal' : 'reveal'}
        </button>
      ) : (
        <div className="mt-3 border-t border-ink/10 pt-3 text-[0.95rem]">{explanation}</div>
      )}
    </div>
  )
}

export function Recap({ chapterId, points }: { chapterId: number; points: ReactNode[] }) {
  const quiz = useContext(Ctx)
  const completed = useAppStore((s) => s.completed.includes(chapterId))
  const markCompleted = useAppStore((s) => s.markCompleted)
  const missing = quiz ? quiz.registered.size - [...quiz.registered].filter((q) => quiz.revealed.has(q)).length : 0

  return (
    <section className="mt-12 rounded-lg border border-ink/15 p-5" aria-label="recap">
      <h2 className="font-display !mt-0 text-xl font-semibold">What you now know</h2>
      <ul className="mt-3 list-disc space-y-1.5 pl-5">
        {points.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
      <div className="mt-4 border-t border-ink/10 pt-3 font-mono text-xs text-muted">
        {completed ? (
          <span style={{ color: 'var(--pos)' }}>✓ chapter complete — the minimap remembers</span>
        ) : missing > 0 ? (
          `answer the ${missing} remaining predict-and-reveal${missing > 1 ? 's' : ''} above to complete the chapter`
        ) : (
          <button
            type="button"
            className="underline underline-offset-2 hover:text-ink"
            onClick={() => markCompleted(chapterId)}
          >
            mark chapter as complete
          </button>
        )}
      </div>
    </section>
  )
}
