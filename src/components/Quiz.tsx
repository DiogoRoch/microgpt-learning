/**
 * Checkpoints — the interactive question system (reworked from the original
 * reveal-only PredictReveal, brief §4.3).
 *
 * Four kinds, one contract: commit an answer, get judged immediately, retry
 * until you get it right (wrong picks are eliminated and earn you a hint),
 * then read the explanation. Results persist per-question in localStorage;
 * a chapter auto-completes when every checkpoint in it is resolved.
 *
 *   PredictReveal — multiple choice: click an option to commit it
 *   NumericGuess  — type the number; tolerance-checked, too-high/low nudges
 *   TryIt         — do something in a live widget; verified from real state
 *   PickLine      — click the actual line of microgpt.py that does the thing
 */
import {
  createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode,
} from 'react'
import { useAppStore, type QuizResult } from '../app/store.ts'
import { SOURCE_LINES } from '../code/source.ts'
import { tokenizeSource } from '../code/highlight.ts'
import { TOKEN_COLORS } from './CodePanel.tsx'

interface QuizCtx {
  register: (qid: string) => void
  resolve: (qid: string, result: QuizResult) => void
  results: Record<string, QuizResult>
  registered: readonly string[]
}

const Ctx = createContext<QuizCtx | null>(null)

export function QuizProvider({ chapterId, children }: { chapterId: number; children: ReactNode }) {
  // registration order = mount order = document order (drives the meter dots)
  const [registered, setRegistered] = useState<readonly string[]>([])
  const results = useAppStore((s) => s.quiz)
  const resolveQuiz = useAppStore((s) => s.resolveQuiz)
  const markCompleted = useAppStore((s) => s.markCompleted)

  const register = useCallback((qid: string) => {
    setRegistered((list) => (list.includes(qid) ? list : [...list, qid]))
  }, [])
  const resolve = useCallback(
    (qid: string, result: QuizResult) => resolveQuiz(qid, result),
    [resolveQuiz],
  )

  useEffect(() => {
    if (registered.length > 0 && registered.every((q) => results[q])) {
      markCompleted(chapterId)
    }
  }, [registered, results, chapterId, markCompleted])

  const value = useMemo(
    () => ({ register, resolve, results, registered }),
    [register, resolve, results, registered],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

/** Register on mount; returns the persisted result (if any) and a resolver. */
function useCheckpoint(qid: string) {
  const quiz = useContext(Ctx)
  useEffect(() => {
    quiz?.register(qid)
  }, [quiz, qid])
  const result = quiz?.results[qid]
  const resolve = useCallback(
    (r: QuizResult) => quiz?.resolve(qid, r),
    [quiz, qid],
  )
  return { result, resolve }
}

const KIND_LABEL = { predict: 'predict', try: 'try it', compute: 'compute', line: 'find the line' } as const
type Kind = keyof typeof KIND_LABEL

/** Shared card chrome so every checkpoint reads as the same species. */
function Shell({
  kind, question, solved, revealed, children, footer,
}: {
  kind: Kind
  question: ReactNode
  solved: boolean
  revealed?: boolean
  children: ReactNode
  footer?: ReactNode
}) {
  return (
    <div
      className="not-prose my-6 rounded-lg border p-4 transition-colors"
      style={{
        borderColor: solved ? 'rgba(232,89,12,0.45)' : 'rgba(22,24,29,0.15)',
        background: solved ? 'rgba(232,89,12,0.04)' : 'rgba(255,255,255,0.6)',
      }}
    >
      <div className="mb-3 flex items-baseline gap-2">
        <span
          className="whitespace-nowrap rounded px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide"
          style={
            solved
              ? { background: 'var(--pos)', color: 'var(--paper)' }
              : kind === 'try'
                ? { background: 'rgba(255,201,77,0.35)', color: 'var(--ink)' }
                : { background: 'var(--ink)', color: 'var(--paper)' }
          }
        >
          {solved ? (revealed ? 'shown' : '✓ ' + KIND_LABEL[kind]) : KIND_LABEL[kind]}
        </span>
        <div className="font-medium">{question}</div>
      </div>
      {children}
      {footer}
    </div>
  )
}

function Explanation({ children }: { children: ReactNode }) {
  return <div className="mt-3 border-t border-ink/10 pt-3 text-[0.95rem]">{children}</div>
}

function Hint({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 rounded bg-[rgba(255,201,77,0.18)] px-3 py-2 text-sm">
      <span className="font-mono text-[10px] uppercase tracking-wide text-muted">hint · </span>
      {children}
    </p>
  )
}

// ---------------------------------------------------------------------------
// PredictReveal — multiple choice with instant judgement and retry
// ---------------------------------------------------------------------------

export interface PredictRevealProps {
  /** unique across the app, e.g. "ch5-no-mask" */
  qid: string
  question: ReactNode
  options: string[]
  /** index of the correct option */
  answerIndex: number
  /** shown once solved — explain from mechanism, not authority */
  explanation: ReactNode
  /** shown after the first wrong pick — a nudge, not the answer */
  hint?: ReactNode
}

export function PredictReveal({ qid, question, options, answerIndex, explanation, hint }: PredictRevealProps) {
  const { result, resolve } = useCheckpoint(qid)
  const [eliminated, setEliminated] = useState<ReadonlySet<number>>(new Set())
  const solved = result != null

  const pick = (i: number) => {
    if (solved || eliminated.has(i)) return
    if (i === answerIndex) {
      resolve({ status: 'correct', misses: eliminated.size })
    } else {
      setEliminated((s) => new Set(s).add(i))
    }
  }

  return (
    <Shell kind="predict" question={question} solved={solved}>
      <div className="flex flex-wrap gap-2" role="group" aria-label="commit an answer">
        {options.map((opt, i) => {
          const isCorrect = solved && i === answerIndex
          const isOut = eliminated.has(i)
          return (
            <button
              key={i}
              type="button"
              disabled={solved || isOut}
              onClick={() => pick(i)}
              aria-label={isOut ? `${opt} — wrong, eliminated` : opt}
              className="rounded border px-3 py-1.5 font-mono text-sm transition-colors focus-visible:outline-2 focus-visible:outline-[var(--hot)] disabled:cursor-default"
              style={{
                borderColor: isCorrect ? 'var(--pos)' : isOut ? 'var(--neg)' : 'rgba(22,24,29,0.2)',
                background: isCorrect ? 'rgba(232,89,12,0.1)' : 'transparent',
                textDecoration: isOut ? 'line-through' : 'none',
                opacity: solved && !isCorrect ? 0.45 : 1,
              }}
            >
              {opt}
              {isCorrect && ' ✓'}
            </button>
          )
        })}
      </div>
      {!solved && eliminated.size === 0 && (
        <p className="mt-2 font-mono text-xs text-muted">commit before you scroll — click your answer</p>
      )}
      {!solved && eliminated.size > 0 && (
        <>
          {hint ? <Hint>{hint}</Hint> : null}
          <p className="mt-2 font-mono text-xs" style={{ color: 'var(--neg)' }}>
            not that one — try again
          </p>
        </>
      )}
      {solved && (
        <>
          {result.misses > 0 && (
            <p className="mt-2 font-mono text-xs text-muted">
              got there in {result.misses + 1} tries
            </p>
          )}
          <Explanation>{explanation}</Explanation>
        </>
      )}
    </Shell>
  )
}

// ---------------------------------------------------------------------------
// NumericGuess — type the number, judged against an engine-computed answer
// ---------------------------------------------------------------------------

export interface NumericGuessProps {
  qid: string
  question: ReactNode
  /** the true value — computed, never hardcoded approximations of engine facts */
  answer: number
  /** |guess − answer| ≤ tolerance counts as correct */
  tolerance?: number
  /** e.g. "parameters", "≈ loss" — rendered after the input */
  unit?: string
  placeholder?: string
  explanation: ReactNode
  hint?: ReactNode
  /** how to print the answer once solved/shown */
  format?: (v: number) => string
}

export function NumericGuess({
  qid, question, answer, tolerance = 0, unit, placeholder, explanation, hint, format,
}: NumericGuessProps) {
  const { result, resolve } = useCheckpoint(qid)
  const [raw, setRaw] = useState('')
  const [misses, setMisses] = useState(0)
  const [verdict, setVerdict] = useState<'high' | 'low' | 'close' | null>(null)
  const solved = result != null
  const fmt = format ?? ((v: number) => String(v))

  const submit = () => {
    const guess = Number(raw.trim().replace(',', '.'))
    if (raw.trim() === '' || !Number.isFinite(guess)) return
    if (Math.abs(guess - answer) <= tolerance) {
      resolve({ status: 'correct', misses })
    } else {
      setMisses((m) => m + 1)
      const nearMiss = Math.abs(guess - answer) <= Math.max(tolerance * 4, Math.abs(answer) * 0.15)
      setVerdict(nearMiss ? 'close' : guess > answer ? 'high' : 'low')
    }
  }

  return (
    <Shell kind="compute" question={question} solved={solved} revealed={result?.status === 'revealed'}>
      {!solved ? (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={raw}
              onChange={(e) => setRaw(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submit()
              }}
              inputMode="decimal"
              placeholder={placeholder ?? 'your number'}
              aria-label="your numeric answer"
              className="w-36 rounded border border-ink/20 bg-white px-3 py-1.5 font-mono text-sm focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
            />
            {unit && <span className="font-mono text-xs text-muted">{unit}</span>}
            <button
              type="button"
              onClick={submit}
              className="rounded bg-ink px-3 py-1.5 font-mono text-xs text-paper hover:opacity-90 focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
            >
              check
            </button>
            {misses >= 3 && (
              <button
                type="button"
                onClick={() => resolve({ status: 'revealed', misses })}
                className="rounded border border-ink/25 px-3 py-1.5 font-mono text-xs hover:bg-ink/5"
              >
                show me
              </button>
            )}
          </div>
          {verdict && (
            <p className="mt-2 font-mono text-xs" style={{ color: 'var(--neg)' }}>
              {verdict === 'close' ? 'very close — tighten it up' : verdict === 'high' ? 'too high' : 'too low'}
            </p>
          )}
          {misses > 0 && hint ? <Hint>{hint}</Hint> : null}
        </>
      ) : (
        <>
          <p className="font-mono text-sm">
            {result.status === 'revealed' ? 'answer: ' : ''}
            <strong style={{ color: 'var(--pos)' }}>{fmt(answer)}</strong>
            {unit ? <span className="text-muted"> {unit}</span> : null}
            {result.status === 'correct' && result.misses > 0 && (
              <span className="text-muted"> · got there in {result.misses + 1} tries</span>
            )}
          </p>
          <Explanation>{explanation}</Explanation>
        </>
      )}
    </Shell>
  )
}

// ---------------------------------------------------------------------------
// TryIt — a hands-on task, verified live from the widget's real state
// ---------------------------------------------------------------------------

export interface TryItProps {
  qid: string
  /** the instruction, e.g. "scrub the stepper to position 0" */
  task: ReactNode
  /** live condition computed by the chapter from actual widget state */
  done: boolean
  /** shown once done — name what they just saw */
  payoff: ReactNode
}

export function TryIt({ qid, task, done, payoff }: TryItProps) {
  const { result, resolve } = useCheckpoint(qid)
  const solved = result != null
  // never un-solve: once the condition has been met it stays met
  const doneRef = useRef(false)
  useEffect(() => {
    if (done && !doneRef.current) {
      doneRef.current = true
      if (!solved) resolve({ status: 'correct', misses: 0 })
    }
  }, [done, solved, resolve])

  return (
    <Shell kind="try" question={task} solved={solved}>
      {solved ? (
        <Explanation>{payoff}</Explanation>
      ) : (
        <p className="font-mono text-xs text-muted">
          <span
            aria-hidden
            className="mr-1.5 inline-block h-2 w-2 animate-pulse rounded-full align-middle"
            style={{ background: 'var(--hot)' }}
          />
          watching the widget — this checks itself off the moment you do it
        </p>
      )}
    </Shell>
  )
}

// ---------------------------------------------------------------------------
// PickLine — click the line of microgpt.py that does the thing
// ---------------------------------------------------------------------------

// Tokenize once at module scope (same tokens the code panel renders).
const TOKENS = tokenizeSource(SOURCE_LINES)

export interface PickLineProps {
  qid: string
  question: ReactNode
  /** candidate 1-based line numbers to show, in order */
  lines: number[]
  /** the correct line */
  answer: number
  explanation: ReactNode
  hint?: ReactNode
}

export function PickLine({ qid, question, lines, answer, explanation, hint }: PickLineProps) {
  const { result, resolve } = useCheckpoint(qid)
  const [eliminated, setEliminated] = useState<ReadonlySet<number>>(new Set())
  const solved = result != null

  const pick = (line: number) => {
    if (solved || eliminated.has(line)) return
    if (line === answer) resolve({ status: 'correct', misses: eliminated.size })
    else setEliminated((s) => new Set(s).add(line))
  }

  return (
    <Shell kind="line" question={question} solved={solved}>
      <div className="overflow-x-auto rounded-lg py-2" style={{ background: 'var(--ink)' }}>
        <pre className="text-[12px] leading-[1.7]">
          <code className="block font-mono">
            {lines.map((lineNo, i) => {
              const isAnswer = solved && lineNo === answer
              const isOut = eliminated.has(lineNo)
              const gap = i > 0 && lines[i - 1]! + 1 !== lineNo
              return (
                <div key={lineNo}>
                  {gap && <div className="select-none px-4 text-white/25">⋯</div>}
                  <button
                    type="button"
                    disabled={solved || isOut}
                    onClick={() => pick(lineNo)}
                    aria-label={`line ${lineNo}: ${SOURCE_LINES[lineNo - 1] ?? ''}${isOut ? ' — wrong, eliminated' : ''}`}
                    className="flex w-full px-4 text-left focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-[var(--hot)] disabled:cursor-default"
                    style={{
                      background: isAnswer ? 'rgba(255,201,77,0.16)' : 'transparent',
                      boxShadow: isAnswer ? 'inset 3px 0 0 var(--hot)' : isOut ? 'inset 3px 0 0 var(--neg)' : undefined,
                      opacity: isOut ? 0.4 : 1,
                      cursor: solved || isOut ? 'default' : 'pointer',
                    }}
                  >
                    <span className="w-8 shrink-0 select-none text-right text-white/30" style={isAnswer ? { color: 'var(--hot)' } : undefined}>
                      {lineNo}
                    </span>
                    <span className="whitespace-pre pl-4" style={{ textDecoration: isOut ? 'line-through' : 'none' }}>
                      {TOKENS[lineNo - 1]!.map((t, ti) => (
                        <span key={ti} style={{ color: TOKEN_COLORS[t.kind] }}>
                          {t.text}
                        </span>
                      ))}
                    </span>
                  </button>
                </div>
              )
            })}
          </code>
        </pre>
      </div>
      {!solved && eliminated.size === 0 && (
        <p className="mt-2 font-mono text-xs text-muted">click the line</p>
      )}
      {!solved && eliminated.size > 0 && (
        <>
          {hint ? <Hint>{hint}</Hint> : null}
          <p className="mt-2 font-mono text-xs" style={{ color: 'var(--neg)' }}>
            not that line — look again
          </p>
        </>
      )}
      {solved && <Explanation>{explanation}</Explanation>}
    </Shell>
  )
}

// ---------------------------------------------------------------------------
// Progress chrome: the meter (chapter header) and the recap (chapter foot)
// ---------------------------------------------------------------------------

/** One dot per checkpoint, in document order. Rendered by ChapterFrame. */
export function CheckpointMeter() {
  const quiz = useContext(Ctx)
  if (!quiz || quiz.registered.length === 0) return null
  const resolved = quiz.registered.filter((q) => quiz.results[q]).length
  return (
    <div
      className="mt-3 flex items-center gap-2"
      role="img"
      aria-label={`${resolved} of ${quiz.registered.length} checkpoints resolved`}
    >
      <div className="flex gap-1">
        {quiz.registered.map((q) => {
          const r = quiz.results[q]
          return (
            <span
              key={q}
              className="h-2 w-2 rounded-full transition-colors"
              style={{
                background: r ? 'var(--pos)' : 'transparent',
                border: r ? 'none' : '1.5px solid rgba(22,24,29,0.3)',
              }}
            />
          )
        })}
      </div>
      <span className="font-mono text-[11px] text-muted">
        {resolved}/{quiz.registered.length} checkpoints
      </span>
    </div>
  )
}

export function Recap({ chapterId, points }: { chapterId: number; points: ReactNode[] }) {
  const quiz = useContext(Ctx)
  const completed = useAppStore((s) => s.completed.includes(chapterId))
  const markCompleted = useAppStore((s) => s.markCompleted)
  const total = quiz?.registered.length ?? 0
  const resolvedList = quiz ? quiz.registered.filter((q) => quiz.results[q]) : []
  const missing = total - resolvedList.length
  const firstTry = quiz
    ? resolvedList.filter((q) => {
        const r = quiz.results[q]!
        return r.status === 'correct' && r.misses === 0
      }).length
    : 0

  return (
    <section className="mt-12 rounded-lg border border-ink/15 p-5" aria-label="recap">
      <h2 className="font-display !mt-0 text-xl font-semibold">What you now know</h2>
      <ul className="mt-3 list-disc space-y-1.5 pl-5">
        {points.map((p, i) => (
          <li key={i}>{p}</li>
        ))}
      </ul>
      <div className="mt-4 border-t border-ink/10 pt-3 font-mono text-xs text-muted">
        {missing > 0 ? (
          `${missing} checkpoint${missing > 1 ? 's' : ''} above still open — clear ${missing > 1 ? 'them' : 'it'} to light this chapter up on the minimap`
        ) : completed ? (
          <span style={{ color: 'var(--pos)' }}>
            ✓ chapter complete
            {total > 0 && ` — ${firstTry}/${total} on the first try`}
            {' · the minimap remembers'}
          </span>
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
