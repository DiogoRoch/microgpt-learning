/**
 * The shared chapter chrome: narrative column + sticky code panel + prev/next
 * navigation, wrapped in a fresh CodeSync scope per chapter.
 */
import { type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { CHAPTERS, type Chapter } from '../app/chapters.ts'
import { useAppStore } from '../app/store.ts'
import { CodePanel } from './CodePanel.tsx'
import { CodeSyncProvider } from './CodeSync.tsx'
import { Minimap } from './Minimap.tsx'
import { CheckpointMeter, QuizProvider } from './Quiz.tsx'

export function ChapterFrame({
  chapter, children, hideCodePanel = false, fullFileCode = false,
}: {
  chapter: Chapter
  children: ReactNode
  /** chapters that embed code panels inline (e.g. playground) can hide the side panel */
  hideCodePanel?: boolean
  /** show the whole file in the side panel instead of just this chapter's lines
   *  (the big-picture page hovers stages across the entire file) */
  fullFileCode?: boolean
}) {
  const prev = CHAPTERS[chapter.id - 1]
  const next = CHAPTERS[chapter.id + 1]
  const completed = useAppStore((s) => s.completed)
  const codeRanges = fullFileCode ? undefined : chapter.lines.length ? chapter.lines : undefined

  return (
    <CodeSyncProvider key={chapter.id}>
      <QuizProvider chapterId={chapter.id}>
      <div className="mx-auto flex max-w-[1500px] gap-6 px-4 py-6 md:px-8">
        <Minimap currentChapter={chapter.id} />
        <div className="min-w-0 flex-1">
          <header className="mb-8">
            <div className="font-mono text-xs text-muted">
              chapter {chapter.id} / 11
              {completed.includes(chapter.id) && <span style={{ color: 'var(--pos)' }}> ✓ completed</span>}
            </div>
            <h1 className="font-display mt-1 text-4xl font-semibold tracking-tight">{chapter.title}</h1>
            <p className="mt-2 max-w-xl text-lg text-muted">{chapter.subtitle}</p>
            <CheckpointMeter />
          </header>

          <div
            className={
              hideCodePanel
                ? 'max-w-5xl'
                : 'grid grid-cols-[minmax(0,1fr)] items-start gap-8 xl:grid-cols-[minmax(0,1fr)_minmax(360px,44%)]'
            }
          >
            <article className="prose-custom min-w-0">{children}</article>
            {!hideCodePanel && (
              <>
                <aside className="sticky top-6 hidden xl:block">
                  <CodePanel ranges={codeRanges} />
                </aside>
                {/* below xl the panel follows the article instead of floating beside it */}
                <div className="mt-10 xl:hidden">
                  <CodePanel ranges={codeRanges} maxHeight="50vh" />
                </div>
              </>
            )}
          </div>

          <nav className="mt-16 flex justify-between border-t border-ink/10 pt-6" aria-label="chapter navigation">
            {prev ? (
              <Link
                to={`/ch/${prev.slug}`}
                className="group max-w-[45%] focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
              >
                <div className="font-mono text-xs text-muted">← previous</div>
                <div className="font-display text-lg group-hover:underline">{prev.title}</div>
              </Link>
            ) : (
              <span />
            )}
            {next ? (
              <Link
                to={`/ch/${next.slug}`}
                className="group max-w-[45%] text-right focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
              >
                <div className="font-mono text-xs text-muted">next →</div>
                <div className="font-display text-lg group-hover:underline">{next.title}</div>
              </Link>
            ) : (
              <span />
            )}
          </nav>
        </div>
      </div>
      </QuizProvider>
    </CodeSyncProvider>
  )
}
