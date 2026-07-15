/**
 * Placeholder chapter body used while chapters are built out phase by phase.
 * Real chapters replace these files in Phases 3–5.
 */
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'

export function makeStub(id: number) {
  const chapter = CHAPTERS[id]!
  return function ChapterStubPage() {
    return (
      <ChapterFrame chapter={chapter}>
        <div className="rounded-lg border border-dashed border-ink/20 p-8 text-muted">
          <p className="font-mono text-sm">
            This chapter&apos;s interactive core is under construction. The code panel on the right
            already shows the exact lines of microgpt.py it will teach.
          </p>
        </div>
      </ChapterFrame>
    )
  }
}
