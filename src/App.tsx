import { Suspense, useEffect } from 'react'
import { BrowserRouter, Link, Navigate, Route, Routes, useParams } from 'react-router-dom'
import { CHAPTERS, chapterBySlug } from './app/chapters.ts'
import {
  Gallery,
  chapterModules,
  preloadAllChapters,
  preloadChapter,
} from './app/chapterModules.ts'
import { GIST_URL } from './code/source.ts'

function ChapterRoute() {
  const { slug } = useParams()
  const chapter = slug ? chapterBySlug(slug) : undefined
  if (!chapter) return <Navigate to="/" replace />
  const Page = chapterModules[chapter.id]!
  return <Page />
}

function Header() {
  return (
    <header className="border-b border-ink/10">
      <div className="mx-auto flex max-w-[1500px] items-baseline justify-between px-4 py-3 md:px-8">
        <Link to="/" className="focus-visible:outline-2 focus-visible:outline-[var(--hot)]">
          <span className="font-display text-lg font-semibold">MicroGPT</span>
          <span className="ml-2 hidden font-mono text-xs text-muted sm:inline">explained interactively</span>
        </Link>
        <nav aria-label="chapters" className="flex items-center gap-1 overflow-x-auto">
          {CHAPTERS.map((c) => (
            <Link
              key={c.id}
              to={c.id === 0 ? '/' : `/ch/${c.slug}`}
              title={c.title}
              onMouseEnter={() => preloadChapter(c.id)}
              onFocus={() => preloadChapter(c.id)}
              className="rounded px-1.5 py-0.5 font-mono text-[11px] text-muted hover:bg-ink/5 hover:text-ink focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
            >
              {c.id}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  )
}

function Footer() {
  return (
    <footer className="mt-20 border-t border-ink/10 py-8 text-center font-mono text-xs text-muted">
      <p>
        Everything here is{' '}
        <a href={GIST_URL} target="_blank" rel="noreferrer" className="underline underline-offset-2 hover:text-ink">
          microgpt.py by Andrej Karpathy
        </a>
        {' '}— every number on screen is computed by a parity-tested reimplementation of that file.
      </p>
    </footer>
  )
}

const base = import.meta.env.BASE_URL.replace(/\/$/, '')

export default function App() {
  // Warm every chapter chunk once the browser is idle, so later navigations
  // resolve without a Suspense fallback flash.
  useEffect(() => {
    const ric = window.requestIdleCallback
    if (ric) {
      const handle = ric(() => preloadAllChapters())
      return () => window.cancelIdleCallback?.(handle)
    }
    const handle = window.setTimeout(preloadAllChapters, 300)
    return () => window.clearTimeout(handle)
  }, [])

  return (
    <BrowserRouter basename={base} future={{ v7_startTransition: true }}>
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-[var(--hot)] focus:px-3 focus:py-1"
      >
        skip to content
      </a>
      <Header />
      <main id="main">
        <Suspense
          fallback={
            <div className="p-16 text-center font-mono text-sm text-muted" role="status">
              loading chapter…
            </div>
          }
        >
          <Routes>
            <Route path="/" element={<ChapterRouteById id={0} />} />
            <Route path="/ch/:slug" element={<ChapterRoute />} />
            <Route path="/dev/gallery" element={<Gallery />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </main>
      <Footer />
    </BrowserRouter>
  )
}

function ChapterRouteById({ id }: { id: number }) {
  const Page = chapterModules[id]!
  return <Page />
}
