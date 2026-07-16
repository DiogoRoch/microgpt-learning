/**
 * Lazily-loaded chapter (and gallery) chunks, made *preloadable*.
 *
 * Each chapter is its own code-split bundle. Left alone, navigating between
 * chapters unmounts the current page and shows the Suspense fallback while the
 * next chunk is fetched — a blank "white flash" on the paper canvas. We fix
 * that two ways: (1) navigation runs inside a React transition (see
 * `v7_startTransition` on the router) so the previous chapter stays on screen
 * until the next one is ready, and (2) we warm the chunks ahead of time — on
 * idle after first paint, and on hover/focus of any link that targets them —
 * so the transition resolves near-instantly.
 */
import { lazy, type ComponentType, type LazyExoticComponent } from 'react'

export type PreloadableComponent = LazyExoticComponent<ComponentType> & {
  preload: () => Promise<unknown>
}

function lazyWithPreload(
  factory: () => Promise<{ default: ComponentType }>,
): PreloadableComponent {
  const Component = lazy(factory) as PreloadableComponent
  Component.preload = factory
  return Component
}

export const chapterModules: PreloadableComponent[] = [
  lazyWithPreload(() => import('../chapters/ch00.tsx')),
  lazyWithPreload(() => import('../chapters/ch01.tsx')),
  lazyWithPreload(() => import('../chapters/ch02.tsx')),
  lazyWithPreload(() => import('../chapters/ch03.tsx')),
  lazyWithPreload(() => import('../chapters/ch04.tsx')),
  lazyWithPreload(() => import('../chapters/ch05.tsx')),
  lazyWithPreload(() => import('../chapters/ch06.tsx')),
  lazyWithPreload(() => import('../chapters/ch07.tsx')),
  lazyWithPreload(() => import('../chapters/ch08.tsx')),
  lazyWithPreload(() => import('../chapters/ch09.tsx')),
  lazyWithPreload(() => import('../chapters/ch10.tsx')),
  lazyWithPreload(() => import('../chapters/ch11.tsx')),
]

export const Gallery = lazyWithPreload(() => import('../pages/Gallery.tsx'))

/** Warm a single chapter's chunk (e.g. on hover) so its route swaps instantly. */
export function preloadChapter(id: number): void {
  chapterModules[id]?.preload()
}

let preloadedAll = false

/** Warm every chapter chunk once — call from an idle callback after first paint. */
export function preloadAllChapters(): void {
  if (preloadedAll) return
  preloadedAll = true
  for (const m of chapterModules) m.preload()
  Gallery.preload()
}
