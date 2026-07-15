/**
 * KaTeX rendering for math asides. This module (and KaTeX's CSS+fonts) only
 * loads in chunks that actually show math.
 */
import katex from 'katex'
import 'katex/dist/katex.min.css'
import { useMemo } from 'react'

export function K({ tex, block = false }: { tex: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: block, throwOnError: false }),
    [tex, block],
  )
  return <span className={block ? 'block overflow-x-auto py-1' : undefined} dangerouslySetInnerHTML={{ __html: html }} />
}
