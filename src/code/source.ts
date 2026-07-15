/**
 * The source of truth, loaded at build time from reference/microgpt.py
 * (fetched by tools/fetch_reference.sh — see the pre-scripts in package.json).
 * The file is intentionally not committed; it ships only inside the built app,
 * attributed and linked to the gist.
 */
import raw from '../../reference/microgpt.py?raw'

export const GIST_URL = 'https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95'
export const GIST_REV = '14fb038816c7aae0bb9342c2dbf1a51dd134a5ff'

export const SOURCE = raw.replace(/\n$/, '')

/** 1-based access; lines[0] is line 1. */
export const SOURCE_LINES: string[] = SOURCE.split('\n')

/** Inclusive 1-based slice of the file. */
export function sourceRange(start: number, end: number): string[] {
  return SOURCE_LINES.slice(start - 1, end)
}
