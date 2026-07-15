/**
 * Test-side loader for the reference source. Node's test runner can't use
 * Vite's `?raw` imports at config-collection time in every context, so tests
 * read the fetched file straight from disk (pretest fetches it).
 */
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const p = join(dirname(fileURLToPath(import.meta.url)), '..', 'reference', 'microgpt.py')
export const SOURCE_LINES: string[] = readFileSync(p, 'utf8').replace(/\n$/, '').split('\n')
export const TOTAL_LINES = 200
