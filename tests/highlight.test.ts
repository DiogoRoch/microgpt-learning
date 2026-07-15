import { describe, expect, it } from 'vitest'
import { tokenizeSource } from '../src/code/highlight.ts'
import { SOURCE_LINES, TOTAL_LINES } from './sourceForTest.ts'
import { CHAPTERS, lineOwners } from '../src/app/chapters.ts'

describe('mini Python highlighter on the real file', () => {
  const tokenized = tokenizeSource(SOURCE_LINES)

  it('reconstructs the file verbatim from tokens (never eats a character)', () => {
    for (let i = 0; i < SOURCE_LINES.length; i++) {
      expect(tokenized[i]!.map((t) => t.text).join(''), `line ${i + 1}`).toBe(SOURCE_LINES[i]!)
    }
  })

  it('classifies known lines correctly', () => {
    const kinds = (lineNo: number) => tokenized[lineNo - 1]!.map((t) => `${t.kind}:${t.text}`)
    // line 12: random.seed(42) # Let there be order among chaos
    expect(kinds(12)).toContain('num:42')
    expect(kinds(12).some((k) => k.startsWith('comment:# Let there be order'))).toBe(true)
    // line 30: class Value:
    expect(kinds(30)).toContain('kw:class')
    expect(kinds(30)).toContain('def:Value')
    // line 1/7: docstring delimiters tracked across lines
    expect(tokenized[0]![0]!.kind).toBe('str')
    expect(tokenized[6]![0]!.kind).toBe('str') // closing """ of the docstring
    // line 17 contains a URL inside a string
    expect(tokenized[16]!.some((t) => t.kind === 'str' && t.text.includes('makemore'))).toBe(true)
  })
})

describe('file map', () => {
  it('the source has exactly 199 lines', () => {
    expect(SOURCE_LINES).toHaveLength(TOTAL_LINES)
  })

  it('every line has exactly one primary chapter (blank separators too)', () => {
    const owners = lineOwners()
    const unowned: number[] = []
    for (let l = 1; l <= TOTAL_LINES; l++) {
      const line = SOURCE_LINES[l - 1]!
      if (owners[l] === -1 && line.trim() !== '') unowned.push(l)
    }
    expect(unowned, `unowned non-blank lines: ${unowned.join(',')}`).toEqual([])
  })

  it('chapter ranges never overlap', () => {
    const seen = new Map<number, number>()
    for (const ch of CHAPTERS) {
      for (const [a, b] of ch.lines) {
        for (let l = a; l <= b; l++) {
          expect(seen.has(l), `line ${l} claimed by both ch${seen.get(l)} and ch${ch.id}`).toBe(false)
          seen.set(l, ch.id)
        }
      }
    }
  })

  it('slugs and ids are unique and sequential', () => {
    expect(new Set(CHAPTERS.map((c) => c.slug)).size).toBe(CHAPTERS.length)
    expect(CHAPTERS.map((c) => c.id)).toEqual(Array.from({ length: 12 }, (_, i) => i))
  })
})
