import { describe, expect, it } from 'vitest'
import { Tokenizer } from '../src/engine/tokenizer.ts'
import tokenizerGolden from '../golden/tokenizer.json'
import docsGolden from '../golden/docs.json'
import step0State from '../golden/step0_state.json'
import forcedDecodeInit from '../golden/forced_decode_init.json'

describe('tokenizer parity with golden dumps', () => {
  const tok = Tokenizer.fromMeta(tokenizerGolden)

  it('has the expected vocabulary shape (a-z + BOS = 27)', () => {
    expect(tokenizerGolden.uchars).toHaveLength(26)
    expect(tok.vocabSize).toBe(27)
    expect(tok.bos).toBe(26)
    expect(tokenizerGolden.vocab_size).toBe(tok.vocabSize)
    expect(tokenizerGolden.bos).toBe(tok.bos)
  })

  it('golden uchars are sorted and unique (sorted(set(...)))', () => {
    const sorted = [...new Set(tokenizerGolden.uchars)].sort()
    expect(tokenizerGolden.uchars).toEqual(sorted)
  })

  it('rebuilding the vocab from the shuffled doc head matches the file', () => {
    // 1024 names comfortably cover all 26 letters.
    const rebuilt = Tokenizer.fromDocs(docsGolden.head)
    expect(rebuilt.uchars).toEqual(tokenizerGolden.uchars)
  })

  it('encodes emma exactly as the reference forced decode did', () => {
    expect(tok.encodeDoc(forcedDecodeInit.doc)).toEqual(forcedDecodeInit.tokens)
  })

  it('encodes the step-0 training doc exactly as the reference did', () => {
    expect(tok.encodeDoc(step0State.doc)).toEqual(step0State.tokens)
  })

  it('round-trips every doc in the golden head', () => {
    for (const doc of docsGolden.head) {
      expect(tok.decode(tok.encodeChars(doc))).toBe(doc)
    }
  })

  it('rejects characters outside the vocabulary', () => {
    expect(() => tok.encodeChars('Emma!')).toThrow(/not in the vocabulary/)
  })
})
