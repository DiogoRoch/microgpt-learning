/**
 * Character-level tokenizer — mirrors microgpt.py exactly:
 *
 *   uchars = sorted(set(''.join(docs)))   # ids 0..n-1
 *   BOS = len(uchars)                     # one special token, id = n
 *   vocab_size = len(uchars) + 1
 *
 * Training/inference bracket every document with BOS on BOTH sides:
 *   tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
 */

export interface TokenizerMeta {
  uchars: string[]
  bos: number
  vocab_size: number
}

export class Tokenizer {
  readonly uchars: readonly string[]
  readonly bos: number
  readonly vocabSize: number
  private readonly charToId: Map<string, number>

  constructor(uchars: readonly string[]) {
    this.uchars = uchars
    this.bos = uchars.length
    this.vocabSize = uchars.length + 1
    this.charToId = new Map(uchars.map((ch, i) => [ch, i]))
  }

  /** Build from a document list the way the file does: sorted unique chars. */
  static fromDocs(docs: readonly string[]): Tokenizer {
    return new Tokenizer([...new Set(docs.join(''))].sort())
  }

  static fromMeta(meta: TokenizerMeta): Tokenizer {
    return new Tokenizer(meta.uchars)
  }

  /** Character ids only — throws on characters outside the vocabulary. */
  encodeChars(s: string): number[] {
    return [...s].map((ch) => {
      const id = this.charToId.get(ch)
      if (id === undefined) throw new Error(`character ${JSON.stringify(ch)} is not in the vocabulary`)
      return id
    })
  }

  /** A training/inference document: [BOS] + chars + [BOS]. */
  encodeDoc(doc: string): number[] {
    return [this.bos, ...this.encodeChars(doc), this.bos]
  }

  /** Inverse of encodeChars; BOS renders as '·' (it has no character). */
  decode(ids: readonly number[]): string {
    return ids.map((id) => (id === this.bos ? '·' : (this.uchars[id] ?? '?'))).join('')
  }

  isInVocab(ch: string): boolean {
    return this.charToId.has(ch)
  }
}
