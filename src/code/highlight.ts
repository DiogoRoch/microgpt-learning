/**
 * A tiny Python tokenizer-highlighter, hand-rolled for exactly one file.
 *
 * Why not Shiki/CodeMirror: we highlight a single, known, 199-line file whose
 * every line the app is built around. A 60-line tokenizer gives us complete
 * control over line anatomy (needed for per-line sync, hover provenance and
 * the minimap) at zero bundle cost. A test reconstructs the file verbatim
 * from the tokens so the tokenizer can never silently eat characters, and a
 * snapshot-style test pins known lines to known token kinds.
 */

export type TokKind = 'kw' | 'builtin' | 'str' | 'num' | 'comment' | 'def' | 'op' | 'plain'

export interface Tok {
  kind: TokKind
  text: string
}

const KEYWORDS = new Set([
  'def', 'return', 'for', 'in', 'if', 'not', 'else', 'elif', 'while', 'break', 'continue',
  'import', 'from', 'class', 'lambda', 'and', 'or', 'None', 'True', 'False', 'is', 'with', 'as',
])
const BUILTINS = new Set([
  'print', 'len', 'range', 'open', 'sorted', 'set', 'sum', 'max', 'min', 'zip', 'enumerate',
  'isinstance', 'float', 'int', 'str', 'list', 'dict', 'self', '__init__', '__slots__',
])

const WORD = /^[A-Za-z_][A-Za-z0-9_]*/
const NUM = /^\d+(?:\.\d+)?(?:e-?\d+)?/i

/**
 * Tokenize one line. `inString` carries multi-line ('''…''') string state
 * between lines; returns the state after this line.
 */
export function tokenizeLine(line: string, inString: boolean): { toks: Tok[]; inString: boolean } {
  const toks: Tok[] = []
  let i = 0
  let expectDefName = false
  const push = (kind: TokKind, text: string) => {
    if (text.length === 0) return
    const last = toks[toks.length - 1]
    if (last && last.kind === kind) last.text += text
    else toks.push({ kind, text })
  }

  if (inString) {
    const end = line.indexOf('"""')
    if (end === -1) {
      push('str', line)
      return { toks, inString: true }
    }
    push('str', line.slice(0, end + 3))
    i = end + 3
    inString = false
  }

  while (i < line.length) {
    const rest = line.slice(i)
    if (rest.startsWith('#')) {
      push('comment', rest)
      break
    }
    if (rest.startsWith('"""')) {
      const end = line.indexOf('"""', i + 3)
      if (end === -1) {
        push('str', rest)
        return { toks, inString: true }
      }
      push('str', line.slice(i, end + 3))
      i = end + 3
      continue
    }
    const ch = line[i]!
    if (ch === "'" || ch === '"') {
      // f-strings arrive here via the preceding word check below
      let j = i + 1
      while (j < line.length && line[j] !== ch) j++
      push('str', line.slice(i, j + 1))
      i = j + 1
      continue
    }
    const num = NUM.exec(rest)
    if (num && !/[A-Za-z0-9_]/.test(line[i - 1] ?? ' ')) {
      push('num', num[0])
      i += num[0].length
      continue
    }
    const word = WORD.exec(rest)
    if (word) {
      const w = word[0]
      if ((w === 'f' || w === 'r') && (line[i + w.length] === "'" || line[i + w.length] === '"')) {
        // string prefix: color it with the string
        const quote = line[i + w.length]!
        let j = i + w.length + 1
        while (j < line.length && line[j] !== quote) j++
        push('str', line.slice(i, j + 1))
        i = j + 1
        continue
      }
      if (KEYWORDS.has(w)) {
        push('kw', w)
        expectDefName = w === 'def' || w === 'class'
      } else if (BUILTINS.has(w)) {
        push('builtin', w)
        expectDefName = false
      } else if (expectDefName) {
        push('def', w)
        expectDefName = false
      } else {
        push('plain', w)
      }
      i += w.length
      continue
    }
    if (/[-+*/%=<>!&|^~@(){}[\],.:;]/.test(ch)) {
      push('op', ch)
      expectDefName = false
      i++
      continue
    }
    push('plain', ch)
    if (ch !== ' ' && ch !== '\t') expectDefName = false
    i++
  }
  return { toks, inString }
}

/** Tokenize a whole file (multi-line-string state threaded through). */
export function tokenizeSource(lines: readonly string[]): Tok[][] {
  const out: Tok[][] = []
  let inString = false
  for (const line of lines) {
    const r = tokenizeLine(line, inString)
    out.push(r.toks)
    inString = r.inString
  }
  return out
}
