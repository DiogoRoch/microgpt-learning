/**
 * A tiny expression parser for the chapter-2 autograd sandbox. Builds REAL V
 * graphs via the engine's own ops, so the sandbox exercises the same code
 * paths as the model (including Python-style desugaring: -x is x*-1,
 * a/b is a*b**-1).
 *
 * Grammar:
 *   expr   := term (('+'|'-') term)*
 *   term   := unary (('*'|'/') unary)*
 *   unary  := '-' unary | power
 *   power  := atom ('**' NUMBER)?          — exponents must be literal numbers,
 *                                            exactly like Value.__pow__
 *   atom   := NUMBER | IDENT | FUNC '(' expr ')' | '(' expr ')'
 *   FUNC   := log | exp | relu
 */
import { V } from '../engine/graph.ts'

interface Token {
  kind: 'num' | 'ident' | 'op'
  text: string
}

function lex(src: string): Token[] | string {
  const toks: Token[] = []
  let i = 0
  while (i < src.length) {
    const ch = src[i]!
    if (ch === ' ' || ch === '\t') {
      i++
      continue
    }
    if (/\d/.test(ch) || (ch === '.' && /\d/.test(src[i + 1] ?? ''))) {
      const m = /^\d*\.?\d+(e-?\d+)?/i.exec(src.slice(i))!
      toks.push({ kind: 'num', text: m[0] })
      i += m[0].length
      continue
    }
    if (/[a-zA-Z_]/.test(ch)) {
      const m = /^[a-zA-Z_][a-zA-Z0-9_]*/.exec(src.slice(i))!
      toks.push({ kind: 'ident', text: m[0] })
      i += m[0].length
      continue
    }
    if (src.startsWith('**', i)) {
      toks.push({ kind: 'op', text: '**' })
      i += 2
      continue
    }
    if ('+-*/()'.includes(ch)) {
      toks.push({ kind: 'op', text: ch })
      i++
      continue
    }
    return `unexpected character '${ch}'`
  }
  return toks
}

const FUNCS = new Set(['log', 'exp', 'relu'])

export interface ParsedExpr {
  root: V
  /** leaf variables in first-appearance order */
  vars: Map<string, V>
}

export function parseExpr(src: string, values: Record<string, number>): ParsedExpr | { error: string } {
  const lexed = lex(src)
  if (typeof lexed === 'string') return { error: lexed }
  const toks = lexed
  if (toks.length === 0) return { error: 'empty expression' }
  let p = 0
  const vars = new Map<string, V>()

  const peek = () => toks[p]
  const eat = (text?: string): Token | null => {
    const t = toks[p]
    if (!t) return null
    if (text !== undefined && t.text !== text) return null
    p++
    return t
  }

  function expr(): V | string {
    let left = term()
    if (typeof left === 'string') return left
    for (;;) {
      const t = peek()
      if (t?.kind === 'op' && (t.text === '+' || t.text === '-')) {
        p++
        const right = term()
        if (typeof right === 'string') return right
        left = t.text === '+' ? left.add(right) : left.sub(right)
      } else return left
    }
  }

  function term(): V | string {
    let left = unary()
    if (typeof left === 'string') return left
    for (;;) {
      const t = peek()
      if (t?.kind === 'op' && (t.text === '*' || t.text === '/')) {
        p++
        const right = unary()
        if (typeof right === 'string') return right
        left = t.text === '*' ? left.mul(right) : left.div(right)
      } else return left
    }
  }

  function unary(): V | string {
    if (eat('-')) {
      const inner = unary()
      if (typeof inner === 'string') return inner
      return inner.neg() // Python's __neg__: a real x * -1 node
    }
    return power()
  }

  function power(): V | string {
    const base = atom()
    if (typeof base === 'string') return base
    if (peek()?.text === '**') {
      p++
      let sign = 1
      if (eat('-')) sign = -1
      const t = eat()
      if (!t || t.kind !== 'num') return 'exponent must be a literal number (Value.__pow__ only supports that)'
      return base.pow(sign * Number(t.text))
    }
    return base
  }

  function atom(): V | string {
    const t = eat()
    if (!t) return 'unexpected end of expression'
    if (t.kind === 'num') return new V(Number(t.text))
    if (t.kind === 'ident') {
      if (FUNCS.has(t.text)) {
        if (!eat('(')) return `${t.text} needs parentheses: ${t.text}(x)`
        const inner = expr()
        if (typeof inner === 'string') return inner
        if (!eat(')')) return 'missing )'
        if (t.text === 'log') {
          const arg = inner.data
          if (arg <= 0) return `log of ${arg} — the file would crash here too (math domain error)`
          return inner.log()
        }
        if (t.text === 'exp') return inner.exp()
        return inner.relu()
      }
      let v = vars.get(t.text)
      if (!v) {
        v = new V(values[t.text] ?? 1)
        vars.set(t.text, v)
      }
      return v
    }
    if (t.text === '(') {
      const inner = expr()
      if (typeof inner === 'string') return inner
      if (!eat(')')) return 'missing )'
      return inner
    }
    return `unexpected '${t.text}'`
  }

  const root = expr()
  if (typeof root === 'string') return { error: root }
  if (p !== toks.length) return { error: `unexpected '${toks[p]!.text}'` }
  if (!(root instanceof V) || root.children.length === 0) {
    // a bare number/variable — legal but nothing to differentiate
    return { root, vars }
  }
  return { root, vars }
}
