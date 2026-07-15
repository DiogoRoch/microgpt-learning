/**
 * Shared access to the model at any checkpoint of the shipped precomputed run
 * + live traces of the user's running example. Models are cached per step;
 * run.json (~1 MB) loads lazily, only in chapters that use it.
 */
import { useEffect, useMemo, useState } from 'react'
import { loadRun, modelAtSnapshot, snapshotForStep, type RunData } from '../data/loadRun.ts'
import { Model } from '../engine/model.ts'
import { softmaxProbs } from '../engine/tensor.ts'
import { Tokenizer } from '../engine/tokenizer.ts'
import type { GptCallTrace } from '../engine/trace.ts'
import tokenizerGolden from '../../golden/tokenizer.json'

export const tokenizer = Tokenizer.fromMeta(tokenizerGolden)
export const labelOf = (id: number) => (id === tokenizer.bos ? '·' : (tokenizer.uchars[id] ?? '?'))
export const VOCAB_LABELS: string[] = [...tokenizer.uchars, '·']

let runPromise: Promise<RunData> | null = null
const modelCache = new Map<number, Model>()

export function getRun(): Promise<RunData> {
  runPromise ??= loadRun()
  return runPromise
}

/** Model at the snapshot closest below `step` (cached). */
export async function getModelAt(step: number): Promise<Model> {
  const run = await getRun()
  const snap = snapshotForStep(run, step)
  let m = modelCache.get(snap.step)
  if (!m) {
    m = modelAtSnapshot(run, snap)
    modelCache.set(snap.step, m)
  }
  return m
}

export function useRun(): RunData | null {
  const [run, setRun] = useState<RunData | null>(null)
  useEffect(() => {
    let alive = true
    void getRun().then((r) => alive && setRun(r))
    return () => {
      alive = false
    }
  }, [])
  return run
}

export function useModelAt(step: number): Model | null {
  const [model, setModel] = useState<Model | null>(null)
  useEffect(() => {
    let alive = true
    void getModelAt(step).then((m) => alive && setModel(m))
    return () => {
      alive = false
    }
  }, [step])
  return model
}

export interface ExampleTrace {
  word: string
  tokens: number[]
  n: number
  calls: GptCallTrace[]
  /** probs per position (from the traced logits) */
  probs: Float64Array[]
  /** -log p(target) per position */
  lossT: number[]
  /** mean loss over the doc */
  loss: number
  model: Model
}

/** Run the model on a word and record every intermediate (on demand only). */
export function traceWord(model: Model, word: string): ExampleTrace {
  const tokens = tokenizer.encodeDoc(word)
  const calls: GptCallTrace[] = []
  const { n } = model.docLoss(tokens, calls)
  const probs = calls.map((c) => softmaxProbs(c.logits))
  const lossT = probs.map((p, i) => -Math.log(p[tokens[i + 1]!]!))
  const loss = lossT.reduce((a, b) => a + b, 0) / n
  return { word, tokens, n, calls, probs, lossT, loss, model }
}

/** Live trace of a word at a checkpoint; null while the run loads. */
export function useTrace(word: string, step: number): ExampleTrace | null {
  const model = useModelAt(step)
  return useMemo(() => (model ? traceWord(model, word) : null), [model, word])
}
