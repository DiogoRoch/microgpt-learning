/**
 * React handle on the training worker: spawn once per mount, stream progress
 * into state, expose train/stop/reset. The main thread only ever holds
 * numbers; all math happens in the worker.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { WorkerRequest, WorkerResponse } from '../worker/protocol.ts'

export interface TrainerState {
  ready: boolean
  training: boolean
  step: number
  losses: number[]
  /** latest sampled names (evolving during training) */
  samples: string[]
  sampleHistory: Array<{ step: number; names: string[] }>
  elapsedMs: number | null
}

export interface Trainer extends TrainerState {
  init: (weights: 'golden' | { seed: number }) => void
  train: (opts?: { untilStep?: number; reportEvery?: number; samplesPerReport?: number; paceMs?: number }) => void
  stop: () => void
}

export function useTrainer(): Trainer {
  const workerRef = useRef<Worker | null>(null)
  const lossesRef = useRef<number[]>([])
  const [state, setState] = useState<TrainerState>({
    ready: false, training: false, step: 0, losses: [], samples: [], sampleHistory: [], elapsedMs: null,
  })

  useEffect(() => {
    const worker = new Worker(new URL('../worker/train.worker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data
      switch (msg.type) {
        case 'ready':
          lossesRef.current = []
          setState((s) => ({ ...s, ready: true, training: false, step: msg.step, losses: [], samples: [], sampleHistory: [], elapsedMs: null }))
          break
        case 'progress': {
          setState((s) => ({
            ...s,
            training: true,
            step: msg.step,
            losses: [...s.losses, ...msg.lossesChunk],
            samples: msg.samples ?? s.samples,
            sampleHistory: msg.samples ? [...s.sampleHistory, { step: msg.step, names: msg.samples }] : s.sampleHistory,
          }))
          break
        }
        case 'trained':
          setState((s) => ({ ...s, training: false, step: msg.step, losses: msg.losses.length ? msg.losses : s.losses, elapsedMs: msg.elapsedMs }))
          break
        case 'stopped':
          setState((s) => ({ ...s, training: false, step: msg.step }))
          break
        default:
          break
      }
    }
    const req: WorkerRequest = { type: 'init', weights: 'golden' }
    worker.postMessage(req)
    return () => worker.terminate()
  }, [])

  const send = useCallback((req: WorkerRequest) => workerRef.current?.postMessage(req), [])

  const init = useCallback((weights: 'golden' | { seed: number }) => send({ type: 'init', weights }), [send])
  const train = useCallback(
    (opts?: { untilStep?: number; reportEvery?: number; samplesPerReport?: number; paceMs?: number }) => {
      setState((s) => ({ ...s, training: true }))
      send({
        type: 'train',
        untilStep: opts?.untilStep ?? 1000,
        reportEvery: opts?.reportEvery ?? 5,
        samplesPerReport: opts?.samplesPerReport ?? 6,
        paceMs: opts?.paceMs ?? 12,
      })
    },
    [send],
  )
  const stop = useCallback(() => send({ type: 'stop' }), [send])

  return { ...state, init, train, stop }
}
