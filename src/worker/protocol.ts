/**
 * Typed message protocol between the UI and the training/inference worker.
 * The worker owns a Model + Adam and streams progress; heavy work never
 * touches the main thread.
 */

export type WorkerRequest =
  | {
      /** Load weights and reset the optimizer. */
      type: 'init'
      /** 'golden' = the exact initial weights the Python file drew; a number = fresh seeded init. */
      weights: 'golden' | { seed: number }
      numSteps?: number
    }
  | {
      /** Train from the current step; streams progress every reportEvery steps. */
      type: 'train'
      untilStep: number
      reportEvery: number
      /** Sample a few names at each report (shows babble → names). */
      samplesPerReport?: number
      temperature?: number
      /** ms to sleep between chunks — pacing so humans can watch (0 = flat out) */
      paceMs?: number
    }
  | { type: 'stop' }
  | { type: 'sample'; count: number; temperature: number; seed: number }
  | { type: 'getSnapshot' }

export type WorkerResponse =
  | { type: 'ready'; numParams: number; step: number }
  | { type: 'progress'; step: number; loss: number; lossesChunk: number[]; samples?: string[] }
  | { type: 'stopped'; step: number }
  | { type: 'trained'; step: number; elapsedMs: number; losses: number[] }
  | { type: 'samples'; names: string[] }
  | { type: 'snapshot'; step: number; params: Float32Array }
  | { type: 'error'; message: string }
