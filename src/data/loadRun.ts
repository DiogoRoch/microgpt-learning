/**
 * Typed loader for the shipped precomputed run (src/data/run.json, produced
 * by tools/train_checkpoint.ts). Snapshots are base64-encoded little-endian
 * f32 flat param vectors in the file's param order.
 */
import type { Config } from '../engine/model.ts'
import { Model } from '../engine/model.ts'

export interface RunSnapshot {
  step: number
  loss: number | null
  params_b64: string
  samples: string[]
}

export interface RunData {
  config: Config
  numSteps: number
  snapshotEvery: number
  trainMs: number
  /** per-step losses from this repo's TS engine run */
  losses: number[]
  /** per-step losses from the Python reference run (golden) */
  pythonLosses: number[]
  finalLossPython: number
  maxAbsLossDiff: number
  snapshots: RunSnapshot[]
}

/** base64 → Float32Array, browser and Node. Copies to an aligned buffer. */
export function decodeF32(b64: string): Float32Array {
  let bytes: Uint8Array
  if (typeof atob === 'function') {
    const bin = atob(b64)
    bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  } else {
    bytes = Uint8Array.from(Buffer.from(b64, 'base64'))
  }
  return new Float32Array(bytes.buffer, 0, bytes.byteLength / 4)
}

/** Lazy-load the run artifact (it's ~1 MB — only chapters that need it pay). */
export async function loadRun(): Promise<RunData> {
  const mod = await import('./run.json')
  return mod.default as unknown as RunData
}

/** Materialize the model at a snapshot. */
export function modelAtSnapshot(run: RunData, snapshot: RunSnapshot): Model {
  return Model.fromFlat(run.config, decodeF32(snapshot.params_b64))
}

/** The snapshot at-or-before a given step (for scrubbing). */
export function snapshotForStep(run: RunData, step: number): RunSnapshot {
  let best = run.snapshots[0]!
  for (const s of run.snapshots) if (s.step <= step) best = s
  return best
}
