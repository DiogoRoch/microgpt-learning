/**
 * Training/inference worker. Replays the file's exact run (golden init
 * weights + golden shuffled doc order) or trains from a fresh seeded init.
 * Chunked loop so stop messages get through; the main thread only ever
 * receives numbers to draw.
 */
import { Adam, MICRO_CONFIG, Model, trainStep } from '../engine/model.ts'
import { RNG } from '../engine/rng.ts'
import { Tokenizer } from '../engine/tokenizer.ts'
import type { WorkerRequest, WorkerResponse } from './protocol.ts'
import initWeights from '../../golden/init_weights.json'
import tokenizerGolden from '../../golden/tokenizer.json'
import docsGolden from '../../golden/docs.json'

const tok = Tokenizer.fromMeta(tokenizerGolden)
const docs: string[] = docsGolden.head
const docTokens = docs.map((d) => tok.encodeDoc(d))

let model: Model | null = null
let adam: Adam | null = null
let step = 0
let training = false
let stopRequested = false

const post = (msg: WorkerResponse, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(msg, transfer)

function init(weights: 'golden' | { seed: number }, numSteps = 1000): void {
  model =
    weights === 'golden'
      ? Model.fromWeights(MICRO_CONFIG, initWeights as Record<string, number[][]>)
      : Model.init(MICRO_CONFIG, new RNG(weights.seed))
  adam = new Adam(model.numParams, numSteps)
  step = 0
  post({ type: 'ready', numParams: model.numParams, step })
}

function sampleNames(count: number, temperature: number, rng: RNG): string[] {
  const names: string[] = []
  for (let i = 0; i < count; i++) names.push(tok.decode(model!.sample(rng, temperature)))
  return names
}

async function train(untilStep: number, reportEvery: number, samplesPerReport: number, temperature: number) {
  if (!model || !adam) {
    post({ type: 'error', message: 'train before init' })
    return
  }
  training = true
  stopRequested = false
  const losses: number[] = []
  const t0 = performance.now()
  const CHUNK = 10
  while (step < untilStep) {
    const end = Math.min(step + CHUNK, untilStep)
    for (; step < end; step++) {
      const tokens = docTokens[step % docTokens.length]!
      losses.push(trainStep(model, adam, tokens, step))
      if ((step + 1) % reportEvery === 0 || step + 1 === untilStep) {
        post({
          type: 'progress',
          step: step + 1,
          loss: losses[losses.length - 1]!,
          samples: samplesPerReport > 0 ? sampleNames(samplesPerReport, temperature, new RNG(1000 + step)) : undefined,
        })
      }
    }
    // Yield so a queued 'stop' can be handled between chunks.
    await new Promise((r) => setTimeout(r, 0))
    if (stopRequested) {
      training = false
      post({ type: 'stopped', step })
      return
    }
  }
  training = false
  post({ type: 'trained', step, elapsedMs: performance.now() - t0, losses })
}

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data
  switch (msg.type) {
    case 'init':
      init(msg.weights, msg.numSteps)
      break
    case 'train':
      if (!training) void train(msg.untilStep, msg.reportEvery, msg.samplesPerReport ?? 0, msg.temperature ?? 0.5)
      break
    case 'stop':
      stopRequested = true
      break
    case 'sample':
      if (model) post({ type: 'samples', names: sampleNames(msg.count, msg.temperature, new RNG(msg.seed)) })
      break
    case 'getSnapshot':
      if (model) {
        const params = model.flatParams()
        post({ type: 'snapshot', step, params }, [params.buffer])
      }
      break
  }
}
