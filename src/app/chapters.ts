/**
 * Chapter registry + the file map: which lines of microgpt.py each chapter
 * teaches. The minimap uses `lines` to light up regions as chapters complete;
 * "you now understand this entire file" is literal — every line of the
 * 199-line file has exactly one primary chapter (checked by a test).
 *
 * Ranges are inclusive [start, end] against the pinned gist revision.
 */

export interface Chapter {
  id: number
  slug: string
  title: string
  /** short nav label */
  short: string
  subtitle: string
  /** primary line ranges of reference/microgpt.py this chapter teaches */
  lines: Array<[number, number]>
}

export const CHAPTERS: Chapter[] = [
  {
    id: 0,
    slug: 'big-picture',
    title: 'The Big Picture',
    short: 'Big picture',
    subtitle: 'One file, one screen: the entire algorithm as a living data-flow map.',
    lines: [[1, 13], [92, 93]],
  },
  {
    id: 1,
    slug: 'data-tokenizer',
    title: 'Data & Tokenizer',
    short: 'Tokenizer',
    subtitle: '32,033 names become integers — and why BOS brackets both sides.',
    lines: [[14, 27]],
  },
  {
    id: 2,
    slug: 'autograd',
    title: 'Autograd',
    short: 'Autograd',
    subtitle: 'A 40-line Value class that remembers how it computed everything.',
    lines: [[29, 72]],
  },
  {
    id: 3,
    slug: 'parameters',
    title: 'Parameters',
    short: 'Parameters',
    subtitle: 'All 4,192 numbers the model will ever learn, mapped.',
    lines: [[74, 90]],
  },
  {
    id: 4,
    slug: 'embeddings',
    title: 'Embeddings',
    short: 'Embeddings',
    subtitle: 'wte[token] + wpe[pos], then rmsnorm — a token becomes a vector.',
    lines: [[103, 106], [108, 112]],
  },
  {
    id: 5,
    slug: 'attention',
    title: 'Attention',
    short: 'Attention',
    subtitle: 'The centerpiece: growing keys and values, and why there is no mask.',
    lines: [[94, 95], [114, 134]],
  },
  {
    id: 6,
    slug: 'mlp-residual',
    title: 'MLP & the Residual Stream',
    short: 'MLP',
    subtitle: '16 → 64 → ReLU → 16, and the spine everything reads from and writes to.',
    lines: [[135, 141]],
  },
  {
    id: 7,
    slug: 'loss',
    title: 'Logits → Probabilities → Loss',
    short: 'Loss',
    subtitle: 'Softmax, -log p(target), and why training starts at exactly ln(27).',
    lines: [[97, 101], [143, 144], [160, 169]],
  },
  {
    id: 8,
    slug: 'backward-adam',
    title: 'Backward & Adam',
    short: 'Backward',
    subtitle: 'Gradients flow back over the whole graph; Adam turns them into learning.',
    lines: [[146, 149], [171, 182]],
  },
  {
    id: 9,
    slug: 'training',
    title: 'Training',
    short: 'Training',
    subtitle: 'Watch the real model learn names, live in your browser.',
    lines: [[151, 158], [183, 184]],
  },
  {
    id: 10,
    slug: 'inference',
    title: 'Inference',
    short: 'Inference',
    subtitle: 'Sampling with the KV cache, and what temperature really does.',
    lines: [[186, 200]],
  },
  {
    id: 11,
    slug: 'playground',
    title: 'Playground',
    short: 'Playground',
    subtitle: 'Everything unlocked. The full file, conquered.',
    lines: [],
  },
]

export const TOTAL_LINES = 200

export function chapterBySlug(slug: string): Chapter | undefined {
  return CHAPTERS.find((c) => c.slug === slug)
}

/** chapter id owning each 1-based line (playground owns nothing). */
export function lineOwners(): Int8Array {
  const owners = new Int8Array(TOTAL_LINES + 1).fill(-1)
  for (const ch of CHAPTERS) {
    for (const [a, b] of ch.lines) {
      for (let l = a; l <= b; l++) owners[l] = ch.id
    }
  }
  return owners
}
