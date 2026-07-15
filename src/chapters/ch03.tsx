/**
 * Chapter 3 — Parameters. A treemap of exactly where the 4,192 numbers live,
 * the Gaussian init (real histogram of the real init values), a matrix
 * explorer, and the untrained→trained foreshadowing.
 */
import { useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { CompareToggle } from '../components/Compare.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { MatrixHeatmap } from '../viz/MatrixHeatmap.tsx'
import { useAppStore } from '../app/store.ts'
import { useModelAt, tokenizer } from '../app/useModel.ts'
import initWeights from '../../golden/init_weights.json'

const weights = initWeights as Record<string, number[][]>

interface MatInfo {
  key: string
  rows: number
  cols: number
  line: number
  role: string
  rowLabel?: (r: number) => string
  colLabel?: (c: number) => string
}

const tokenName = (r: number) => (r === 26 ? "'·' BOS (token 26)" : `'${tokenizer.uchars[r]}' (token ${r})`)

const MATS: MatInfo[] = [
  { key: 'wte', rows: 27, cols: 16, line: 81, role: 'token embeddings — one 16-dim vector per vocabulary token. Row 4 is what the model knows about the letter e.', rowLabel: tokenName, colLabel: (c) => `dim ${c}` },
  { key: 'wpe', rows: 16, cols: 16, line: 81, role: 'position embeddings — one 16-dim vector per position 0…15. How "being the 3rd character" feels.', rowLabel: (r) => `position ${r}`, colLabel: (c) => `dim ${c}` },
  { key: 'lm_head', rows: 27, cols: 16, line: 81, role: 'the output head — projects the final 16-dim state onto 27 scores, one per next-token candidate.', rowLabel: tokenName, colLabel: (c) => `dim ${c}` },
  { key: 'layer0.attn_wq', rows: 16, cols: 16, line: 83, role: 'attention query projection — "what am I looking for?"', colLabel: (c) => `x dim ${c}`, rowLabel: (r) => `q dim ${r} (head ${Math.floor(r / 4)})` },
  { key: 'layer0.attn_wk', rows: 16, cols: 16, line: 84, role: 'attention key projection — "what do I advertise?"', colLabel: (c) => `x dim ${c}`, rowLabel: (r) => `k dim ${r} (head ${Math.floor(r / 4)})` },
  { key: 'layer0.attn_wv', rows: 16, cols: 16, line: 85, role: 'attention value projection — "what do I hand over if attended to?"', colLabel: (c) => `x dim ${c}`, rowLabel: (r) => `v dim ${r} (head ${Math.floor(r / 4)})` },
  { key: 'layer0.attn_wo', rows: 16, cols: 16, line: 86, role: 'attention output projection — mixes the 4 concatenated heads back into the residual stream.', colLabel: (c) => `head ${Math.floor(c / 4)} out ${c % 4}`, rowLabel: (r) => `x dim ${r}` },
  { key: 'layer0.mlp_fc1', rows: 64, cols: 16, line: 87, role: 'MLP expansion — 16 → 64. Each of the 64 rows is one ReLU neuron′s detector pattern.', rowLabel: (r) => `neuron ${r}`, colLabel: (c) => `x dim ${c}` },
  { key: 'layer0.mlp_fc2', rows: 16, cols: 64, line: 88, role: 'MLP contraction — 64 → 16, writing the detected features back into the residual stream.', rowLabel: (r) => `x dim ${r}`, colLabel: (c) => `neuron ${c}` },
]

const count = (m: MatInfo) => m.rows * m.cols
const TOTAL = MATS.reduce((a, m) => a + count(m), 0)

/** Slice-and-dice treemap: columns proportional to param count. */
function Treemap({ selected, onSelect }: { selected: string; onSelect: (k: string) => void }) {
  const { setHighlight } = useCodeSync()
  const W = 640
  const H = 180
  let x = 0
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="group" aria-label="where the 4,192 parameters live">
      {MATS.map((m) => {
        const w = (count(m) / TOTAL) * W
        const rect = (
          <g key={m.key} transform={`translate(${x},0)`}>
            <rect
              width={w - 2}
              height={H}
              rx={6}
              fill={selected === m.key ? 'var(--ink)' : 'rgba(22,24,29,0.08)'}
              stroke={selected === m.key ? 'var(--hot)' : 'rgba(22,24,29,0.25)'}
              strokeWidth={selected === m.key ? 2 : 1}
              style={{ cursor: 'pointer' }}
              onClick={() => {
                onSelect(m.key)
                setHighlight([m.line])
              }}
            />
            {w > 90 ? (
              <>
                <text
                  x={8}
                  y={20}
                  fontSize={11}
                  fontWeight={600}
                  className="pointer-events-none font-mono"
                  fill={selected === m.key ? 'var(--paper)' : 'var(--ink)'}
                >
                  {m.key.replace('layer0.', '')}
                </text>
                <text
                  x={8}
                  y={36}
                  fontSize={10}
                  className="pointer-events-none font-mono"
                  fill={selected === m.key ? 'rgba(250,249,247,0.7)' : 'var(--muted)'}
                >
                  {m.rows}×{m.cols} = {count(m).toLocaleString()}
                </text>
              </>
            ) : (
              <text
                transform={`translate(${(w - 2) / 2 + 4}, 12) rotate(90)`}
                fontSize={10}
                fontWeight={600}
                className="pointer-events-none font-mono"
                fill={selected === m.key ? 'var(--paper)' : 'var(--ink)'}
              >
                {m.key.replace('layer0.', '')} · {count(m).toLocaleString()}
              </text>
            )}
          </g>
        )
        x += w
        return rect
      })}
    </svg>
  )
}

/** Histogram of all 4,192 real init values with the σ=0.08 Gaussian overlaid. */
function InitHistogram() {
  const { bins, maxBin, curve } = useMemo(() => {
    const all: number[] = []
    for (const m of MATS) for (const row of weights[m.key]!) all.push(...row)
    const B = 41
    const lo = -0.3
    const hi = 0.3
    const bins = new Array<number>(B).fill(0)
    for (const v of all) {
      const b = Math.floor(((v - lo) / (hi - lo)) * B)
      if (b >= 0 && b < B) bins[b]!++
    }
    const maxBin = Math.max(...bins)
    // expected counts under N(0, 0.08²): N · binWidth · pdf(center)
    const binW = (hi - lo) / B
    const sigma = 0.08
    const curve = Array.from({ length: B }, (_, i) => {
      const cx = lo + (i + 0.5) * binW
      const pdf = Math.exp(-(cx * cx) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI))
      return all.length * binW * pdf
    })
    return { bins, maxBin, curve }
  }, [])
  const W = 640
  const H = 120
  const bw = W / bins.length
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="histogram of the 4,192 initial parameter values with the sigma 0.08 gaussian curve">
      {bins.map((b, i) => (
        <rect
          key={i}
          x={i * bw + 1}
          y={H - (b / maxBin) * (H - 10)}
          width={bw - 2}
          height={(b / maxBin) * (H - 10)}
          fill="rgba(25,113,194,0.45)"
        />
      ))}
      <path
        d={curve.map((c, i) => `${i === 0 ? 'M' : 'L'} ${i * bw + bw / 2} ${H - (c / maxBin) * (H - 10)}`).join(' ')}
        fill="none"
        stroke="var(--pos)"
        strokeWidth={2}
      />
      <text x={W / 2} y={H - 2} textAnchor="middle" fontSize={9} className="font-mono" fill="var(--muted)">
        −0.3 … +0.3 · vermilion curve = N(0, 0.08²)
      </text>
    </svg>
  )
}

function MatrixExplorer() {
  const [selectedKey, setSelectedKey] = useState('wte')
  const step = useAppStore((s) => s.checkpointStep)
  const model = useModelAt(step)
  const m = MATS.find((x) => x.key === selectedKey)!
  const data = useMemo(() => {
    if (step === 0 || !model) return weights[m.key]!.flat()
    return [...model.sd[m.key]!.data]
  }, [m, model, step])
  return (
    <div className="not-prose my-4 space-y-3">
      <Treemap selected={selectedKey} onSelect={setSelectedKey} />
      <div className="flex flex-wrap items-start gap-4">
        <MatrixHeatmap
          data={data}
          rows={m.rows}
          cols={m.cols}
          label={m.key}
          cellSize={m.rows > 30 ? 7 : 11}
          rowLabel={m.rowLabel}
          colLabel={m.colLabel}
        />
        <div className="max-w-xs space-y-2">
          <p className="text-sm">{m.role}</p>
          <CompareToggle />
          <p className="text-xs text-muted">
            toggle to see the same numbers after 1000 training steps — chapter 9 shows the
            journey between
          </p>
        </div>
      </div>
    </div>
  )
}

const chapter = CHAPTERS[3]!

export default function Ch03() {
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Lines 80–89 allocate every number this model will ever learn:{' '}
        <strong>4,192</strong> of them, in nine matrices. Nothing else in the file
        changes during training — no hidden state, no growing memory. If the trained
        model &quot;knows&quot; that names often end in <em>a</em>, that knowledge is,
        physically, some of these numbers having particular values.
      </p>

      <h2>The whole budget, to scale</h2>
      <p>
        Click any block — areas are proportional to parameter count. Notice how much of
        the budget the MLP takes (2,048 of 4,192), and that <code>wte</code> and{' '}
        <code>lm_head</code> are two <em>separate</em> 27×16 matrices.
      </p>
      <MatrixExplorer />

      <h2>Born Gaussian</h2>
      <p>
        Line 80&apos;s <code>matrix</code> lambda fills every entry with{' '}
        <code>random.gauss(0, std)</code>, std = 0.08. Here is the actual histogram of
        the actual 4,192 initial values (the same ones the Python file draws with seed
        42), with the ideal bell curve on top:
      </p>
      <div className="not-prose my-4">
        <InitHistogram />
      </div>
      <p>
        Small and centered on zero: at the start, every projection is a faint random
        mixing, every prediction near-uniform. That is why the <Term t="loss">loss</Term>{' '}
        begins at ln 27 (chapter 7) — and everything that improves on it must be written
        into these matrices by <Term t="gradient">gradients</Term> (chapter 8).
      </p>

      <Aside kind="wild" title="GPT-2 ties wte and lm_head; microgpt doesn't">
        In GPT-2 the same matrix embeds tokens on the way in and scores them on the way
        out (weight tying) — it saves parameters and often helps. microgpt keeps them
        separate for simplicity: 432 parameters each, learned independently. Also missing
        relative to GPT-2: every bias vector (all linears here are pure matrix
        multiplies) and layernorm&apos;s learnable scale/shift. At GPT-2 scale the same
        blueprint holds ~124 million parameters — 30,000× this file, same nine roles.
      </Aside>

      <PredictReveal
        qid="ch3-wte-a"
        question={<>You want to inspect what the model learned about the letter &apos;a&apos;. Which parameters do you look at?</>}
        options={['row 0 of wte (and row 0 of lm_head)', 'column 0 of wpe', 'all of mlp_fc1']}
        answerIndex={0}
        explanation={
          <>
            &apos;a&apos; is token 0, so <code>wte[0]</code> is its 16-number
            representation going <em>in</em>, and <code>lm_head[0]</code> is the direction
            that scores &apos;a&apos; as the <em>next</em> token coming out. Select wte in
            the treemap and toggle trained — row 0 is visibly no longer noise. (Attention
            and MLP weights also shape how &apos;a&apos; behaves, but they&apos;re shared
            across all tokens.)
          </>
        }
      />
      <PredictReveal
        qid="ch3-double-embd"
        question={<>If n_embd doubled from 16 to 32, how many parameters would wte have?</>}
        options={['432 (unchanged)', '864', '1,728']}
        answerIndex={1}
        explanation={
          <>
            wte is vocab_size × n_embd = 27 × 32 = <strong>864</strong> — embeddings scale
            linearly with width. The MLP scales <em>quadratically</em> (fc1 becomes
            128×32 = 4,096), which is why width dominates the budget of big models.
          </>
        }
      />
      <PredictReveal
        qid="ch3-no-bias"
        question={<>How many bias parameters does this model have?</>}
        options={['one per linear layer (7)', 'one per neuron (155)', 'zero']}
        answerIndex={2}
        explanation={
          <>
            <strong>Zero.</strong> Line 93 says it up front: &quot;no biases&quot;. Every
            layer is a pure <code>matrix @ vector</code>; the file&apos;s{' '}
            <code>linear()</code> has no <code>+ b</code> term anywhere. 4,192 = 432 + 256
            + 432 + 4·256 + 1,024 + 1,024, all weights.
          </>
        }
      />

      <Recap
        chapterId={3}
        points={[
          <>4,192 parameters in nine matrices; the treemap is the entire memory of the model.</>,
          <>Initialization is Gaussian noise, σ = 0.08 — small, centered, meaningless until training writes into it.</>,
          <>wte row t is token t&apos;s input representation; lm_head row t is its output direction; they are separate (untied) here.</>,
        ]}
      />
    </ChapterFrame>
  )
}
