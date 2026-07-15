/**
 * Chapter 8 — Backward & Adam. The step-0 gradient field over the real
 * matrices (aggregate views — the full graph has ~10⁵ nodes and is run, not
 * drawn), the real m/v/m̂/v̂ numbers across the first three golden steps, and
 * an Adam-vs-SGD playground on a bumpy loss surface.
 */
import { useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { K } from '../components/Katex.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { MatrixHeatmap } from '../viz/MatrixHeatmap.tsx'
import { fmt } from '../viz/color.ts'
import { stateDictKeys } from '../engine/model_scalar.ts'
import step0State from '../../golden/step0_state.json'
import step1State from '../../golden/step1_state.json'
import step2State from '../../golden/step2_state.json'
import initWeights from '../../golden/init_weights.json'
import tokenizerGolden from '../../golden/tokenizer.json'

const SHAPES: Record<string, [number, number]> = {
  wte: [27, 16], wpe: [16, 16], lm_head: [27, 16],
  'layer0.attn_wq': [16, 16], 'layer0.attn_wk': [16, 16], 'layer0.attn_wv': [16, 16], 'layer0.attn_wo': [16, 16],
  'layer0.mlp_fc1': [64, 16], 'layer0.mlp_fc2': [16, 64],
}

/** flat param-vector offset of each matrix, in the file's order */
const OFFSETS = (() => {
  const out: Record<string, number> = {}
  let off = 0
  for (const key of stateDictKeys(1)) {
    out[key] = off
    off += SHAPES[key]![0] * SHAPES[key]![1]
  }
  return out
})()

const uchars: string[] = tokenizerGolden.uchars
const tokenName = (r: number) => (r === 26 ? "'·' BOS" : `'${uchars[r]}'`)

function GradField() {
  const [key, setKey] = useState('wte')
  const { setHighlight } = useCodeSync()
  const [rows, cols] = SHAPES[key]!
  const grads = useMemo(() => step0State.grads.slice(OFFSETS[key]!, OFFSETS[key]! + rows * cols), [key, rows, cols])
  const nonzero = useMemo(() => step0State.grads.filter((g) => g !== 0).length, [])
  return (
    <div className="not-prose my-4 space-y-3" onMouseEnter={() => setHighlight([171, 172])}>
      <div className="flex flex-wrap items-center gap-2">
        <select
          value={key}
          onChange={(e) => setKey(e.target.value)}
          className="rounded border border-ink/20 bg-white px-2 py-1 font-mono text-xs"
          aria-label="matrix whose gradients to display"
        >
          {stateDictKeys(1).map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
        <span className="font-mono text-xs text-muted">
          ∂loss/∂{key} after step 0&apos;s backward() on &quot;{step0State.doc}&quot; —{' '}
          {nonzero.toLocaleString()} of 4,192 grads are nonzero
        </span>
      </div>
      <MatrixHeatmap
        data={grads}
        rows={rows}
        cols={cols}
        label={`∂loss/∂${key}`}
        cellSize={rows > 30 ? 7 : 12}
        rowLabel={key === 'wte' || key === 'lm_head' ? tokenName : (r) => (key === 'wpe' ? `position ${r}` : `row ${r}`)}
        colLabel={(c) => `dim ${c}`}
      />
      {key === 'wte' && (
        <p className="max-w-xl text-sm text-muted">
          Entire rows are exactly zero: &quot;{step0State.doc}&quot; contains only{' '}
          {[...new Set(step0State.doc)].sort().map((c) => `'${c}'`).join(', ')} (+BOS), so
          only those embeddings took part in the forward pass — no participation, no
          gradient. Compare <code>lm_head</code>: every row is nonzero, because softmax
          spreads a little probability (and therefore a little blame) over all 27 tokens
          at every position.
        </p>
      )}
      {key === 'wpe' && (
        <p className="max-w-xl text-sm text-muted">
          Rows 7–15 are exactly zero: &quot;{step0State.doc}&quot; has n = {step0State.n}{' '}
          positions, so later position embeddings never entered the graph.
        </p>
      )}
    </div>
  )
}

/** The real optimizer numbers for one real parameter across golden steps 0–2. */
function AdamInspector() {
  const { setHighlight } = useCodeSync()
  const PARAMS = useMemo(() => {
    const yIdx = uchars.indexOf('y')
    const aIdx = uchars.indexOf('a')
    return [
      { label: `wte['y'][0] — used at step 0`, idx: OFFSETS['wte']! + yIdx * 16 },
      { label: `wte['a'][0] — absent from "${step0State.doc}"`, idx: OFFSETS['wte']! + aIdx * 16 },
      { label: `lm_head['a'][0] — blamed at every position`, idx: OFFSETS['lm_head']! + aIdx * 16 },
      { label: `mlp_fc1[0][0]`, idx: OFFSETS['layer0.mlp_fc1']! },
    ]
  }, [])
  const [sel, setSel] = useState(0)
  const idx = PARAMS[sel]!.idx
  const states = [step0State, step1State, step2State]
  const initFlat = useMemo(() => {
    const flat: number[] = []
    for (const key of stateDictKeys(1)) for (const row of (initWeights as Record<string, number[][]>)[key]!) flat.push(...row)
    return flat
  }, [])

  return (
    <div className="not-prose my-4 space-y-2" onMouseEnter={() => setHighlight([174, 175, 176, 177, 178, 179, 180, 181, 182])}>
      <select
        value={sel}
        onChange={(e) => setSel(Number(e.target.value))}
        className="rounded border border-ink/20 bg-white px-2 py-1 font-mono text-xs"
        aria-label="which real parameter to inspect"
      >
        {PARAMS.map((p, i) => (
          <option key={i} value={i}>
            {p.label}
          </option>
        ))}
      </select>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse font-mono text-xs">
          <thead>
            <tr className="border-b border-ink/20 text-left text-muted">
              <th className="py-1 pr-3">step</th>
              <th className="py-1 pr-3">doc</th>
              <th className="py-1 pr-3">grad</th>
              <th className="py-1 pr-3">m</th>
              <th className="py-1 pr-3">v</th>
              <th className="py-1 pr-3">m̂ (corrected)</th>
              <th className="py-1 pr-3">lr_t</th>
              <th className="py-1 pr-3">p after update</th>
            </tr>
          </thead>
          <tbody>
            {states.map((st, k) => {
              const mHat = st.m[idx]! / (1 - 0.85 ** (k + 1))
              const prev = k === 0 ? initFlat[idx]! : states[k - 1]!.params[idx]!
              return (
                <tr key={k} className="border-b border-ink/10">
                  <td className="py-1 pr-3">{k}</td>
                  <td className="py-1 pr-3">{st.doc}</td>
                  <td className="py-1 pr-3">{fmt(st.grads[idx]!, 5)}</td>
                  <td className="py-1 pr-3">{fmt(st.m[idx]!, 5)}</td>
                  <td className="py-1 pr-3">{st.v[idx]!.toExponential(2)}</td>
                  <td className="py-1 pr-3" style={{ color: 'var(--neg)' }}>
                    {fmt(mHat, 5)}
                  </td>
                  <td className="py-1 pr-3">{st.lr_t.toFixed(5)}</td>
                  <td className="py-1 pr-3">
                    {fmt(st.params[idx]!, 5)}{' '}
                    <span className="text-muted">({st.params[idx]! - prev >= 0 ? '+' : ''}{fmt(st.params[idx]! - prev, 5)})</span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-muted">
        these are the actual numbers from the reference run (golden dumps), not a
        simulation — pick the &apos;a&apos; embedding to see a parameter sit out a step
        (grad 0) while its m and v quietly decay
      </p>
    </div>
  )
}

/** Adam vs SGD on a bumpy 1-D loss surface. */
function AdamPlayground() {
  const [lr, setLr] = useState(0.15)
  const [beta1, setBeta1] = useState(0.85)
  const [beta2, setBeta2] = useState(0.99)
  const [decay, setDecay] = useState(true)
  const { setHighlight } = useCodeSync()

  const f = (w: number) => 0.4 * (w - 1.5) ** 2 + 0.35 * Math.sin(4 * w) + 0.6
  const df = (w: number) => 0.8 * (w - 1.5) + 1.4 * Math.cos(4 * w)

  const N = 60
  const run = useMemo(() => {
    let wA = -2.2
    let m = 0
    let v = 0
    let wS = -2.2
    const adam: number[] = [wA]
    const sgd: number[] = [wS]
    for (let t = 0; t < N; t++) {
      const lrT = decay ? lr * (1 - t / N) : lr
      const g = df(wA)
      m = beta1 * m + (1 - beta1) * g
      v = beta2 * v + (1 - beta2) * g * g
      const mHat = m / (1 - beta1 ** (t + 1))
      const vHat = v / (1 - beta2 ** (t + 1))
      wA -= (lrT * mHat) / (Math.sqrt(vHat) + 1e-8)
      adam.push(wA)
      wS -= lrT * df(wS)
      sgd.push(wS)
    }
    return { adam, sgd }
  }, [lr, beta1, beta2, decay])

  const W = 620
  const H = 200
  const xr = (w: number) => ((w + 3.2) / 6.4) * W
  const yr = (l: number) => H - 14 - ((l - 0) / 4.2) * (H - 28)
  const curve = Array.from({ length: 200 }, (_, i) => {
    const w = -3.2 + (i / 199) * 6.4
    return `${i === 0 ? 'M' : 'L'} ${xr(w).toFixed(1)} ${yr(f(w)).toFixed(1)}`
  }).join(' ')

  return (
    <div className="not-prose my-4 space-y-3 rounded-lg border border-ink/15 bg-white/60 p-4" onMouseEnter={() => setHighlight([147, 175, 176, 177, 178, 179, 180, 181])}>
      <div className="flex flex-wrap gap-4 font-mono text-xs">
        <label>
          lr {lr.toFixed(2)}
          <input type="range" min={0.01} max={0.5} step={0.01} value={lr} onChange={(e) => setLr(Number(e.target.value))} className="ml-2 w-24 accent-[var(--neg)]" />
        </label>
        <label>
          β₁ {beta1.toFixed(2)}
          <input type="range" min={0} max={0.99} step={0.01} value={beta1} onChange={(e) => setBeta1(Number(e.target.value))} className="ml-2 w-24 accent-[var(--neg)]" />
        </label>
        <label>
          β₂ {beta2.toFixed(2)}
          <input type="range" min={0.5} max={0.999} step={0.001} value={beta2} onChange={(e) => setBeta2(Number(e.target.value))} className="ml-2 w-24 accent-[var(--neg)]" />
        </label>
        <label className="flex items-center gap-1">
          <input type="checkbox" checked={decay} onChange={(e) => setDecay(e.target.checked)} className="accent-[var(--neg)]" />
          linear lr decay
        </label>
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Adam and SGD trajectories descending a bumpy loss curve">
        <path d={curve} fill="none" stroke="rgba(22,24,29,0.4)" strokeWidth={1.5} />
        {run.sgd.map((w, i) => (
          <circle key={`s${i}`} cx={xr(w)} cy={yr(f(w))} r={2.5} fill="rgba(22,24,29,0.3)" />
        ))}
        {run.adam.map((w, i) => (
          <circle key={`a${i}`} cx={xr(w)} cy={yr(f(w))} r={2.5} fill="var(--pos)" opacity={0.4 + (0.6 * i) / run.adam.length} />
        ))}
        <text x={8} y={14} fontSize={10} className="font-mono" fill="var(--pos)">
          ● Adam (ends at w = {fmt(run.adam[run.adam.length - 1]!, 3)})
        </text>
        <text x={8} y={28} fontSize={10} className="font-mono" fill="var(--muted)">
          ● plain SGD, same lr (ends at w = {fmt(run.sgd[run.sgd.length - 1]!, 3)})
        </text>
      </svg>
      <p className="text-xs text-muted">
        {N} steps from w = −2.2 on a bowl with ripples. Adam&apos;s v normalizes step
        sizes (big where gradients are small and steady, careful where they thrash), and
        β₁ momentum carries it across the little dips that trap SGD. Try β₁ = 0 or lr
        tiny — watch it behave like SGD again.
      </p>
    </div>
  )
}

const chapter = CHAPTERS[8]!

export default function Ch08() {
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Chapter 2 walked <code>backward()</code> through a five-node graph. Line 172 runs
        the <em>same two-phase walk</em> — topological sort, then{' '}
        <code>child.grad += local_grad · v.grad</code> — over the graph a whole document
        builds: roughly a hundred thousand Value nodes. We won&apos;t draw that (nobody
        should); we&apos;ll look at what lands in the 4,192 slots that matter.
      </p>

      <h2>The gradient field, step 0</h2>
      <GradField />

      <h2>Adam: gradient in, sized step out</h2>
      <p>
        Raw gradients make miserable step sizes. Lines 174–182 run{' '}
        <Term t="Adam">Adam</Term>: for every parameter, keep a running mean of its
        gradient (m, &quot;which way, on average?&quot;) and of its square (v, &quot;how
        big, typically?&quot;), then step by m̂/√v̂ — direction from the average, size
        normalized by the typical magnitude.
      </p>
      <AdamInspector />

      <Aside kind="math" title="Bias correction: why divide by (1 − βᵗ⁺¹)">
        <p>
          m starts at 0, so after one step m = (1−β₁)g = 0.15g — an underestimate purely
          because the buffer is young. Dividing by <K tex="1 - \beta_1^{t+1}" /> (= 0.15
          at t = 0) rescales it to exactly g. As t grows the correction fades to 1. Same
          for v with β₂. It&apos;s bookkeeping, not magic:{' '}
          <K block tex="\hat{m} = \frac{m_t}{1-\beta_1^{t+1}},\qquad \hat{v} = \frac{v_t}{1-\beta_2^{t+1}},\qquad p \mathrel{-}= \mathrm{lr}_t\,\frac{\hat m}{\sqrt{\hat v} + 10^{-8}}" />
        </p>
      </Aside>

      <h2>Feel the hyperparameters</h2>
      <AdamPlayground />

      <Aside kind="wild" title="The very same Adam trains the big ones">
        AdamW — Adam plus decoupled weight decay — is the default optimizer for
        essentially every large language model. The betas you just dragged are real knobs
        from real training runs (β₁ = 0.9, β₂ = 0.95 are typical at scale; microgpt&apos;s
        0.85/0.99 suit its tiny, noisy one-doc batches). Linear decay to zero, line 175,
        is one of the two standard schedules (the other being cosine).
      </Aside>

      <PredictReveal
        qid="ch8-sparse"
        question={<>At step 0, only 3,728 of 4,192 gradients are nonzero. Where are the 464 zeros?</>}
        options={['numerical underflow', "wte rows of letters not in the doc + wpe rows past the doc's length", 'the attention matrices']}
        answerIndex={1}
        explanation={
          <>
            Gradient only flows through nodes that participated in the forward pass.
            &quot;yuheng&quot; touches 6 letters + BOS → 20 unused wte rows (320 params),
            and n = 7 → 9 unused wpe rows (144 params). 320 + 144 = 464. You can count
            the blank rows in the heatmap above.
          </>
        }
      />
      <PredictReveal
        qid="ch8-mhat"
        question={<>At step 0 (t = 0), m = 0.15·g. What is m̂ after bias correction?</>}
        options={['0.15·g', 'exactly g', '6.67·g²']}
        answerIndex={1}
        explanation={
          <>
            m̂ = m / (1 − 0.85¹) = 0.15g / 0.15 = <strong>g</strong>. On the very first
            step, bias-corrected Adam trusts the raw gradient completely — check the m̂
            column in the real-numbers table above.
          </>
        }
      />
      <PredictReveal
        qid="ch8-lr500"
        question={<>What is the learning rate at step 500 of 1000?</>}
        options={['0.01', '0.005', '0.0005']}
        answerIndex={1}
        explanation={
          <>
            lr_t = 0.01 · (1 − 500/1000) = <strong>0.005</strong> — line 175&apos;s linear
            decay, halfway down its ramp to zero. Early steps take bold strides while the
            weights are noise; late steps only fine-tune.
          </>
        }
      />

      <Recap
        chapterId={8}
        points={[
          <>backward() at full scale is chapter 2&apos;s walk over ~10⁵ nodes; only what participated gets gradient.</>,
          <>Adam = per-parameter momentum (m) + magnitude normalization (v) + bias correction + linear lr decay.</>,
          <>The m̂/v̂/update numbers shown here are the reference run&apos;s own, dumped at steps 0–2 and replayed exactly.</>,
        ]}
      />
    </ChapterFrame>
  )
}
