/**
 * Chapter 5 — Attention, the centerpiece. The real example stepped one token
 * at a time (q/k/v, the growing KV lists, per-head scores → softmax →
 * weighted sum), the incremental ⇄ matrix-with-mask equivalence toggle, and
 * free play over any string, trained vs untrained.
 */
import { useEffect, useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { K } from '../components/Katex.tsx'
import { CompareToggle } from '../components/Compare.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { StepPlayer, useStepPlayer } from '../components/StepPlayer.tsx'
import { TokenTape } from '../components/TokenTape.tsx'
import { VectorChips } from '../viz/VectorChips.tsx'
import { AttnMatrix } from '../viz/AttnMatrix.tsx'
import { fmt, seqColor } from '../viz/color.ts'
import { useAppStore } from '../app/store.ts'
import { labelOf, tokenizer, useTrace, type ExampleTrace } from '../app/useModel.ts'

const HEAD_NAMES = ['head 0', 'head 1', 'head 2', 'head 3']

/** q/k/v with the 4-per-head slicing made visible. */
function QkvRow({ label, values, line, onSync }: { label: string; values: number[]; line: number; onSync: (l: number[]) => void }) {
  return (
    <div className="flex items-center gap-2" onMouseEnter={() => onSync([line])}>
      <span className="w-8 shrink-0 text-right font-mono text-xs font-semibold">{label}</span>
      <div className="flex flex-wrap gap-2">
        {[0, 1, 2, 3].map((h) => (
          <div key={h} className="rounded border border-ink/10 p-0.5">
            <VectorChips values={values.slice(h * 4, h * 4 + 4)} cellSize={22} vmax={1.2} label={`${label}_h${h}`} />
          </div>
        ))}
      </div>
    </div>
  )
}

function AttentionStepper({ trace }: { trace: ExampleTrace }) {
  const player = useStepPlayer(trace.n, 0.8)
  const [head, setHead] = useState(0)
  const { setHighlight } = useCodeSync()
  const pos = Math.min(player.index, trace.n - 1)
  const call = trace.calls[pos]!
  const layer = call.layers[0]!
  const h = layer.heads[head]!

  useEffect(() => setHighlight([116, 117, 118, 119, 120, 121, 122]), [setHighlight, player.index])

  // the growing key cache: k vectors of every position up to pos
  const cachedKs = trace.calls.slice(0, pos + 1).map((c) => c.layers[0]!.k)

  return (
    <div className="not-prose my-4 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TokenTape tokens={trace.tokens.slice(0, trace.n)} labelOf={labelOf} showIds={false} activeIndex={pos} />
        <CompareToggle />
      </div>
      <StepPlayer
        player={player}
        length={trace.n}
        label="attention position stepper"
        format={(i) => `pos ${i} '${labelOf(trace.tokens[i]!)}'`}
      />

      <div className="max-w-full space-y-2 overflow-x-auto">
        <div className="font-mono text-xs text-muted">
          1 · three projections of the same normalized x — sliced into 4 heads of 4 dims
        </div>
        <QkvRow label="q" values={layer.q} line={118} onSync={setHighlight} />
        <QkvRow label="k" values={layer.k} line={119} onSync={setHighlight} />
        <QkvRow label="v" values={layer.v} line={120} onSync={setHighlight} />
      </div>

      <div className="max-w-full space-y-1 overflow-x-auto" onMouseEnter={() => setHighlight([121, 122, 127, 128])}>
        <div className="font-mono text-xs text-muted">
          2 · the cache after this position — keys[0] now holds {pos + 1} row{pos ? 's' : ''}; nothing
          from the future exists in it
        </div>
        <div className="space-y-0.5">
          {cachedKs.map((k, t) => (
            <div key={t} className="flex items-center gap-2" style={{ opacity: t === pos ? 1 : 0.75 }}>
              <span className="w-8 text-right font-mono text-[10px] text-muted">
                {t === pos ? '+ ' : ''}k[{t}]
              </span>
              <VectorChips values={k} cellSize={13} vmax={1.2} label={`keys[0][${t}]`} />
              <span className="font-mono text-[10px] text-muted">&apos;{labelOf(trace.tokens[t]!)}&apos;</span>
            </div>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex items-center gap-2 font-mono text-xs text-muted">
          <span>3 · one head&apos;s view —</span>
          <div role="radiogroup" aria-label="attention head" className="inline-flex rounded border border-ink/20 p-0.5">
            {HEAD_NAMES.map((name, i) => (
              <button
                key={i}
                type="button"
                role="radio"
                aria-checked={head === i}
                onClick={() => setHead(i)}
                className="rounded px-2 py-0.5 font-mono text-[11px]"
                style={{ background: head === i ? 'var(--ink)' : 'transparent', color: head === i ? 'var(--paper)' : 'var(--muted)' }}
              >
                {name}
              </button>
            ))}
          </div>
        </div>
        <div className="grid gap-3 md:grid-cols-2">
          <div className="space-y-2 rounded-lg border border-ink/10 p-3" onMouseEnter={() => setHighlight([129, 130])}>
            <div className="font-mono text-[11px] text-muted">
              attn_logits = q_h·k_h[t] / √{4} — then softmax
            </div>
            {h.attn_logits.map((l, t) => (
              <div key={t} className="flex items-center gap-2 font-mono text-[11px]">
                <span className="w-10 text-muted">
                  &apos;{labelOf(trace.tokens[t]!)}&apos; t={t}
                </span>
                <span className="w-14 text-right tabular-nums">{fmt(l, 3)}</span>
                <span className="text-muted">→</span>
                <div className="h-3.5 flex-1 rounded-sm bg-ink/5">
                  <div
                    className="h-full rounded-sm transition-[width] duration-300 motion-reduce:transition-none"
                    style={{ width: `${(h.attn_weights[t]! * 100).toFixed(1)}%`, background: seqColor(h.attn_weights[t]!) }}
                  />
                </div>
                <span className="w-12 text-right tabular-nums">{(h.attn_weights[t]! * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <div className="space-y-2 rounded-lg border border-ink/10 p-3" onMouseEnter={() => setHighlight([131, 132, 133, 134])}>
            <div className="font-mono text-[11px] text-muted">
              head_out = Σ_t weight[t] · v_h[t] — a blend of the values it attended to
            </div>
            <VectorChips values={h.head_out} cellSize={26} showValues vmax={1} label={`head${head}_out`} />
            <div className="font-mono text-[11px] text-muted">
              all 4 heads concatenated (x_attn), then mixed by attn_wo and added back to the
              residual:
            </div>
            <VectorChips values={layer.x_attn} cellSize={13} vmax={1.2} label="x_attn" />
            <VectorChips values={layer.x_after_attn} cellSize={13} vmax={2} label="x + resid" />
          </div>
        </div>
      </div>
    </div>
  )
}

/** The equivalence toggle: incremental (as the file runs) ⇄ matrix + mask. */
function MaskEquivalence({ trace }: { trace: ExampleTrace }) {
  const [mode, setMode] = useState<'incremental' | 'matrix'>('incremental')
  const [head, setHead] = useState(0)
  const { setHighlight } = useCodeSync()
  const n = trace.n
  const weights = trace.calls.map((c) => c.layers[0]!.heads[head]!.attn_weights)
  const cell = 30
  const pad = 22

  useEffect(() => {
    setHighlight(mode === 'incremental' ? [121, 122, 127, 128, 129] : [])
  }, [mode, setHighlight])

  return (
    <div className="not-prose my-4 space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <div role="radiogroup" aria-label="attention picture" className="inline-flex rounded-lg border border-ink/20 p-0.5">
          {(
            [
              ['incremental', 'as the file runs it'],
              ['matrix', 'textbook: matrix + mask'],
            ] as const
          ).map(([m, label]) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={mode === m}
              onClick={() => setMode(m)}
              className="rounded-md px-3 py-1 font-mono text-xs"
              style={{ background: mode === m ? 'var(--ink)' : 'transparent', color: mode === m ? 'var(--paper)' : 'var(--muted)' }}
            >
              {label}
            </button>
          ))}
        </div>
        <select
          value={head}
          onChange={(e) => setHead(Number(e.target.value))}
          className="rounded border border-ink/20 bg-white px-2 py-1 font-mono text-xs"
          aria-label="head"
        >
          {HEAD_NAMES.map((nm, i) => (
            <option key={i} value={i}>
              {nm}
            </option>
          ))}
        </select>
      </div>
      <div className="overflow-x-auto">
      <svg
        width={pad + n * cell}
        height={pad + n * cell}
        viewBox={`0 0 ${pad + n * cell} ${pad + n * cell}`}
        role="img"
        aria-label={mode === 'incremental' ? 'attention as computed incrementally: upper cells never exist' : 'attention as a full matrix with a causal mask'}
      >
        {Array.from({ length: n }, (_, i) => (
          <g key={i}>
            <text x={pad + i * cell + cell / 2} y={pad - 7} textAnchor="middle" fontSize={11} className="font-mono" fill="var(--muted)">
              {labelOf(trace.tokens[i]!)}
            </text>
            <text x={pad - 8} y={pad + i * cell + cell / 2 + 4} textAnchor="middle" fontSize={11} className="font-mono" fill="var(--muted)">
              {labelOf(trace.tokens[i]!)}
            </text>
          </g>
        ))}
        {Array.from({ length: n }, (_, r) =>
          Array.from({ length: n }, (_, c) => {
            if (c <= r) {
              const v = weights[r]![c]!
              return (
                <g key={`${r}-${c}`}>
                  <rect x={pad + c * cell} y={pad + r * cell} width={cell - 1} height={cell - 1} rx={2} fill={seqColor(v)} />
                  <text
                    x={pad + c * cell + (cell - 1) / 2}
                    y={pad + r * cell + (cell - 1) / 2 + 3}
                    textAnchor="middle"
                    fontSize={9}
                    className="pointer-events-none font-mono"
                    fill={v > 0.45 ? 'var(--paper)' : 'var(--ink)'}
                  >
                    {v >= 0.995 ? '1' : v.toFixed(2).replace(/^0/, '')}
                  </text>
                </g>
              )
            }
            // upper triangle: nonexistent vs masked
            return mode === 'matrix' ? (
              <g key={`${r}-${c}`}>
                <rect x={pad + c * cell} y={pad + r * cell} width={cell - 1} height={cell - 1} rx={2} fill="var(--ink)" opacity={0.85} />
                <text
                  x={pad + c * cell + (cell - 1) / 2}
                  y={pad + r * cell + (cell - 1) / 2 + 3}
                  textAnchor="middle"
                  fontSize={9}
                  className="pointer-events-none font-mono"
                  fill="rgba(250,249,247,0.75)"
                >
                  −∞
                </text>
              </g>
            ) : null
          }),
        )}
      </svg>
      </div>
      <p className="max-w-xl text-sm text-muted">
        {mode === 'incremental' ? (
          <>
            The file&apos;s picture: row t is computed at position t, when only t+1 keys
            exist. The upper triangle isn&apos;t masked — <strong>it was never
            computed</strong>. <code>k_h = [ki[…] for ki in keys[li]]</code> simply has
            nothing from the future in it.
          </>
        ) : (
          <>
            The textbook picture: compute all T×T scores at once, set the upper triangle
            to −∞ so softmax gives it zero weight. The surviving numbers are{' '}
            <strong>identical</strong> to the incremental ones — softmax over a masked row
            equals softmax over the shorter row. Two framings, one computation.
          </>
        )}
      </p>
    </div>
  )
}

function FreePlay() {
  const [word, setWord] = useState('alexandra')
  const step = useAppStore((s) => s.checkpointStep)
  const cleaned = useMemo(() => [...word.toLowerCase()].filter((c) => tokenizer.isInVocab(c)).join('').slice(0, 14), [word])
  const trace = useTrace(cleaned || 'emma', step)
  if (!trace) return null
  const tokens = trace.tokens.slice(0, trace.n).map((t) => labelOf(t))
  return (
    <div className="not-prose my-4 space-y-3 rounded-lg border border-ink/15 bg-white/60 p-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex-1 font-mono text-xs text-muted">
          any string over the vocabulary:
          <input
            value={word}
            onChange={(e) => setWord(e.target.value)}
            spellCheck={false}
            className="mt-1 w-full rounded border border-ink/20 bg-white px-3 py-2 font-mono focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
            aria-label="attention free play input"
          />
        </label>
        <CompareToggle />
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {HEAD_NAMES.map((name, h) => (
          <div key={h} className="overflow-x-auto">
            <div className="mb-1 font-mono text-xs text-muted">{name}</div>
            <AttnMatrix
              weights={trace.calls.map((c) => c.layers[0]!.heads[h]!.attn_weights)}
              tokens={tokens}
              cellSize={Math.min(30, Math.floor(360 / trace.n))}
            />
          </div>
        ))}
      </div>
      <p className="text-xs text-muted">
        trained heads often develop habits — one watching the previous token, one watching
        BOS, one scanning for vowels. untrained heads are fog. (This model is tiny; expect
        tendencies, not laws.)
      </p>
    </div>
  )
}

const chapter = CHAPTERS[5]!

export default function Ch05() {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)

  return (
    <ChapterFrame chapter={chapter}>
      <p>
        So far each position lives alone: &quot;m at position 2&quot; knows nothing about
        the e and m before it. Attention is where positions <em>meet</em>. Every position
        publishes a <strong>key</strong> (&quot;what I advertise&quot;), a{' '}
        <strong>value</strong> (&quot;what I hand over&quot;), and asks a question with
        its <strong>query</strong>. Scores between q and the existing k&apos;s become
        weights; the weights blend the v&apos;s. That&apos;s the whole trick.
      </p>

      <h2>Step the real thing</h2>
      <p>
        This is your example &quot;{example}&quot; flowing through the real weights, one
        position at a time — exactly the order the file computes it. Watch part 2: the{' '}
        <Term t="KV cache">KV cache</Term> <em>grows</em>. That growth is the fact around
        which this whole chapter orbits.
      </p>
      {trace ? <AttentionStepper trace={trace} /> : <p className="font-mono text-sm text-muted">loading…</p>}

      <Aside kind="math" title="The scaling: why divide by √head_dim">
        <p>
          A dot product of d independent ~unit terms has standard deviation ~<K tex="\sqrt{d}" />.
          Dividing by <K tex="\sqrt{d}" /> (here √4 = 2, line 129) keeps logits O(1) no
          matter the head size, so softmax starts in its sensitive regime instead of
          saturating — <K tex="\text{attn}_t = \frac{q\cdot k_t}{\sqrt{d}}" />. Remove it
          and logits double in spread: after softmax, winners take (nearly) all, and
          gradients through the losers vanish.
        </p>
      </Aside>

      <h2>Where&apos;s the mask? There is no mask</h2>
      <p>
        Every attention tutorial shows a T×T matrix with its upper triangle crossed out —
        the &quot;causal mask&quot; that stops positions from seeing the future. Search
        microgpt.py for it. It is not there. The file computes attention{' '}
        <em>incrementally</em>: at position t, the lists <code>keys[li]</code> and{' '}
        <code>values[li]</code> contain exactly t+1 entries.{' '}
        <strong>You cannot attend to a key that hasn&apos;t been appended yet.</strong>{' '}
        Causality isn&apos;t enforced; it&apos;s inherited from time itself. Toggle
        between the two pictures — the numbers don&apos;t change:
      </p>
      {trace ? <MaskEquivalence trace={trace} /> : null}

      <Aside kind="wild" title="The matrix formulation (what GPT-2 actually executes)">
        <p>
          Frameworks batch all positions into matrices:{' '}
          <K block tex="\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V" />
          where M has 0 on/below the diagonal and −∞ above. That&apos;s a{' '}
          <em>training-time throughput trick</em>: all rows at once on a GPU. At{' '}
          <em>inference</em> time, real systems do exactly what microgpt does — keep a
          growing KV cache and compute one new row per token. The file you&apos;re reading
          is written in inference form even during training; the mask picture and the
          cache picture are the same math wearing different clothes.
        </p>
      </Aside>

      <h2>Free play</h2>
      <FreePlay />

      <PredictReveal
        qid="ch5-pos0"
        question={<>At position 0, what are the attention weights — in every head, at any training step?</>}
        options={['[1.0] exactly', 'uniform over 16 slots', 'depends on the weights']}
        answerIndex={0}
        explanation={
          <>
            One key exists, so softmax runs over a single logit: e&#x2E31;/e&#x2E31; = 1,
            whatever the logit&apos;s value. The parameters are irrelevant. This is
            pinned as an engine test in this app — and you can verify it by scrubbing the
            stepper to pos 0.
          </>
        }
      />
      <PredictReveal
        qid="ch5-no-mask"
        question={<>Which line of microgpt.py implements the causal mask?</>}
        options={['line 129 (the scores)', 'line 130 (the softmax)', 'no line — the cache makes it unnecessary']}
        answerIndex={2}
        explanation={
          <>
            Lines 127–128 slice <code>keys[li]</code>/<code>values[li]</code>, which hold
            only positions ≤ t because line 121–122 appends one entry per call. The
            future isn&apos;t hidden — it doesn&apos;t exist yet. The −∞ mask in the
            textbook picture produces identical numbers by <em>removing</em> what this
            code never computes.
          </>
        }
      />
      <PredictReveal
        qid="ch5-no-scale"
        question={<>Delete the /√head_dim on line 129. What happens to the attention weights?</>}
        options={['they stop summing to 1', 'distributions get peakier — winners take nearly all', 'nothing — softmax normalizes anyway']}
        answerIndex={1}
        explanation={
          <>
            Softmax still normalizes (they always sum to 1) — but the <em>spread</em> of
            the logits doubles, and softmax is exponential in that spread: e² ≈ 7.4× ratio
            per 2 units of logit gap instead of e ≈ 2.7×. Sharper weights mean tiny
            gradients for every non-winner — harder training, same math.
          </>
        }
      />

      <Recap
        chapterId={5}
        points={[
          <>Attention = query·key scores over the cached past, softmaxed into weights, blending cached values — per head, on 4-dim slices.</>,
          <>The KV cache grows one entry per position; causality is implicit because future keys simply don&apos;t exist yet. No mask anywhere.</>,
          <>The textbook matrix+mask picture computes identical numbers — it&apos;s a batching optimization, not different math.</>,
        ]}
      />
    </ChapterFrame>
  )
}
