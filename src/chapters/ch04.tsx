/**
 * Chapter 4 — Embeddings. wte[token] + wpe[pos], the non-redundant rmsnorm,
 * and the flagship manipulation: nudge one embedding weight and watch the
 * change ripple downstream (but never upstream — causality made visible).
 */
import { useEffect, useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap, TryIt } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { K } from '../components/Katex.tsx'
import { CompareToggle } from '../components/Compare.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { StepPlayer, useStepPlayer } from '../components/StepPlayer.tsx'
import { TokenTape } from '../components/TokenTape.tsx'
import { VectorChips } from '../viz/VectorChips.tsx'
import { BarDistribution } from '../viz/BarDistribution.tsx'
import { fmt } from '../viz/color.ts'
import { useAppStore } from '../app/store.ts'
import { labelOf, traceWord, useTrace, VOCAB_LABELS } from '../app/useModel.ts'
import { Model } from '../engine/model.ts'

function EmbeddingStepper({ onAllVisited }: { onAllVisited: () => void }) {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)
  const player = useStepPlayer(trace?.n ?? 1, 1)
  const { setHighlight } = useCodeSync()
  const [visited, setVisited] = useState<ReadonlySet<number>>(new Set([0]))

  useEffect(() => setHighlight([109, 110, 111, 112]), [setHighlight, player.index])
  useEffect(() => {
    setVisited((s) => (s.has(player.index) ? s : new Set(s).add(player.index)))
  }, [player.index])
  useEffect(() => {
    if (trace && visited.size >= trace.n) onAllVisited()
  }, [trace, visited, onAllVisited])

  if (!trace) return <p className="font-mono text-sm text-muted">loading the precomputed run…</p>
  const call = trace.calls[Math.min(player.index, trace.n - 1)]!
  return (
    <div className="not-prose my-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TokenTape tokens={trace.tokens} labelOf={labelOf} showIds={false} activeIndex={call.pos_id} />
        <CompareToggle />
      </div>
      <StepPlayer
        player={player}
        length={trace.n}
        label="position stepper"
        format={(i) => `pos ${i} '${labelOf(trace.tokens[i]!)}'`}
      />
      <div className="space-y-1.5 overflow-x-auto py-1">
        <VectorChips values={call.tok_emb} label={`wte[${call.token_id}]`} />
        <VectorChips values={call.pos_emb} label={`wpe[${call.pos_id}]`} />
        <div className="border-t border-ink/10" />
        <VectorChips values={call.x_emb_sum} label="x = t + p" />
        <VectorChips values={call.x_emb_norm} label="rmsnorm(x)" />
      </div>
      <p className="font-mono text-xs text-muted">
        wte[{call.token_id}] is &apos;{labelOf(call.token_id)}&apos;&apos;s vector — same at every position
        it appears; wpe[{call.pos_id}] belongs to the slot. Their sum is where the two
        streams of identity (&quot;which character&quot; + &quot;where&quot;) merge.
      </p>
    </div>
  )
}

function RmsnormDemo() {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)
  const [k, setK] = useState(1)
  const { setHighlight } = useCodeSync()
  if (!trace) return null
  const x = trace.calls[0]!.x_emb_sum
  const scaled = x.map((v) => v * k)
  const ms = scaled.reduce((a, b) => a + b * b, 0) / scaled.length
  const scale = (ms + 1e-5) ** -0.5
  const out = scaled.map((v) => v * scale)
  return (
    <div className="not-prose my-4 space-y-2 rounded-lg border border-ink/15 bg-white/60 p-4" onMouseEnter={() => setHighlight([103, 104, 105, 106])}>
      <label htmlFor="rms-k" className="font-mono text-xs text-muted">
        multiply the input vector by k = {k.toFixed(2)} and watch rmsnorm shrug it off:
      </label>
      <input
        id="rms-k"
        type="range"
        min={0.25}
        max={4}
        step={0.05}
        value={k}
        onChange={(e) => setK(Number(e.target.value))}
        className="w-full accent-[var(--neg)]"
      />
      <div className="space-y-1.5 overflow-x-auto">
        <VectorChips values={scaled} label={`x · ${k.toFixed(2)}`} vmax={2} />
        <VectorChips values={out} label="rmsnorm" vmax={2} />
      </div>
      <p className="font-mono text-xs text-muted">
        ms = {fmt(ms)} → scale = (ms + 1e-5)^-0.5 = {fmt(scale)} — the output&apos;s
        root-mean-square is pinned to ≈ 1 no matter what k does
      </p>
    </div>
  )
}

function PerturbLab({ onEditAt }: { onEditAt: (pos: number, n: number) => void }) {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)
  const [dim, setDim] = useState(3)
  const [delta, setDelta] = useState(0.8)
  const [posIdx, setPosIdx] = useState(1)
  const { setHighlight } = useCodeSync()

  useEffect(() => {
    if (trace) onEditAt(Math.min(posIdx, trace.n - 1), trace.n)
  }, [trace, posIdx, onEditAt])

  const perturbed = useMemo(() => {
    if (!trace) return null
    const pos = Math.min(posIdx, trace.n - 1)
    const token = trace.tokens[pos]!
    const flat = trace.model.flatParams()
    flat[token * 16 + dim] = flat[token * 16 + dim]! + delta // wte is the first matrix in param order
    return traceWord(Model.fromFlat(trace.model.cfg, flat), trace.word)
  }, [trace, dim, delta, posIdx])

  if (!trace || !perturbed) return null
  const pos = Math.min(posIdx, trace.n - 1)
  const token = trace.tokens[pos]!
  const ripple = trace.probs.map((p, i) => {
    let d = 0
    for (let v = 0; v < p.length; v++) d += Math.abs(perturbed.probs[i]![v]! - p[v]!)
    return d
  })
  const maxRipple = Math.max(...ripple, 1e-9)

  return (
    <div className="not-prose my-4 space-y-3 rounded-lg border border-ink/15 bg-white/60 p-4" onMouseEnter={() => setHighlight([109])}>
      <div className="flex flex-wrap items-end gap-4">
        <label className="font-mono text-xs text-muted">
          position
          <select
            value={pos}
            onChange={(e) => setPosIdx(Number(e.target.value))}
            className="ml-2 rounded border border-ink/20 bg-white px-2 py-1 font-mono text-sm"
            aria-label="which position's token embedding to edit"
          >
            {trace.calls.map((c) => (
              <option key={c.pos_id} value={c.pos_id}>
                {c.pos_id}: &apos;{labelOf(c.token_id)}&apos;
              </option>
            ))}
          </select>
        </label>
        <label className="font-mono text-xs text-muted">
          dimension {dim}
          <input
            type="range"
            min={0}
            max={15}
            value={dim}
            onChange={(e) => setDim(Number(e.target.value))}
            className="ml-2 w-28 accent-[var(--neg)]"
            aria-label="which of the 16 dimensions to edit"
          />
        </label>
        <label className="font-mono text-xs text-muted">
          nudge {delta >= 0 ? '+' : ''}
          {delta.toFixed(2)}
          <input
            type="range"
            min={-1.5}
            max={1.5}
            step={0.05}
            value={delta}
            onChange={(e) => setDelta(Number(e.target.value))}
            className="ml-2 w-28 accent-[var(--pos)]"
            aria-label="how much to add to the selected weight"
          />
        </label>
      </div>
      <p className="font-mono text-xs">
        editing <span style={{ color: 'var(--pos)' }}>wte[{token}][{dim}]</span> — the
        &apos;{labelOf(token)}&apos; embedding, dimension {dim}: {fmt(trace.calls[pos]!.tok_emb[dim]!)} →{' '}
        {fmt(trace.calls[pos]!.tok_emb[dim]! + delta)}
      </p>
      <div>
        <div className="mb-1 font-mono text-xs text-muted">
          next-token distribution at pos {pos} — outline: original, solid: perturbed
        </div>
        <div className="relative">
          <BarDistribution probs={perturbed.probs[pos]!} labels={VOCAB_LABELS} height={110} fullScale={false} />
        </div>
      </div>
      <div>
        <div className="mb-1 font-mono text-xs text-muted">
          how much every position&apos;s prediction changed (Σ |Δp|): the ripple flows{' '}
          <em>forward only</em>
        </div>
        <div className="flex items-end gap-1" role="img" aria-label="per-position change caused by the edit">
          {ripple.map((r, i) => (
            <div key={i} className="flex flex-col items-center gap-0.5">
              <div
                className="w-7 rounded-t-sm"
                style={{
                  height: `${4 + (r / maxRipple) * 48}px`,
                  background: i < pos ? 'rgba(22,24,29,0.15)' : 'var(--pos)',
                }}
                title={`pos ${i}: Σ|Δp| = ${fmt(r)}`}
              />
              <span className="font-mono text-[10px] text-muted">{labelOf(trace.tokens[i]!)}</span>
            </div>
          ))}
        </div>
        <p className="mt-1 font-mono text-xs text-muted">
          positions before {pos} are exactly zero: they ran before this token existed in
          the KV cache. Chapter 5 explains why.
        </p>
      </div>
    </div>
  )
}

const chapter = CHAPTERS[4]!

export default function Ch04() {
  const example = useAppStore((s) => s.example)
  const [allVisited, setAllVisited] = useState(false)
  const [editedLast, setEditedLast] = useState(false)
  const onAllVisited = useMemo(() => () => setAllVisited(true), [])
  const onEditAt = useMemo(() => (pos: number, n: number) => {
    if (n >= 2 && pos === n - 1) setEditedLast(true)
  }, [])
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Chapter 1 turned &quot;{example}&quot; into integers. But you can&apos;t do calculus
        on an integer. Lines 109–111 trade each token id for something differentiable: a
        learned 16-number vector — <code>wte[token_id]</code> — plus a second vector for{' '}
        <em>where it sits</em> — <code>wpe[pos_id]</code>. Their sum is the model&apos;s
        entire working state, x.
      </p>

      <h2>Watch a word become vectors</h2>
      <EmbeddingStepper onAllVisited={onAllVisited} />
      <p>
        Step through the positions: repeated letters (like the two m&apos;s in emma) reuse
        the <em>same</em> <code>wte</code> row but get different <code>wpe</code> rows —
        that&apos;s the only way the model can tell them apart. Toggle
        untrained/trained: at step 0 both matrices are σ=0.08 noise; after 1000 steps the
        rows have structure.
      </p>

      <TryIt
        qid="ch4-visit-all"
        task={<>Walk &quot;{example}&quot; all the way through: visit every position on the stepper and watch which of the two vectors changes at each step.</>}
        done={allVisited}
        payoff={
          <>
            The pattern you just saw: <code>wte</code> only changes when the <em>letter</em>{' '}
            changes, <code>wpe</code> changes at <em>every</em> step. Repeated letters get
            identical tok_emb rows — the position vector is the only thing telling the
            model &quot;this is the second one.&quot;
          </>
        }
      />
      <PredictReveal
        qid="ch4-same-token"
        question={<>&quot;emma&quot; has two m&apos;s (positions 2 and 3). Do they enter the model as the same vector x?</>}
        options={['yes — same token, same x', 'no — same wte row, different wpe rows', 'no — wte differs per position']}
        answerIndex={1}
        hint={<>x is a <em>sum</em> of two lookups. Which of the two indices differs between the m&apos;s?</>}
        explanation={
          <>
            Both m&apos;s share <code>wte[12]</code>, but position 2 adds{' '}
            <code>wpe[2]</code> and position 3 adds <code>wpe[3]</code>. Scrub the stepper
            above between them and watch tok_emb stay frozen while pos_emb changes.
            Without wpe, the model literally could not distinguish &quot;mm&quot; from
            &quot;m&quot;.
          </>
        }
      />

      <h2>rmsnorm: the volume knob</h2>
      <p>
        Line 112 immediately renormalizes x. <Term t="rmsnorm">rmsnorm</Term> divides the
        vector by its root-mean-square, pinning its typical size to ≈1 regardless of
        input scale:
      </p>
      <RmsnormDemo />
      <PredictReveal
        qid="ch4-rms-scale"
        question={<>If x is multiplied by 10, rmsnorm(10x) is…</>}
        options={['10× rmsnorm(x)', 'essentially identical to rmsnorm(x)', 'all zeros']}
        answerIndex={1}
        hint={<>You have the slider right above — drag k and watch the output row.</>}
        explanation={
          <>
            ms grows 100×, so scale = (ms+1e-5)^-0.5 shrinks 10×, cancelling exactly (the
            +1e-5 makes it &quot;essentially&quot; rather than &quot;exactly&quot;). You
            just verified it on the slider. Direction survives; magnitude is discarded.
          </>
        }
      />
      <Aside kind="math" title="Why the rmsnorm after the embedding sum isn't redundant">
        <p>
          A second rmsnorm follows at line 117 (start of the attention block), so this one
          looks useless — normalize twice, same result. But look at the data flow: line
          117&apos;s rmsnorm feeds only the attention branch, while <em>this</em> x is
          also what the <Term t="residual stream">residual stream</Term> carries forward
          to the additions at lines 134 and 141. Remove line 112 and the raw, unnormalized
          sum rides the residual all the way to lm_head. The comment on line 112 says
          exactly this: &quot;not redundant due to the backward pass via the residual
          connection.&quot; In gradient terms:{' '}
          <K tex="\tfrac{\partial \text{loss}}{\partial \text{wte}}" /> has a path that
          bypasses line 117 entirely.
        </p>
      </Aside>

      <h2>Poke it: edit one of the 4,192</h2>
      <p>
        The definition of &quot;every number matters&quot;: pick one weight of one token
        embedding and nudge it. The whole forward pass below re-runs live (it&apos;s a few
        microseconds), so the distribution you see is real, not simulated:
      </p>
      <PerturbLab onEditAt={onEditAt} />

      <TryIt
        qid="ch4-edit-last"
        task={<>Move the edit to the <em>last</em> position of &quot;{example}&quot; and watch the ripple chart.</>}
        done={editedLast}
        payoff={
          <>
            The ripple collapsed to a single bar: an edit at the last position can only
            change the last prediction, because every earlier position had already finished
            computing before this token existed. Causality isn&apos;t a rule the model
            checks — it&apos;s the arrow of time in the computation itself.
          </>
        }
      />
      <PredictReveal
        qid="ch4-ripple"
        question={<>You edit the embedding of the token at position 2. Which positions&apos; predictions can change?</>}
        options={['all of them', 'only position 2', 'positions 2 and later']}
        answerIndex={2}
        hint={<>Look at the gray bars in the lab. Which side of the edit are they always on?</>}
        explanation={
          <>
            The edit changes x at position 2, which changes the k and v that position
            appends to the cache — so every <em>later</em> position that attends to it
            shifts too. Positions 0 and 1 finished before the edit&apos;s token ever
            appeared: their bars above are exactly 0. You&apos;ve just seen causality —
            chapter 5 shows the mechanism.
          </>
        }
      />

      <Aside kind="wild" title="Real GPTs: same two tables, plus better position tricks">
        GPT-2 does literally this — wte + wpe, learned, added. Modern models mostly
        replace learned absolute positions with rotary embeddings (RoPE), which encode{' '}
        <em>relative</em> offsets directly inside attention. The token-embedding table
        survives at every scale: GPT-2&apos;s wte is 50,257 × 768.
      </Aside>

      <Recap
        chapterId={4}
        points={[
          <>x = wte[token] + wpe[pos]: what + where, merged into one 16-dim state.</>,
          <>rmsnorm pins the scale of x; the one at line 112 matters because the residual stream bypasses the next one.</>,
          <>One weight edit ripples through every later position — never backwards.</>,
        ]}
      />
    </ChapterFrame>
  )
}
