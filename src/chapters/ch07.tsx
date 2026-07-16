/**
 * Chapter 7 — Logits → probabilities → loss. Stable softmax (with the
 * shift-invariance slider), -log p(target), averaging over positions, and
 * the ln(27) cold-start fact — derived, verified live on a zero-weight model.
 */
import { useEffect, useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { NumericGuess, PredictReveal, Recap, TryIt } from '../components/Quiz.tsx'
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
import { labelOf, useTrace, VOCAB_LABELS } from '../app/useModel.ts'
import { softmaxProbs } from '../engine/tensor.ts'

function LossStepper({ onView }: { onView: (pos: number) => void }) {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)
  const player = useStepPlayer(trace?.n ?? 1, 1)
  const { setHighlight } = useCodeSync()
  useEffect(() => setHighlight([163, 164, 165, 166, 167, 168]), [setHighlight, player.index])
  useEffect(() => {
    if (trace) onView(Math.min(player.index, trace.n - 1))
  }, [trace, player.index, onView])

  if (!trace) return <p className="font-mono text-sm text-muted">loading…</p>
  const pos = Math.min(player.index, trace.n - 1)
  const call = trace.calls[pos]!
  const target = trace.tokens[pos + 1]!
  const p = trace.probs[pos]!
  return (
    <div className="not-prose my-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TokenTape tokens={trace.tokens} labelOf={labelOf} showIds={false} activeIndex={pos} showTargets />
        <CompareToggle />
      </div>
      <StepPlayer player={player} length={trace.n} label="loss position stepper" format={(i) => `pos ${i}`} />
      <div className="space-y-2 overflow-x-auto" onMouseEnter={() => setHighlight([143, 165])}>
        <VectorChips values={call.logits} label="logits (27)" cellSize={17} />
      </div>
      <div onMouseEnter={() => setHighlight([166])}>
        <div className="mb-1 font-mono text-xs text-muted">
          softmax(logits) — target &apos;{labelOf(target)}&apos; marked; p(target) ={' '}
          {fmt(p[target]!)}
        </div>
        <BarDistribution probs={p} labels={VOCAB_LABELS} marker={target} fullScale={false} height={120} />
      </div>
      <div className="rounded bg-ink/5 px-3 py-2 font-mono text-[13px]" onMouseEnter={() => setHighlight([167, 168, 169])}>
        loss_t = −log p(&apos;{labelOf(target)}&apos;) = −log({fmt(p[target]!)}) ={' '}
        <strong>{fmt(trace.lossT[pos]!)}</strong>
        <span className="text-muted">
          {' '}
          · mean over {trace.n} positions = {fmt(trace.loss)}
        </span>
      </div>
      <div className="flex items-end gap-1" role="img" aria-label="per-position losses">
        {trace.lossT.map((l, i) => (
          <div key={i} className="flex flex-col items-center gap-0.5">
            <span className="font-mono text-[9px] text-muted">{fmt(l, 2)}</span>
            <div
              className="w-7 rounded-t-sm"
              style={{ height: `${6 + l * 16}px`, background: i === pos ? 'var(--hot)' : 'rgba(22,24,29,0.35)' }}
            />
            <span className="font-mono text-[10px] text-muted">
              →{labelOf(trace.tokens[i + 1]!)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

/** Shift-invariance: add any constant to all logits, probs don't move. */
function ShiftDemo({ onShift }: { onShift: (shift: number) => void }) {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)
  const [shift, setShift] = useState(0)
  const { setHighlight } = useCodeSync()
  useEffect(() => onShift(shift), [shift, onShift])
  const shifted = useMemo(() => {
    if (!trace) return null
    const logits = trace.calls[0]!.logits.map((l) => l + shift)
    return softmaxProbs(logits)
  }, [trace, shift])
  if (!trace || !shifted) return null
  const orig = trace.probs[0]!
  let maxDiff = 0
  for (let i = 0; i < orig.length; i++) maxDiff = Math.max(maxDiff, Math.abs(orig[i]! - shifted[i]!))
  return (
    <div className="not-prose my-4 space-y-2 rounded-lg border border-ink/15 bg-white/60 p-4" onMouseEnter={() => setHighlight([97, 98, 99, 100, 101])}>
      <label htmlFor="shift" className="font-mono text-xs text-muted">
        add {shift >= 0 ? '+' : ''}
        {shift.toFixed(1)} to ALL 27 logits — max probability change: {maxDiff.toExponential(1)}
      </label>
      <input
        id="shift"
        type="range"
        min={-10}
        max={10}
        step={0.1}
        value={shift}
        onChange={(e) => setShift(Number(e.target.value))}
        className="w-full accent-[var(--neg)]"
      />
      <BarDistribution probs={shifted} labels={VOCAB_LABELS} fullScale={false} height={90} />
      <p className="text-xs text-muted">
        softmax only sees <em>differences</em> between logits — which is exactly why line
        98 may subtract the max with no effect on the result (except keeping{' '}
        <code>math.exp</code> from overflowing).
      </p>
    </div>
  )
}

const chapter = CHAPTERS[7]!

export default function Ch07() {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)
  const hardest = useMemo(() => {
    if (!trace) return null
    let best = 0
    for (let i = 1; i < trace.lossT.length; i++) if (trace.lossT[i]! > trace.lossT[best]!) best = i
    return best
  }, [trace])
  const [hardestFound, setHardestFound] = useState(false)
  const onView = useMemo(
    () => (pos: number) => {
      setHardestFound((f) => f || (hardest != null && pos === hardest))
    },
    [hardest],
  )
  const [bigShift, setBigShift] = useState(false)
  const onShift = useMemo(() => (s: number) => setBigShift((b) => b || Math.abs(s) >= 8), [])
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        The model&apos;s final act at each position is line 143:{' '}
        <code>logits = linear(x, lm_head)</code> — 27 raw scores, one per token. Scores
        are opinions; training needs a <em>grade</em>. Three moves turn opinions into one
        number that says exactly how wrong the model was: softmax, pick the target,
        −log.
      </p>

      <h2>Follow the grade for &quot;{example}&quot;</h2>
      <LossStepper onView={onView} />
      <p>
        Read the loss bar chart: the model isn&apos;t equally wrong everywhere. Early
        positions (first letters) are genuinely uncertain — many names could start this
        way — so their loss stays high even in a well-trained model. Later positions get
        easier.
      </p>

      <TryIt
        qid="ch7-hardest"
        task={<>Find the hardest prediction in &quot;{example}&quot;: scrub the stepper to the position with the tallest loss bar.</>}
        done={hardestFound}
        payoff={
          trace && hardest != null ? (
            <>
              The worst moment is predicting &apos;{labelOf(trace.tokens[hardest + 1]!)}&apos;
              at position {hardest}: p(target) is only {fmt(trace.probs[hardest]![trace.tokens[hardest + 1]!]!, 3)},
              so −log p = {fmt(trace.lossT[hardest]!, 3)}. Genuine uncertainty — several
              letters were plausible there — costs loss no matter how good the model gets.
              Averaging over positions (line 169) is how one name becomes one grade.
            </>
          ) : (
            <>Genuine uncertainty — several letters were plausible there — costs loss no matter how good the model gets.</>
          )
        }
      />
      <p>
        Softmax exponentiates each logit and normalizes:{' '}
        <Term t="softmax">probabilities</Term> that sum to 1. Line 98 first subtracts the
        largest logit from all of them. Why that&apos;s free:
      </p>
      <ShiftDemo onShift={onShift} />
      <TryIt
        qid="ch7-shift"
        task={<>Slam the slider to an extreme — add at least ±8 to all 27 logits — and read the &quot;max probability change&quot; readout.</>}
        done={bigShift}
        payoff={
          <>
            The probabilities moved by ~10⁻¹⁶ — floating-point dust. Softmax genuinely
            only sees the <em>differences</em> between logits, which is why line 98 can
            subtract the max for free. e⁸ ≈ 2981, though: without that subtraction,
            confident logits would overflow <code>math.exp</code> long before the math
            went wrong.
          </>
        }
      />
      <Aside kind="math" title="Stability, and one sneaky detail about gradients">
        <p>
          <K block tex="\mathrm{softmax}(z)_i = \frac{e^{z_i - c}}{\sum_j e^{z_j - c}} \quad\text{for any } c" />
          The c&apos;s cancel — algebraically invisible, numerically vital: with c = max(z),
          the biggest exponent is e⁰ = 1, so <code>math.exp</code> can never overflow.
          Sneaky detail: line 98 reads <code>val.data</code> — the raw float, not the
          Value. The subtraction is a <em>constant</em> to autograd; no gradient flows
          through the max. None needs to: the derivative of softmax w.r.t. a uniform shift
          is exactly zero, as your slider just demonstrated.
        </p>
      </Aside>

      <h2>−log: the honest scoring rule</h2>
      <p>
        <Term t="cross-entropy">Cross-entropy</Term> grades only the probability given to
        the <em>true</em> next token: confident-and-right ≈ 0 loss;
        confident-and-wrong → the loss explodes (−log 0.001 ≈ 6.9). Line 169 then averages
        over the document&apos;s positions:{' '}
        <code>loss = (1 / n) * sum(losses)</code> — so long and short names grade on the
        same scale.
      </p>

      <PredictReveal
        qid="ch7-ln27"
        question={<>Set every weight in the model to zero. What is the loss — exactly?</>}
        options={['exactly 0', 'exactly ln 27 ≈ 3.296', 'undefined — division by zero']}
        answerIndex={1}
        hint={<>Zero weights → all logits 0 → softmax of 27 equal scores gives…?</>}
        explanation={
          <>
            Zero weights → all 27 logits are 0 → softmax gives 1/27 everywhere → every
            position&apos;s loss is −log(1/27) = ln 27. This app&apos;s test suite
            actually runs a zero-weight model through the full pipeline and asserts
            ln 27 to twelve decimal places. The random σ=0.08 init lands <em>near</em>{' '}
            it (3.37 on the first doc) — that&apos;s your training-curve starting point.
          </>
        }
      />
      <NumericGuess
        qid="ch7-vocab100"
        question={<>Same file, but a dataset with 99 unique characters (vocab_size = 100). What does the initial loss become?</>}
        answer={Math.log(100)}
        tolerance={0.05}
        placeholder="loss"
        unit="≈"
        format={(v) => v.toFixed(2)}
        hint={<>Same rule as before: −log(1/vocab_size) = ln(vocab_size). A calculator (or the ² and log keys in your head) gets you there.</>}
        explanation={
          <>
            Uniform guessing over 100 tokens: −log(1/100) = ln 100 ≈{' '}
            <strong>4.61</strong>. The cold-start loss is a property of vocabulary size
            alone — which is why &quot;loss 3.3&quot; means nothing without knowing the
            tokenizer, and why papers report per-token perplexity instead.
          </>
        }
      />
      <PredictReveal
        qid="ch7-perfect"
        question={<>The model assigns p = 1.0 to the correct token at every position of a name. The name&apos;s loss is…</>}
        options={['1.0', '0', '−1']}
        answerIndex={1}
        hint={<>What is −log(1)?</>}
        explanation={
          <>
            −log(1) = 0, averaged over positions: still 0. Unreachable in practice — the
            first character after BOS is genuinely unpredictable — which is why the
            trained loss settles near 2.0–2.6, not 0. There is irreducible entropy in
            names; the model can only learn the structure, not the coin flips.
          </>
        }
      />

      <Recap
        chapterId={7}
        points={[
          <>logits (27 scores) → softmax (probabilities) → −log p(target) (one position&apos;s loss) → mean over the doc.</>,
          <>Softmax is shift-invariant; subtracting the max costs nothing and prevents overflow — and autograd treats it as a constant.</>,
          <>Cold start = ln(vocab_size): 3.296 here, 4.61 at vocab 100. Watch the real curve start there in chapter 9.</>,
        ]}
      />
    </ChapterFrame>
  )
}
