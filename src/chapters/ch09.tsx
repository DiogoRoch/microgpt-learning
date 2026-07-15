/**
 * Chapter 9 — Training. The real model trains live in a Web Worker (loss
 * curve drawing itself, samples evolving from babble to names), plus the
 * precomputed 1000-step history for instant time-travel into any snapshot.
 */
import { useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Aside } from '../components/Aside.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { LossCurve } from '../viz/LossCurve.tsx'
import { MatrixHeatmap } from '../viz/MatrixHeatmap.tsx'
import { useTrainer } from '../app/useTrainer.ts'
import { useRun, useModelAt, tokenizer } from '../app/useModel.ts'
import { snapshotForStep } from '../data/loadRun.ts'
import facts from '../data/facts.json'
import docsGolden from '../../golden/docs.json'

function LiveTraining() {
  const trainer = useTrainer()
  const { setHighlight } = useCodeSync()
  const [pace, setPace] = useState<'watch' | 'flat-out'>('watch')

  const busy = trainer.training
  const done = trainer.step >= 1000

  return (
    <div className="not-prose my-4 space-y-3 rounded-lg border border-ink/15 bg-white/60 p-4" onMouseEnter={() => setHighlight([151, 152, 153, 184])}>
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          disabled={!trainer.ready || busy || done}
          onClick={() => trainer.train({ untilStep: 1000, reportEvery: 5, samplesPerReport: 6, paceMs: pace === 'watch' ? 14 : 0 })}
          className="rounded bg-ink px-4 py-2 font-mono text-sm text-paper hover:opacity-90 focus-visible:outline-2 focus-visible:outline-[var(--hot)] disabled:opacity-30"
        >
          {done ? 'trained ✓' : busy ? 'training…' : trainer.step > 0 ? `resume from step ${trainer.step}` : '▶ train the real model'}
        </button>
        {busy && (
          <button type="button" onClick={trainer.stop} className="rounded border border-ink/25 px-3 py-2 font-mono text-sm hover:bg-ink/5">
            pause
          </button>
        )}
        <button
          type="button"
          disabled={busy}
          onClick={() => trainer.init('golden')}
          className="rounded border border-ink/25 px-3 py-2 font-mono text-sm hover:bg-ink/5 disabled:opacity-30"
        >
          reset to step 0
        </button>
        <label className="ml-auto flex items-center gap-2 font-mono text-xs text-muted">
          pace
          <select
            value={pace}
            onChange={(e) => setPace(e.target.value as 'watch' | 'flat-out')}
            className="rounded border border-ink/20 bg-white px-2 py-1"
            aria-label="training pace"
          >
            <option value="watch">watchable (~20 s)</option>
            <option value="flat-out">flat out (~1 s)</option>
          </select>
        </label>
      </div>
      <LossCurve losses={trainer.losses} totalSteps={1000} />
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <div className="mb-1 font-mono text-xs text-muted">
            samples @ step {trainer.step} (temperature 0.5)
          </div>
          <div className="min-h-16 rounded bg-ink/5 p-2 font-mono text-sm leading-relaxed">
            {trainer.samples.length ? trainer.samples.join(' · ') : '— press train —'}
          </div>
        </div>
        <div>
          <div className="mb-1 font-mono text-xs text-muted">how they evolved</div>
          <div className="max-h-32 overflow-y-auto rounded bg-ink/5 p-2 font-mono text-[11px] leading-relaxed">
            {trainer.sampleHistory.length === 0 && '—'}
            {trainer.sampleHistory
              .filter((_, i) => i % 8 === 0 || i === trainer.sampleHistory.length - 1)
              .map((h) => (
                <div key={h.step}>
                  <span className="text-muted">step {String(h.step).padStart(4, ' ')}:</span> {h.names.slice(0, 4).join(', ')}
                </div>
              ))}
          </div>
        </div>
      </div>
      {trainer.elapsedMs != null && (
        <p className="font-mono text-xs" style={{ color: 'var(--pos)' }}>
          1000 steps of the complete algorithm ran in your browser tab in{' '}
          {(trainer.elapsedMs / 1000).toFixed(2)} s{pace === 'watch' ? ' (including watchable pacing)' : ''}. The
          pure-Python reference takes ~{Math.round(facts.pythonTrainSec)} s for the same math.
        </p>
      )}
    </div>
  )
}

function TimeTravel() {
  const run = useRun()
  const [step, setStep] = useState(1000)
  const model = useModelAt(step)
  const { setHighlight } = useCodeSync()
  const snap = run ? snapshotForStep(run, step) : null
  const wte = useMemo(() => (model ? [...model.sd['wte']!.data] : null), [model])

  if (!run || !snap) return <p className="font-mono text-sm text-muted">loading the precomputed run…</p>
  return (
    <div className="not-prose my-4 space-y-3" onMouseEnter={() => setHighlight([156, 157, 158])}>
      <LossCurve losses={run.losses} totalSteps={1000} overlay={run.pythonLosses} step={Math.min(step, 999)} onScrub={setStep} />
      <div className="flex flex-wrap items-start gap-6">
        <div>
          <div className="mb-1 font-mono text-xs text-muted">
            snapshot @ step {snap.step} — its 8 stored samples:
          </div>
          <div className="max-w-64 rounded bg-ink/5 p-2 font-mono text-sm leading-relaxed">
            {snap.samples.join(' · ')}
          </div>
          <div className="mt-2 font-mono text-xs text-muted">
            doc at step {Math.min(step, 999)}: &quot;{docsGolden.head[Math.min(step, 999)]}&quot;
          </div>
        </div>
        {wte && (
          <div>
            <div className="mb-1 font-mono text-xs text-muted">wte at this snapshot</div>
            <MatrixHeatmap
              data={wte}
              rows={27}
              cols={16}
              label="wte"
              cellSize={8}
              rowLabel={(r) => (r === 26 ? "'·' BOS" : `'${tokenizer.uchars[r]}'`)}
            />
          </div>
        )}
      </div>
      <p className="max-w-xl text-sm text-muted">
        The gray curve underneath is the Python file&apos;s own run — this app&apos;s
        engine (blue) tracks it to within {run.maxAbsLossDiff.toExponential(1)} at every
        one of the 1000 steps. Drag anywhere: the weights, samples and the heatmap jump
        to the nearest stored snapshot (every {run.snapshotEvery} steps).
      </p>
    </div>
  )
}

const chapter = CHAPTERS[9]!

export default function Ch09() {
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Everything is assembled: tokenizer, forward pass, loss, gradients, optimizer.
        Line 153 runs the loop — <code>for step in range(1000)</code>: take{' '}
        <em>one name</em> (line 156, <code>docs[step % len(docs)]</code>), build its loss,
        backward, Adam, repeat. No batches, no epochs: a thousand names, seen once each.
      </p>

      <h2>Train it. Right here.</h2>
      <p>
        This button runs the complete algorithm — the same initial weights and the same
        document order as the Python file, in a Web Worker:
      </p>
      <LiveTraining />
      <p>
        Watch the first seconds: the curve starts at the ln 27 line (chapter 7&apos;s
        prediction, now measured) and the early samples are alphabet soup. Structure
        arrives fast — vowels alternate, lengths become name-like — then progress slows
        into a noisy grind. The wiggle never goes away: every step grades a{' '}
        <em>different single name</em>, and some names are simply harder.
      </p>

      <h2>Time travel</h2>
      <p>
        The shipped history of the exact reference run, one snapshot every 25 steps —
        scrub it:
      </p>
      <TimeTravel />

      <Aside kind="wild" title="What's different at real scale">
        Batches (thousands of documents per step, gradients averaged), epochs (multiple
        passes over data), warmup before decay, gradient clipping, distributed training
        across thousands of GPUs — all engineering around the identical five-line loop:
        forward, loss, backward, optimizer step, zero grads. GPT-2 was this file with
        bigger matrices and better plumbing.
      </Aside>

      <PredictReveal
        qid="ch9-wiggle"
        question={<>Why does the loss curve wiggle instead of descending smoothly?</>}
        options={['floating-point noise', 'each step grades one different document', 'Adam adds randomness']}
        answerIndex={1}
        explanation={
          <>
            There is no randomness after init — the wiggle is the <em>data</em>. Step
            grades &quot;sophia&quot; (easy, common patterns), next step grades
            &quot;xzavier&quot; (hard). One-doc losses jump around a slowly falling
            average. Batched training smooths this by averaging many documents per step.
          </>
        }
      />
      <PredictReveal
        qid="ch9-repeats"
        question={<>Does the model ever see the same name twice during its 1000 steps?</>}
        options={['yes — it cycles every 100 names', 'no — 1000 steps < 32,033 names', 'yes — shuffling repeats names']}
        answerIndex={1}
        explanation={
          <>
            <code>docs[step % len(docs)]</code> with len(docs) = 32,033 and only 1000
            steps: the modulo never wraps. The model learns names in general from 1000
            examples seen <em>once each</em> — it literally cannot have memorized the
            other 31,033.
          </>
        }
      />
      <PredictReveal
        qid="ch9-more-steps"
        question={<>Train for 100× longer (100,000 steps). Does the loss reach 0?</>}
        options={['yes, eventually', 'no — it flattens well above 0', 'only with a bigger vocab']}
        answerIndex={1}
        explanation={
          <>
            Names are partly irreducible coin flips: after &quot;ma&quot;, both
            &quot;ria&quot; and &quot;ya&quot; are genuinely possible, and no model can
            know which. That residual entropy keeps per-token loss bounded away from
            zero (around 2.0 for this dataset and tokenizer). Loss 0 would mean names are
            fully deterministic — they aren&apos;t.
          </>
        }
      />

      <Recap
        chapterId={9}
        points={[
          <>Training is a thousand turns of: one name → loss → backward → Adam. Nothing else.</>,
          <>The curve starts at ln 27 exactly as derived, falls fast, then grinds — the wiggle is per-document difficulty.</>,
          <>Your browser ran the same run as the Python file (same init, same doc order) and matched its curve to ~1e-7.</>,
        ]}
      />
    </ChapterFrame>
  )
}
