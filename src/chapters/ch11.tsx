/**
 * Chapter 11 — Playground. Everything unlocked: any prefix, any checkpoint,
 * any intermediate anywhere in the network, batch export — and the closing
 * screen: the full source, every line now conquered.
 */
import { useEffect, useMemo, useState } from 'react'
import { CHAPTERS, TOTAL_LINES, lineOwners } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { NumericGuess, PickLine, PredictReveal, Recap, TryIt } from '../components/Quiz.tsx'
import facts from '../data/facts.json'
import { CodePanel } from '../components/CodePanel.tsx'
import { StepPlayer, useStepPlayer } from '../components/StepPlayer.tsx'
import { TokenTape } from '../components/TokenTape.tsx'
import { VectorChips } from '../viz/VectorChips.tsx'
import { AttnMatrix } from '../viz/AttnMatrix.tsx'
import { BarDistribution } from '../viz/BarDistribution.tsx'
import { LossCurve } from '../viz/LossCurve.tsx'
import { fmt } from '../viz/color.ts'
import { useAppStore } from '../app/store.ts'
import { labelOf, tokenizer, traceWord, useModelAt, useRun, VOCAB_LABELS } from '../app/useModel.ts'
import { RNG } from '../engine/rng.ts'

function Inspector({ onAtZero }: { onAtZero: (loss: number) => void }) {
  const [word, setWord] = useState('emma')
  const [checkpoint, setCheckpoint] = useState(1000)
  const run = useRun()
  const model = useModelAt(checkpoint)
  const cleaned = useMemo(() => [...word.toLowerCase()].filter((c) => tokenizer.isInVocab(c)).join('').slice(0, 14), [word])
  const trace = useMemo(() => (model && cleaned ? traceWord(model, cleaned) : null), [model, cleaned])
  const player = useStepPlayer(trace?.n ?? 1, 1)
  useEffect(() => {
    if (checkpoint === 0 && trace) onAtZero(trace.loss)
  }, [checkpoint, trace, onAtZero])

  const STAGES = [
    'tok_emb', 'pos_emb', 'x_emb_sum', 'x_emb_norm', 'x_ln_attn', 'q', 'k', 'v',
    'x_attn', 'x_wo', 'x_after_attn', 'x_ln_mlp', 'fc1', 'relu', 'fc2', 'x_after_mlp', 'logits',
  ] as const
  const [stages, setStages] = useState<Set<string>>(new Set(['x_emb_norm', 'q', 'k', 'x_after_attn', 'x_after_mlp', 'logits']))

  if (!run || !trace) return <p className="font-mono text-sm text-muted">loading…</p>
  const pos = Math.min(player.index, trace.n - 1)
  const call = trace.calls[pos]!
  const layer = call.layers[0]!
  const vecOf = (s: string): number[] =>
    s === 'tok_emb' || s === 'pos_emb' || s === 'x_emb_sum' || s === 'x_emb_norm' || s === 'logits'
      ? (call[s as 'tok_emb'] as number[])
      : (layer[s as 'q'] as number[])

  return (
    <div className="not-prose my-4 space-y-4 rounded-lg border border-ink/15 bg-white/60 p-4">
      <div className="flex flex-wrap items-end gap-4">
        <label className="font-mono text-xs text-muted">
          prefix / word
          <input
            value={word}
            onChange={(e) => setWord(e.target.value)}
            spellCheck={false}
            className="mt-1 block w-44 rounded border border-ink/20 bg-white px-3 py-2 font-mono focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
          />
        </label>
        <label className="grow font-mono text-xs text-muted">
          checkpoint: step {checkpoint}
          <input
            type="range"
            min={0}
            max={1000}
            step={run.snapshotEvery}
            value={checkpoint}
            onChange={(e) => setCheckpoint(Number(e.target.value))}
            className="mt-2 block w-full accent-[var(--neg)]"
            aria-label="training checkpoint"
          />
        </label>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TokenTape tokens={trace.tokens.slice(0, trace.n)} labelOf={labelOf} showIds={false} activeIndex={pos} />
        <span className="font-mono text-xs text-muted">doc loss {fmt(trace.loss)}</span>
      </div>
      <StepPlayer player={player} length={trace.n} label="position" format={(i) => `pos ${i}`} />
      <div className="flex flex-wrap gap-1">
        {STAGES.map((s) => (
          <button
            key={s}
            type="button"
            aria-pressed={stages.has(s)}
            onClick={() =>
              setStages((prev) => {
                const next = new Set(prev)
                if (next.has(s)) next.delete(s)
                else next.add(s)
                return next
              })
            }
            className="rounded border px-2 py-0.5 font-mono text-[11px]"
            style={{
              borderColor: stages.has(s) ? 'var(--ink)' : 'rgba(22,24,29,0.2)',
              background: stages.has(s) ? 'var(--ink)' : 'transparent',
              color: stages.has(s) ? 'var(--paper)' : 'var(--muted)',
            }}
          >
            {s}
          </button>
        ))}
      </div>
      <div className="space-y-1.5 overflow-x-auto py-1">
        {STAGES.filter((s) => stages.has(s)).map((s) => (
          <VectorChips key={s} values={vecOf(s)} label={s} cellSize={s === 'fc1' || s === 'relu' ? 9 : 17} />
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <div>
          <div className="mb-1 font-mono text-xs text-muted">attention per head (rows ≤ pos {pos})</div>
          <div className="grid grid-cols-2 gap-2">
            {[0, 1, 2, 3].map((h) => (
              <div key={h}>
                <div className="font-mono text-[10px] text-muted">head {h}</div>
                <AttnMatrix
                  weights={trace.calls.map((c) => c.layers[0]!.heads[h]!.attn_weights)}
                  tokens={trace.tokens.slice(0, trace.n).map(labelOf)}
                  cellSize={Math.min(22, Math.floor(200 / trace.n))}
                  uptoRow={pos}
                  activeRow={pos}
                  caption={false}
                />
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="mb-1 font-mono text-xs text-muted">
            p(next) at pos {pos} — target &apos;{labelOf(trace.tokens[pos + 1]!)}&apos;, loss_t {fmt(trace.lossT[pos]!)}
          </div>
          <BarDistribution probs={trace.probs[pos]!} labels={VOCAB_LABELS} marker={trace.tokens[pos + 1]!} fullScale={false} height={130} />
          <div className="mt-3">
            <LossCurve losses={run.losses} totalSteps={1000} step={Math.min(checkpoint, 999)} onScrub={(s) => setCheckpoint(Math.round(s / run.snapshotEvery) * run.snapshotEvery)} height={120} />
          </div>
        </div>
      </div>
    </div>
  )
}

function Exporter() {
  const model = useModelAt(1000)
  const [count, setCount] = useState(50)
  const [names, setNames] = useState<string[] | null>(null)
  const generate = () => {
    if (!model) return
    const rng = new RNG(Date.now() & 0xffffff)
    setNames(Array.from({ length: count }, () => tokenizer.decode(model.sample(rng, 0.5))))
  }
  const download = () => {
    if (!names) return
    const blob = new Blob([names.join('\n') + '\n'], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'microgpt-names.txt'
    a.click()
    URL.revokeObjectURL(url)
  }
  return (
    <div className="not-prose my-4 space-y-3 rounded-lg border border-ink/15 bg-white/60 p-4">
      <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
        <label>
          count
          <input
            type="number"
            min={1}
            max={500}
            value={count}
            onChange={(e) => setCount(Math.max(1, Math.min(500, Number(e.target.value))))}
            className="ml-2 w-20 rounded border border-ink/20 bg-white px-2 py-1"
          />
        </label>
        <button type="button" onClick={generate} disabled={!model} className="rounded bg-ink px-3 py-1.5 text-paper hover:opacity-90 disabled:opacity-30">
          generate
        </button>
        {names && (
          <button type="button" onClick={download} className="rounded border border-ink/25 px-3 py-1.5 hover:bg-ink/5">
            ↓ download .txt
          </button>
        )}
      </div>
      {names && (
        <div className="max-h-40 overflow-y-auto rounded bg-ink/5 p-2 font-mono text-sm leading-relaxed">
          {names.join(' · ')}
        </div>
      )}
    </div>
  )
}

function ClosingFile() {
  const completed = useAppStore((s) => s.completed)
  const owners = useMemo(() => lineOwners(), [])
  let understood = 0
  for (let l = 1; l <= TOTAL_LINES; l++) {
    const o = owners[l]!
    if (o >= 0 && completed.includes(o)) understood++
  }
  const total = [...Array(TOTAL_LINES)].reduce<number>((acc, _, i) => (owners[i + 1]! >= 0 ? acc + 1 : acc), 0)
  return (
    <div className="not-prose my-4 space-y-3">
      <p className="font-mono text-sm">
        {understood}/{total} owned lines conquered
        {understood === total ? ' — the whole file is yours.' : ' — the minimap shows what remains.'}
      </p>
      <CodePanel maxHeight="80vh" />
    </div>
  )
}

const chapter = CHAPTERS[11]!

export default function Ch11() {
  const [zeroLoss, setZeroLoss] = useState<number | null>(null)
  const onAtZero = useMemo(() => (loss: number) => setZeroLoss((z) => z ?? loss), [])
  return (
    <ChapterFrame chapter={chapter} hideCodePanel>
      <p>
        No more guided tour. Any prefix in, any intermediate out, any checkpoint of the
        training run — the full instrument panel for a 4,192-parameter mind.
      </p>

      <h2>Inspect anything</h2>
      <Inspector onAtZero={onAtZero} />

      <TryIt
        qid="ch11-zero-loss"
        task={<>One last experiment: drag the checkpoint slider all the way back to step 0 and read the doc loss.</>}
        done={zeroLoss != null}
        payoff={
          <>
            {zeroLoss != null ? (
              <>
                Doc loss {fmt(zeroLoss)} — right at ln 27 ≈ 3.296, for <em>any</em> word
                you type.
              </>
            ) : (
              <>Right at ln 27 ≈ 3.296, for <em>any</em> word you type.</>
            )}{' '}
            You predicted this number in chapter 7, watched training escape it in chapter
            9, and here it is again from the untrained weights — the full circle, measured
            live.
          </>
        }
      />

      <h2>The gauntlet</h2>
      <p>
        Twelve chapters ago this file was a wall of code. Prove it isn&apos;t anymore —
        six questions, spanning the whole thing.
      </p>

      <PickLine
        qid="ch11-tokenizer-line"
        question={<>Click the line that is, by itself, the entire tokenizer.</>}
        lines={[20, 24, 25, 26]}
        answer={24}
        hint={<>Shuffling isn&apos;t tokenizing, and BOS bookkeeping comes after. Which line turns raw text into the vocabulary?</>}
        explanation={
          <>
            <code>uchars = sorted(set(&apos;&apos;.join(docs)))</code> — join everything,
            dedupe, sort. A character&apos;s id is its position in that list; lines 25–26
            just append BOS and count. Chapter 1, one line.
          </>
        }
      />
      <NumericGuess
        qid="ch11-params"
        question={<>How many learnable parameters does this whole model have?</>}
        answer={facts.numParams}
        placeholder="count"
        format={(v) => v.toLocaleString()}
        hint={<>Chapter 3&apos;s treemap: 432 + 256 + 432, four 256s, and two 1,024s.</>}
        explanation={
          <>
            <strong>4,192</strong> — wte 432 + wpe 256 + lm_head 432 + four attention
            matrices at 256 + two MLP matrices at 1,024. You have now personally poked a
            measurable fraction of them.
          </>
        }
      />
      <PredictReveal
        qid="ch11-after-backward"
        question={<>Inside one training step, loss.backward() has just filled all 4,192 grad slots. What runs next?</>}
        options={['the Adam update — which also zeroes each grad', 'sampling, to check progress', 'a second forward pass to verify the gradients']}
        answerIndex={0}
        hint={<>The loop body is only four moves: forward, backward, …, repeat. Nothing in it is optional.</>}
        explanation={
          <>
            Lines 174–182: for every parameter, update m and v, bias-correct, step by
            −lr_t·m̂/(√v̂+1e-8) — and <code>p.grad = 0</code> right there in the same loop,
            ready for the next document. Sampling only happens once, after step 1000.
          </>
        }
      />
      <PredictReveal
        qid="ch11-double-logits"
        question={<>A mischievous edit doubles every logit just before the sampling softmax. What did you actually change?</>}
        options={['nothing — softmax normalizes it away', 'it now samples exactly like temperature 0.5', 'names get twice as long']}
        answerIndex={1}
        hint={<>Line 195 divides logits by T before softmax. Doubling is dividing by…?</>}
        explanation={
          <>
            softmax(2z) = softmax(z / 0.5): doubling the logits <em>is</em> temperature
            0.5 — sharper, more conservative sampling. (Chapter 7&apos;s shift-invariance
            was about <em>adding</em> a constant; <em>multiplying</em> changes the gaps,
            and softmax is exponential in the gaps.)
          </>
        }
      />
      <PredictReveal
        qid="ch11-knowledge"
        question={<>The trained model &quot;knows&quot; names often end in &apos;a&apos;. Where, physically, is that knowledge?</>}
        options={['spread across the 4,192 weights', 'in the KV cache', 'in a stored list of training names']}
        answerIndex={0}
        hint={<>The KV cache is rebuilt from scratch for every sample, and no list of names survives past line 24. What is the only thing training ever changed?</>}
        explanation={
          <>
            Training changed nothing but the nine matrices — so everything the model knows
            is encoded there: lm_head&apos;s &apos;a&apos; row aligning with states that
            follow m-sounds, attention weights that look back at the right letters. The KV
            cache is scratch paper, discarded after every name.
          </>
        }
      />
      <NumericGuess
        qid="ch11-final-loss"
        question={<>Where does the reference run&apos;s training loss end up after step 1000? (Scrub the inspector&apos;s curve if you don&apos;t remember.)</>}
        answer={facts.finalLossPython}
        tolerance={0.05}
        placeholder="final loss"
        unit="≈"
        format={(v) => v.toFixed(2)}
        hint={<>Between the ln 27 cold start (3.30) and the irreducible-entropy floor (~2.0). The curve in the inspector ends there.</>}
        explanation={
          <>
            <strong>{facts.finalLossPython.toFixed(2)}</strong> — down from 3.30, still far
            above 0, because names are partly coin flips no model can call. This app&apos;s
            engine reproduces that number to ~1e-7; you watched the whole descent in
            chapter 9.
          </>
        }
      />

      <h2>Export a name list</h2>
      <Exporter />

      <h2>The file, conquered</h2>
      <ClosingFile />

      <Recap
        chapterId={11}
        points={[
          <>Tokens → embeddings → attention over a growing cache → MLP → logits → softmax → loss → gradients → Adam → loop → sampling. You can now point at the line for each arrow.</>,
          <>Every visualization you used was the real engine, parity-tested against the file to nine decimals.</>,
          <>200 lines. The complete algorithm. Everything else is just efficiency.</>,
        ]}
      />
    </ChapterFrame>
  )
}
