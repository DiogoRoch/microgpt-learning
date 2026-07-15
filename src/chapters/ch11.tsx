/**
 * Chapter 11 — Playground. Everything unlocked: any prefix, any checkpoint,
 * any intermediate anywhere in the network, batch export — and the closing
 * screen: the full source, every line now conquered.
 */
import { useMemo, useState } from 'react'
import { CHAPTERS, TOTAL_LINES, lineOwners } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { Recap } from '../components/Quiz.tsx'
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

function Inspector() {
  const [word, setWord] = useState('emma')
  const [checkpoint, setCheckpoint] = useState(1000)
  const run = useRun()
  const model = useModelAt(checkpoint)
  const cleaned = useMemo(() => [...word.toLowerCase()].filter((c) => tokenizer.isInVocab(c)).join('').slice(0, 14), [word])
  const trace = useMemo(() => (model && cleaned ? traceWord(model, cleaned) : null), [model, cleaned])
  const player = useStepPlayer(trace?.n ?? 1, 1)

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
  return (
    <ChapterFrame chapter={chapter} hideCodePanel>
      <p>
        No more guided tour. Any prefix in, any intermediate out, any checkpoint of the
        training run — the full instrument panel for a 4,192-parameter mind.
      </p>

      <h2>Inspect anything</h2>
      <Inspector />

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
