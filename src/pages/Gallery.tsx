/**
 * Dev gallery (/dev/gallery): every viz primitive rendering REAL golden data
 * — the Phase 2 proof that no number on screen is fake. Also exercises the
 * code-sync mechanism end to end via the StepPlayer.
 */
import { CodePanel } from '../components/CodePanel.tsx'
import { CodeSyncProvider, useCodeSync } from '../components/CodeSync.tsx'
import { StepPlayer, useStepPlayer } from '../components/StepPlayer.tsx'
import { AttnMatrix } from '../viz/AttnMatrix.tsx'
import { BarDistribution } from '../viz/BarDistribution.tsx'
import { MatrixHeatmap } from '../viz/MatrixHeatmap.tsx'
import { VectorChips } from '../viz/VectorChips.tsx'
import step0Trace from '../../golden/step0_trace.json'
import initWeights from '../../golden/init_weights.json'
import tokenizerGolden from '../../golden/tokenizer.json'
import { useEffect } from 'react'

const uchars = tokenizerGolden.uchars
const tokenLabel = (id: number) => (id === tokenizerGolden.bos ? '·' : uchars[id]!)
const docTokens = step0Trace.tokens.map(tokenLabel)
const wte = (initWeights as Record<string, number[][]>)['wte']!

function PositionStepper() {
  const player = useStepPlayer(step0Trace.n, 1.5)
  const { setHighlight } = useCodeSync()
  const call = step0Trace.gpt_calls[player.index]!
  const pos = step0Trace.positions[player.index]!

  // Code-sync demo: each step of the forward loop IS lines 163–168.
  useEffect(() => {
    setHighlight([163, 164, 165, 166, 167, 168])
  }, [setHighlight])

  return (
    <section className="space-y-4">
      <h2 className="font-display text-2xl">
        Step through step 0, doc &quot;{step0Trace.doc}&quot; — real golden intermediates
      </h2>
      <StepPlayer
        player={player}
        length={step0Trace.n}
        label="position stepper"
        format={(i) => `pos ${i} '${docTokens[i]}'`}
      />
      <div className="space-y-1.5 overflow-x-auto py-2">
        <VectorChips values={call.tok_emb} label="tok_emb" />
        <VectorChips values={call.pos_emb} label="pos_emb" />
        <VectorChips values={call.x_emb_sum} label="x (sum)" />
        <VectorChips values={call.x_emb_norm} label="rmsnorm(x)" />
        <VectorChips values={call.logits} label="logits" />
      </div>
      <div>
        <h3 className="mb-2 font-mono text-sm text-muted">
          probs at pos {call.pos_id} — target &apos;{tokenLabel(pos.target_id)}&apos;, loss_t ={' '}
          {pos.loss_t.toFixed(4)}
        </h3>
        <BarDistribution
          probs={pos.probs}
          labels={[...uchars, '·']}
          marker={pos.target_id}
          fullScale={false}
          height={120}
        />
      </div>
    </section>
  )
}

export default function Gallery() {
  const attnHead0 = step0Trace.gpt_calls.map((c) => c.layers[0]!.heads[0]!.attn_weights)
  return (
    <CodeSyncProvider>
      <div className="mx-auto grid max-w-[1400px] gap-8 px-8 py-10 xl:grid-cols-[minmax(0,1fr)_minmax(360px,42%)]">
        <div className="space-y-12">
          <h1 className="font-display text-3xl">Primitives gallery (dev)</h1>
          <PositionStepper />
          <section>
            <h2 className="mb-3 font-display text-2xl">AttnMatrix — head 0, step 0 (untrained)</h2>
            <AttnMatrix weights={attnHead0} tokens={docTokens.slice(0, step0Trace.n)} showValues />
          </section>
          <section>
            <h2 className="mb-3 font-display text-2xl">MatrixHeatmap — wte at init (27 × 16)</h2>
            <MatrixHeatmap
              data={wte.flat()}
              rows={27}
              cols={16}
              label="wte"
              cellSize={13}
              rowLabel={(r) => `'${r === 26 ? '· (BOS)' : uchars[r]}' (token ${r})`}
              colLabel={(c) => `dim ${c}`}
            />
          </section>
        </div>
        <aside className="sticky top-6 hidden xl:block">
          <CodePanel ranges={[[153, 172]]} />
        </aside>
      </div>
    </CodeSyncProvider>
  )
}
