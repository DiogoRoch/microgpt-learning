/**
 * Chapter 6 — MLP & the residual stream. 16 → 64 → ReLU → 16 on real
 * activations (watch neurons die live), and the residual stream as the spine
 * everything reads from and writes to.
 */
import { useEffect, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { CompareToggle } from '../components/Compare.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { StepPlayer, useStepPlayer } from '../components/StepPlayer.tsx'
import { TokenTape } from '../components/TokenTape.tsx'
import { VectorChips } from '../viz/VectorChips.tsx'
import { fmt, signedColor } from '../viz/color.ts'
import { useAppStore } from '../app/store.ts'
import { labelOf, useTrace } from '../app/useModel.ts'

/** 64 neurons as a compact strip; dead ones (ReLU output 0) crossed out. */
function NeuronStrip({ pre, post }: { pre: number[]; post: number[] }) {
  const vmax = Math.max(...pre.map(Math.abs), 0.001)
  const dead = post.filter((v) => v === 0).length
  return (
    <div>
      <div className="flex flex-wrap gap-[2px]">
        {pre.map((v, i) => (
          <div
            key={i}
            title={`neuron ${i}: fc1 = ${fmt(v)} → relu = ${fmt(post[i]!)}`}
            className="flex h-6 w-4 items-center justify-center rounded-[2px] font-mono text-[9px]"
            style={{
              background: signedColor(v, vmax),
              opacity: post[i] === 0 ? 0.9 : 1,
            }}
          >
            {post[i] === 0 ? <span style={{ color: 'var(--paper)' }}>×</span> : ''}
          </div>
        ))}
      </div>
      <p className="mt-1 font-mono text-xs text-muted">
        {dead} of 64 neurons dead at this position (fc1 output ≤ 0 → ReLU emits 0 → their
        gradient is 0 too)
      </p>
    </div>
  )
}

function MlpStepper() {
  const example = useAppStore((s) => s.example)
  const step = useAppStore((s) => s.checkpointStep)
  const trace = useTrace(example, step)
  const player = useStepPlayer(trace?.n ?? 1, 1)
  const { setHighlight } = useCodeSync()
  useEffect(() => setHighlight([136, 137, 138, 139, 140, 141]), [setHighlight, player.index])

  if (!trace) return <p className="font-mono text-sm text-muted">loading…</p>
  const pos = Math.min(player.index, trace.n - 1)
  const layer = trace.calls[pos]!.layers[0]!
  return (
    <div className="not-prose my-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TokenTape tokens={trace.tokens.slice(0, trace.n)} labelOf={labelOf} showIds={false} activeIndex={pos} />
        <CompareToggle />
      </div>
      <StepPlayer player={player} length={trace.n} label="MLP position stepper" format={(i) => `pos ${i} '${labelOf(trace.tokens[i]!)}'`} />
      <div className="space-y-2 overflow-x-auto">
        <div onMouseEnter={() => setHighlight([137])}>
          <VectorChips values={layer.x_ln_mlp} label="rmsnorm(x)" cellSize={20} />
        </div>
        <div className="font-mono text-[11px] text-muted">↓ mlp_fc1 (64×16) — expand to 64 neurons, then ReLU cuts the negatives</div>
        <div onMouseEnter={() => setHighlight([138, 139])}>
          <NeuronStrip pre={layer.fc1} post={layer.relu} />
        </div>
        <div className="font-mono text-[11px] text-muted">↓ mlp_fc2 (16×64) — contract back to 16 dims</div>
        <div onMouseEnter={() => setHighlight([140])}>
          <VectorChips values={layer.fc2} label="fc2" cellSize={20} />
        </div>
        <div className="font-mono text-[11px] text-muted">↓ + x_residual (line 141)</div>
        <div onMouseEnter={() => setHighlight([141])}>
          <VectorChips values={layer.x_after_mlp} label="x (final)" cellSize={20} />
        </div>
      </div>
    </div>
  )
}

/** The residual stream as a spine diagram. */
function ResidualSpine() {
  const { setHighlight } = useCodeSync()
  const [hover, setHover] = useState<string | null>(null)
  const stops: Array<{ id: string; label: string; lines: number[]; kind: 'norm' | 'add' | 'start' | 'end' }> = [
    { id: 'emb', label: 'x = wte[t] + wpe[p], rmsnorm', lines: [111, 112], kind: 'start' },
    { id: 'attn', label: 'x += attention(rmsnorm(x))', lines: [116, 117, 133, 134], kind: 'add' },
    { id: 'mlp', label: 'x += mlp(rmsnorm(x))', lines: [136, 137, 140, 141], kind: 'add' },
    { id: 'head', label: 'logits = lm_head @ x', lines: [143], kind: 'end' },
  ]
  return (
    <div className="not-prose my-4">
      <svg width="100%" viewBox="0 0 560 260" role="img" aria-label="the residual stream: a straight line with two read-transform-add branches">
        <line x1={60} y1={20} x2={60} y2={240} stroke="var(--ink)" strokeWidth={3} />
        <text x={60} y={12} textAnchor="middle" fontSize={11} className="font-mono" fill="var(--muted)">
          the residual stream
        </text>
        {stops.map((s, i) => {
          const y = 40 + i * 60
          return (
            <g
              key={s.id}
              onMouseEnter={() => {
                setHover(s.id)
                setHighlight(s.lines)
              }}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: 'default' }}
            >
              {s.kind === 'add' && (
                <>
                  <path
                    d={`M 60 ${y - 18} C 150 ${y - 18}, 150 ${y - 18}, 200 ${y - 18} L 320 ${y - 18} C 380 ${y - 18}, 380 ${y}, 60 ${y}`}
                    fill="none"
                    stroke={hover === s.id ? 'var(--hot)' : 'rgba(22,24,29,0.35)'}
                    strokeWidth={2}
                  />
                  <rect x={200} y={y - 34} width={120} height={26} rx={6} fill="white" stroke={hover === s.id ? 'var(--hot)' : 'rgba(22,24,29,0.3)'} />
                  <text x={260} y={y - 17} textAnchor="middle" fontSize={10} className="font-mono" fill="var(--ink)">
                    {s.id === 'attn' ? 'attention' : 'MLP'}
                  </text>
                  <circle cx={60} cy={y} r={9} fill="var(--paper)" stroke="var(--ink)" strokeWidth={2} />
                  <text x={60} y={y + 4} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--ink)">
                    +
                  </text>
                </>
              )}
              {s.kind !== 'add' && <circle cx={60} cy={y} r={5} fill="var(--ink)" />}
              <text x={90} y={y + 4} fontSize={11.5} className="font-mono" fill={hover === s.id ? 'var(--ink)' : 'var(--muted)'}>
                {s.label}
              </text>
            </g>
          )
        })}
      </svg>
      <p className="font-mono text-xs text-muted">hover a stop to light up its lines in the file</p>
    </div>
  )
}

const chapter = CHAPTERS[6]!

export default function Ch06() {
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Attention let positions trade information; the MLP is where each position{' '}
        <em>thinks about what it collected</em> — alone, no cross-talk. Lines 136–141:
        expand the 16-dim state to 64 <Term t="head">neurons</Term>, cut everything
        negative (ReLU), project back to 16, and add the result into the stream.
      </p>

      <h2>64 detectors, live</h2>
      <p>
        Each of <code>mlp_fc1</code>&apos;s 64 rows is a pattern detector: its dot product
        with the (normalized) state is high when the state matches its pattern. ReLU turns
        &quot;how much does it match&quot; into &quot;fire this much, or stay silent.&quot;
        The ×-marked cells are silent neurons — try the untrained/trained toggle and watch
        the firing pattern reorganize:
      </p>
      <MlpStepper />

      <h2>The spine</h2>
      <p>
        Notice what the MLP <em>didn&apos;t</em> do: replace the state. Line 141 —{' '}
        <code>x = [a + b for a, b in zip(x, x_residual)]</code> — <em>adds</em> its output
        to what was already there. The transformer is best read as one straight line, the{' '}
        <Term t="residual stream">residual stream</Term>, with sub-networks that read from
        it, compute, and <strong>add back</strong>:
      </p>
      <ResidualSpine />
      <p>
        This is chapter 2&apos;s shared-node picture at full scale. x_residual feeds{' '}
        <em>two</em> consumers — the block and the addition — so in the backward pass its
        gradient <em>accumulates</em> from both paths (<code>+=</code>). The straight-line
        path from the loss back to the embeddings is what keeps early layers trainable;
        in a 96-layer GPT it&apos;s the reason gradients survive the trip at all.
      </p>

      <Aside kind="wild" title="GeLU, gating, and 4×">
        GPT-2 uses GeLU (a smooth ReLU) — microgpt&apos;s ReLU is the honest simplification,
        stated on line 93. Modern models often use gated variants (SwiGLU). The 4×
        expansion ratio (16 → 64 here, 768 → 3072 in GPT-2) is a convention that has
        survived essentially unchanged since the original transformer — and in both cases
        the MLP holds about two-thirds of the non-embedding parameters.
      </Aside>

      <PredictReveal
        qid="ch6-fc1-params"
        question={<>How many parameters does mlp_fc1 hold?</>}
        options={['256', '1,024', '4,096']}
        answerIndex={1}
        explanation={
          <>
            matrix(4·n_embd, n_embd) = 64×16 = <strong>1,024</strong> — and fc2 mirrors it
            with 16×64. Together they&apos;re 2,048 of the 4,192 total: about half the
            model is MLP (you saw this in chapter 3&apos;s treemap).
          </>
        }
      />
      <PredictReveal
        qid="ch6-all-dead"
        question={<>Suppose at some position all 64 ReLU outputs are 0. What does the MLP block contribute there?</>}
        options={['a zero vector — x passes through unchanged', 'x gets zeroed', 'an error: softmax of nothing']}
        answerIndex={0}
        explanation={
          <>
            fc2 of a zero vector is zero, and line 141 adds zero to x_residual:{' '}
            <strong>x continues exactly as it was</strong>. The residual design makes
            &quot;do nothing&quot; the easiest thing a block can learn — a safe default
            that&apos;s also why deep stacks of these blocks train stably.
          </>
        }
      />
      <PredictReveal
        qid="ch6-prenorm"
        question={<>rmsnorm is applied BEFORE each block (lines 117, 137), not after. What stays clean because of that?</>}
        options={['the block inputs and the residual path', 'the loss value', 'the parameter count']}
        answerIndex={0}
        explanation={
          <>
            Pre-norm gives every block a unit-scale input (whatever the stream has
            accumulated), while the stream itself is never squashed between additions —
            the straight line from loss to embeddings stays untouched. Post-norm (the
            original 2017 design) normalizes <em>the stream</em> after each add, which
            makes deep stacks much touchier to train.
          </>
        }
      />

      <Recap
        chapterId={6}
        points={[
          <>MLP = per-position pattern detectors: expand ×4, ReLU silences non-matches, contract, add back.</>,
          <>The residual stream is the spine: blocks read (via rmsnorm), compute, and += into it — never replace it.</>,
          <>Residuals are chapter 2&apos;s gradient accumulation at scale; &quot;contribute nothing&quot; is always available and cheap.</>,
        ]}
      />
    </ChapterFrame>
  )
}
