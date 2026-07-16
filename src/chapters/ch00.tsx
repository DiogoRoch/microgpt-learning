/**
 * Chapter 0 — The Big Picture. The whole file as a living data-flow map:
 * every stage clickable, every number real (facts.json ← golden run).
 */
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { useCodeSync, lineRange } from '../components/CodeSync.tsx'
import facts from '../data/facts.json'

interface Stage {
  id: string
  title: string
  value: string
  detail: string
  chapter: number
  lines: [number, number]
}

const STAGES: Stage[] = [
  { id: 'data', title: 'dataset', value: `${facts.numDocs.toLocaleString()} names`, detail: `"${facts.docHead[0]}", "${facts.docHead[1]}", …`, chapter: 1, lines: [14, 21] },
  { id: 'tok', title: 'tokenizer', value: `vocab_size = ${facts.vocabSize}`, detail: 'a–z → 0–25, BOS → 26', chapter: 1, lines: [23, 27] },
  { id: 'auto', title: 'autograd', value: 'class Value', detail: 'every number remembers its history', chapter: 2, lines: [29, 72] },
  { id: 'params', title: 'parameters', value: `${facts.numParams.toLocaleString()} numbers`, detail: 'gaussian, σ = 0.08', chapter: 3, lines: [74, 90] },
  { id: 'model', title: 'gpt()', value: 'tokens → logits', detail: 'embeddings · attention · MLP', chapter: 4, lines: [92, 144] },
  { id: 'loss', title: 'loss', value: `starts at ${facts.loss0.toFixed(2)}`, detail: '≈ ln 27: uniform guessing', chapter: 7, lines: [160, 169] },
  { id: 'bwd', title: 'backward', value: `${facts.numParams.toLocaleString()} gradients`, detail: 'chain rule over the graph', chapter: 8, lines: [171, 172] },
  { id: 'adam', title: 'Adam', value: 'lr 0.01 → 0', detail: 'm, v, bias correction', chapter: 8, lines: [174, 182] },
  { id: 'loop', title: '×1000 steps', value: `loss → ${facts.finalLossPython.toFixed(2)}`, detail: 'one name per step', chapter: 9, lines: [151, 158] },
  { id: 'sample', title: 'inference', value: `"${facts.samplesPython[0]}", "${facts.samplesPython[1]}", …`, detail: 'temperature 0.5, stop on BOS', chapter: 10, lines: [186, 200] },
]

function FlowMap() {
  const navigate = useNavigate()
  const { setHighlight } = useCodeSync()
  // The last stage pointed at: keeps its card lit and its lines highlighted on
  // both the left minimap and the right code panel, so the mapping stays legible.
  const [activeId, setActiveId] = useState<string | null>(null)

  const point = (s: Stage) => {
    setActiveId(s.id)
    setHighlight(lineRange(s.lines[0], s.lines[1]))
  }

  return (
    <div className="not-prose">
      <ol className="grid grid-cols-1 gap-2 sm:grid-cols-2" aria-label="the algorithm, stage by stage">
        {STAGES.map((s, i) => {
          const active = activeId === s.id
          return (
            <li key={s.id} className="flex items-stretch gap-2">
              <span className="flex w-6 shrink-0 items-center justify-center font-mono text-[10px] text-muted" aria-hidden>
                {i + 1}
              </span>
              <button
                type="button"
                onClick={() => navigate(`/ch/${CHAPTERS[s.chapter]!.slug}`)}
                onMouseEnter={() => point(s)}
                onFocus={() => point(s)}
                className="group flex-1 rounded-lg border px-4 py-3 text-left transition-colors focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
                style={
                  active
                    ? { borderColor: 'var(--hot)', background: 'rgba(255,201,77,0.12)' }
                    : { borderColor: 'rgba(19,19,22,0.15)', background: 'rgba(255,255,255,0.6)' }
                }
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-mono text-sm font-semibold">{s.title}</span>
                  <span className="font-mono text-[10px] text-muted group-hover:text-ink">
                    L{s.lines[0]}–{s.lines[1]} · ch {s.chapter} →
                  </span>
                </div>
                <div className="mt-1 font-mono text-[13px]" style={{ color: 'var(--neg)' }}>
                  {s.value}
                </div>
                <div className="mt-0.5 text-xs text-muted">{s.detail}</div>
              </button>
            </li>
          )
        })}
      </ol>
      <p className="mt-3 font-mono text-xs text-muted">
        hover a stage to trace it through the file — its lines light up on the map at left and
        in the panel at right · click to open its chapter
      </p>
    </div>
  )
}

const chapter = CHAPTERS[0]!

export default function Ch00() {
  return (
    <ChapterFrame chapter={chapter} fullFileCode>
      <p>
        The file on the right is a <em>complete</em> GPT — the same architecture family as the
        chatbots, shrunk to its irreducible core. It downloads a list of names, learns
        how names are spelled, and invents new ones. No frameworks, no libraries, no
        hidden machinery: <strong>200 lines of Python, and every one of them is on this
        page&apos;s edge.</strong>
      </p>
      <p>
        By the end you will understand each line — not &quot;roughly what attention
        does,&quot; but why <code>keys[li].append(k)</code> makes a causal mask
        unnecessary, why the <Term t="loss">loss</Term> starts at exactly ln 27, and which of
        the {facts.numParams.toLocaleString()} <Term t="parameters">parameters</Term> store what the model
        learned about the letter <code>a</code>.
      </p>

      <h2>The whole algorithm, one screen</h2>
      <p>
        Ten stages. The numbers below aren&apos;t illustrations — they come from actually
        running the file (its exact random seed and all), checked against the Python
        original to nine decimal places.
      </p>
      <FlowMap />

      <h2>One name, all the way through</h2>
      <p>
        A single running example threads every chapter: <strong>emma</strong>, the first
        name in the dataset. You&apos;ll watch her become <Term t="token">tokens</Term>{' '}
        <code>[26, 4, 12, 12, 0, 26]</code>, then a 16-number vector, then queries and
        keys, then 27 <Term t="logits">logits</Term>, then a loss — and at the end, the
        trained model will invent fresh names in your browser. The names the Python file
        itself prints with seed 42 — <code>{facts.samplesPython.slice(0, 3).join(', ')}</code>,
        … — are stored here too, as ground truth this app is tested against.
      </p>

      <PredictReveal
        qid="ch0-python-time"
        question={<>The pure-Python file trains in about 4½ minutes. This app re-implements it with fast arrays. How long does the same 1000-step training take in your browser?</>}
        options={['~2 minutes', '~20 seconds', '~0.3 seconds']}
        answerIndex={2}
        explanation={
          <>
            <strong>~0.3 seconds</strong> — measured at {(facts.tsTrainMs / 1000).toFixed(2)}s for
            this build. Karpathy&apos;s point exactly: <em>&quot;This file is the complete
            algorithm. Everything else is just efficiency.&quot;</em> The Python file spends its
            time building ~100,000 little Value objects per step; the math itself is tiny.
            Chapter 9 lets you run it yourself.
          </>
        }
      />
      <PredictReveal
        qid="ch0-vocab"
        question={<>The dataset is 32,033 lowercase names. Why is vocab_size 27 and not 26?</>}
        options={['padding to a power-friendly size', 'one extra token marks where names begin and end', 'uppercase letters share ids']}
        answerIndex={1}
        explanation={
          <>
            The 26 letters get ids 0–25, and one special <Term t="BOS">BOS</Term> token gets id 26.
            It brackets every name on both sides — the left one is the prompt that means
            &quot;a name starts here&quot;, the right one is the model&apos;s way of saying
            &quot;I&apos;m done.&quot; Chapter 1 makes this concrete.
          </>
        }
      />
      <PredictReveal
        qid="ch0-first-loss"
        question={<>Before any training, the model&apos;s first measured loss is 3.37. What sets that number?</>}
        options={['the random seed', 'uniform guessing over 27 tokens: −ln(1/27) ≈ 3.30', 'the length of the first name']}
        answerIndex={1}
        explanation={
          <>
            An untrained model spreads probability almost evenly, so each prediction is worth
            about −log(1/27) = ln 27 ≈ 3.30. The measured 3.37 is that, plus a little noise
            from the random init on one particular document (&quot;yuheng&quot;). Chapter 7
            derives it; chapter 9 shows the very first step of the real loss curve landing there.
          </>
        }
      />

      <Recap
        chapterId={0}
        points={[
          <>The file is the entire algorithm: dataset → tokenizer → autograd → parameters → forward → loss → backward → Adam → loop → sampling.</>,
          <>Every number in this app is computed by a re-implementation that is parity-tested against the file itself.</>,
          <>The minimap on the left is microgpt.py. Chapters light it up; finishing means the whole file is yours.</>,
        ]}
      />
      <p className="mt-8">
        Ready? <Link to={`/ch/${CHAPTERS[1]!.slug}`}>Chapter 1: Data &amp; Tokenizer →</Link>
      </p>
    </ChapterFrame>
  )
}
