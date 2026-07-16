/**
 * Chapter 10 — Inference. Autoregressive sampling with the KV cache, one
 * decision at a time; the temperature slider from near-argmax to chaos; and
 * batch generation with a novelty check against all 32,033 training names.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PickLine, PredictReveal, Recap, TryIt } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { CompareToggle } from '../components/Compare.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { TokenTape } from '../components/TokenTape.tsx'
import { BarDistribution } from '../viz/BarDistribution.tsx'
import { fmt } from '../viz/color.ts'
import { useAppStore } from '../app/store.ts'
import { labelOf, tokenizer, useModelAt, VOCAB_LABELS } from '../app/useModel.ts'
import { softmaxProbs } from '../engine/tensor.ts'
import { RNG } from '../engine/rng.ts'
import facts from '../data/facts.json'
import namesJson from '../data/names.json'

const nameSet = new Set<string>(namesJson.names)
const BOS = tokenizer.bos

function SamplingLab({ onFinished, onTemp }: { onFinished: () => void; onTemp: (t: number) => void }) {
  const step = useAppStore((s) => s.checkpointStep)
  const model = useModelAt(step)
  const [temperature, setTemperature] = useState(0.5)
  const [generated, setGenerated] = useState<number[]>([BOS])
  const [finished, setFinished] = useState(false)
  const [lastPick, setLastPick] = useState<number | null>(null)
  const rngRef = useRef(new RNG(Date.now() & 0xffff))
  const { setHighlight } = useCodeSync()
  useEffect(() => {
    if (finished) onFinished()
  }, [finished, onFinished])
  useEffect(() => onTemp(temperature), [temperature, onTemp])

  // Fresh forward over the tokens so far → logits at the last position.
  const logits = useMemo(() => {
    if (!model) return null
    const out = model.forcedDecode(generated)
    return out[out.length - 1]!
  }, [model, generated])
  const probs = useMemo(() => (logits ? softmaxProbs(logits, temperature) : null), [logits, temperature])

  const sampleNext = useCallback(() => {
    if (!probs || finished || generated.length >= 16) return
    const tokenId = rngRef.current.choiceWeighted(probs)
    setLastPick(tokenId)
    setHighlight([196, 197, 198, 199])
    if (tokenId === BOS) setFinished(true)
    else setGenerated((g) => [...g, tokenId])
  }, [probs, finished, generated.length, setHighlight])

  const restart = useCallback(() => {
    setGenerated([BOS])
    setFinished(false)
    setLastPick(null)
  }, [])

  const word = generated.slice(1).map((t) => labelOf(t)).join('')
  const hitLimit = generated.length >= 16 && !finished

  if (!model || !probs) return <p className="font-mono text-sm text-muted">loading…</p>
  return (
    <div className="not-prose my-4 space-y-3 rounded-lg border border-ink/15 bg-white/60 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={sampleNext}
          disabled={finished || hitLimit}
          className="rounded bg-ink px-4 py-2 font-mono text-sm text-paper hover:opacity-90 focus-visible:outline-2 focus-visible:outline-[var(--hot)] disabled:opacity-30"
        >
          sample next token
        </button>
        <button type="button" onClick={restart} className="rounded border border-ink/25 px-3 py-2 font-mono text-sm hover:bg-ink/5">
          new name
        </button>
        <CompareToggle />
      </div>
      <div onMouseEnter={() => setHighlight([190, 191, 192, 193])}>
        <TokenTape tokens={generated} labelOf={labelOf} activeIndex={generated.length - 1} />
        <div className="mt-1 font-mono text-sm">
          {finished ? (
            <>
              finished: <strong>{word}</strong>{' '}
              <span className="text-muted">— the model emitted BOS: its learned &quot;I&apos;m done&quot;</span>
              {nameSet.has(word) && (
                <span style={{ color: 'var(--pos)' }}> · this exact name is in the training data</span>
              )}
            </>
          ) : hitLimit ? (
            <>
              stopped at block_size = 16 without emitting BOS: <strong>{word}</strong>
            </>
          ) : (
            <>
              so far: <strong>{word || '(empty — just BOS)'}</strong>
            </>
          )}
        </div>
      </div>
      <div onMouseEnter={() => setHighlight([195])}>
        <label htmlFor="temp" className="font-mono text-xs text-muted">
          temperature = {temperature.toFixed(2)} — dividing all logits before softmax
          {temperature === 0.5 ? ' (the file’s setting)' : ''}
        </label>
        <input
          id="temp"
          type="range"
          min={0.05}
          max={1.5}
          step={0.05}
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
          className="w-full accent-[var(--pos)]"
        />
        <div className="mb-1 font-mono text-xs text-muted">
          p(next) after &apos;{word || '·'}&apos; — watch mass {temperature < 0.5 ? 'pile onto the favorite' : 'spread out'} as you drag
        </div>
        <BarDistribution probs={probs} labels={VOCAB_LABELS} highlight={lastPick} fullScale={false} height={130} />
        <p className="mt-1 font-mono text-xs text-muted">
          entropy {(-probs.reduce((a, p) => a + (p > 0 ? p * Math.log2(p) : 0), 0)).toFixed(2)} bits ·
          top token p = {fmt(Math.max(...probs), 3)}
        </p>
      </div>
    </div>
  )
}

function BatchLab() {
  const step = useAppStore((s) => s.checkpointStep)
  const model = useModelAt(step)
  const [temperature, setTemperature] = useState(0.5)
  const [seed, setSeed] = useState(42)
  const { setHighlight } = useCodeSync()
  const names = useMemo(() => {
    if (!model) return null
    const rng = new RNG(seed)
    return Array.from({ length: 20 }, () => tokenizer.decode(model.sample(rng, temperature)))
  }, [model, temperature, seed])
  if (!names) return null
  const novel = names.filter((n) => !nameSet.has(n))
  return (
    <div className="not-prose my-4 space-y-3 rounded-lg border border-ink/15 bg-white/60 p-4" onMouseEnter={() => setHighlight([189, 196])}>
      <div className="flex flex-wrap items-center gap-4 font-mono text-xs">
        <label>
          temperature {temperature.toFixed(2)}
          <input type="range" min={0.05} max={1.5} step={0.05} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className="ml-2 w-32 accent-[var(--pos)]" />
        </label>
        <button type="button" onClick={() => setSeed((s) => s + 1)} className="rounded border border-ink/25 px-3 py-1.5 hover:bg-ink/5">
          ↻ fresh batch
        </button>
        <span className="text-muted">
          {novel.length}/20 are not in the training data
        </span>
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-sm sm:grid-cols-4">
        {names.map((n, i) => (
          <div key={i}>
            {n || <span className="text-muted">(empty)</span>}
            {n && !nameSet.has(n) && (
              <span title="not in the 32,033 training names" style={{ color: 'var(--pos)' }}>
                {' '}
                *
              </span>
            )}
          </div>
        ))}
      </div>
      <p className="text-xs text-muted">* = genuinely invented (checked against all 32,033 names)</p>
    </div>
  )
}

const chapter = CHAPTERS[10]!

export default function Ch10() {
  const [nameDone, setNameDone] = useState(false)
  const onFinished = useMemo(() => () => setNameDone(true), [])
  const [wentCold, setWentCold] = useState(false)
  const onTemp = useMemo(() => (t: number) => setWentCold((s) => s || t <= 0.1), [])
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Training taught the model p(next character | characters so far). Inference just{' '}
        <em>runs</em> that: start with BOS, ask for the distribution, roll a die weighted
        by it, feed the result back in. Lines 186–200 — and note what they reuse: the
        same <code>gpt()</code>, the same growing <Term t="KV cache">KV cache</Term> from
        chapter 5, one position at a time. Generation isn&apos;t a different mode; it&apos;s
        the forward pass with the input coming from the model&apos;s own choices.
      </p>

      <h2>One die roll at a time</h2>
      <SamplingLab onFinished={onFinished} onTemp={onTemp} />

      <TryIt
        qid="ch10-sample-name"
        task={<>Birth a name: press <strong>sample next token</strong> until the model decides it&apos;s finished.</>}
        done={nameDone}
        payoff={
          <>
            The ending wasn&apos;t a rule firing — the model <em>predicted</em> BOS, the
            same way it predicted every letter, because chapter 1&apos;s right-hand BOS
            taught it what &quot;done&quot; looks like. Every chatbot you&apos;ve used ends
            its turn with exactly this move.
          </>
        }
      />

      <PickLine
        qid="ch10-stop-line"
        question={<>Click the line that ends a generated name.</>}
        lines={[193, 195, 197, 199]}
        answer={197}
        hint={<>Not the loop bound — that&apos;s just a ceiling. What does the model have to <em>say</em> for the name to end?</>}
        explanation={
          <>
            <code>if token_id == BOS: break</code> — the stop is a <em>sampled token</em>,
            checked on line 197. Line 193&apos;s <code>range(block_size)</code> is only a
            hard ceiling (pos_id must stay within wpe&apos;s 16 rows); a well-trained model
            almost never hits it.
          </>
        }
      />

      <h2>What temperature actually does</h2>
      <p>
        Line 195 divides every logit by T before softmax. Because softmax is exponential
        in the logits, dividing by T &lt; 1 <em>multiplies the gaps</em> between them:
        favorites become dominant, longshots die. As T → 0 the distribution collapses
        onto the single best token (argmax — sampling becomes deterministic). At T = 1
        you sample the model&apos;s honest beliefs; above 1, flatter than it believes.
        Drag the slider above mid-name and watch the same 27 logits reshape.
      </p>

      <TryIt
        qid="ch10-cold"
        task={<>Freeze the dice: drop the temperature to 0.1 or below and watch the distribution above.</>}
        done={wentCold}
        payoff={
          <>
            Nearly all the probability mass piled onto one bar — sampling has become
            argmax, and generation is now <em>deterministic</em>: restart and sample again
            and you&apos;ll walk the same greedy path to the same name every time. That&apos;s
            the whole creativity dial: T reshapes this one distribution before each die roll.
          </>
        }
      />
      <PredictReveal
        qid="ch10-t0"
        question={<>At temperature ≈ 0, you generate 5 names in a row (same weights). What do you get?</>}
        options={['5 random names', 'the same name, 5 times', 'empty strings']}
        answerIndex={1}
        hint={<>You just made the distribution collapse onto its favorite. Is there anything left for the die roll to decide?</>}
        explanation={
          <>
            As T → 0 the softmax collapses onto the argmax: every die roll has one
            outcome, so generation is fully deterministic — same start (BOS), same
            greedy path, same name, every time. Try it in the batch lab below: slider to
            0.05, then ↻ — twenty copies of one name.
          </>
        }
      />

      <Aside kind="math" title="Temperature in one line">
        softmax(z/T)ᵢ ∝ e^(zᵢ/T) = (e^zᵢ)^(1/T) — raising probabilities to the power 1/T
        and renormalizing. T = 0.5 squares the ratios: a token 3× likelier becomes 9×
        likelier. The file&apos;s 0.5 trades diversity for name-likeness.
      </Aside>

      <h2>A batch of fresh humans</h2>
      <p>
        The file prints 20 samples; with seed 42 they are exactly{' '}
        <code>{facts.samplesPython.slice(0, 4).join(', ')}, …</code> (stored in this app
        as ground truth — the RNGs differ, so your batches below are your own):
      </p>
      <BatchLab />

      <Aside kind="wild" title="Sampling tricks in production">
        Real systems rarely sample raw. Top-k keeps only the k likeliest tokens; nucleus
        (top-p) keeps the smallest set covering p probability mass; repetition penalties
        discourage loops. All of them reshape this same 27-bar (or 50,257-bar)
        distribution before the die roll. And every chatbot you&apos;ve used ends its
        turn the way names end here: by emitting its version of BOS.
      </Aside>

      <PredictReveal
        qid="ch10-novel"
        question={<>Of 20 generated names, how many are typically copies of training names?</>}
        options={['none — it always invents', 'a handful — short common patterns collide with real names', 'all 20 — it memorized']}
        answerIndex={1}
        hint={<>Each training name was seen at most once — memorization is off the table. But what are the <em>most probable</em> letter patterns, by definition?</>}
        explanation={
          <>
            Check the * markers above: usually a few names (especially short ones like
            &quot;ann&quot;) exist in the data — not memorization (each training name was
            seen at most once) but convergence: real names are exactly the high-probability
            patterns. The longer inventions are usually genuinely new.
          </>
        }
      />

      <Recap
        chapterId={10}
        points={[
          <>Inference = the same gpt() + KV cache, fed its own samples: distribution → weighted die roll → repeat.</>,
          <>Temperature divides logits before softmax: T&lt;1 sharpens (T→0 = argmax), T=1 samples honest beliefs.</>,
          <>Names end when the model predicts BOS — a learned stop, not a rule.</>,
        ]}
      />
    </ChapterFrame>
  )
}
