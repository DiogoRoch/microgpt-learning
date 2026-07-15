/**
 * Chapter 1 — Data & Tokenizer. The real 32,033 names, the vocab built live
 * with the file's own recipe, a chars ↔ ids round-trip on user input, and
 * the two-sided BOS story.
 */
import { useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { TokenTape } from '../components/TokenTape.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { Tokenizer } from '../engine/tokenizer.ts'
import { useAppStore } from '../app/store.ts'
import namesJson from '../data/names.json'

const names: string[] = namesJson.names

// Build the vocabulary EXACTLY the way line 24 does — live, from the data.
const tok = Tokenizer.fromDocs(names)
const labelOf = (id: number) => (id === tok.bos ? '·' : (tok.uchars[id] ?? '?'))

// Character frequencies across the whole dataset (for the vocab table).
const charCounts = (() => {
  const counts = new Map<string, number>()
  for (const n of names) for (const ch of n) counts.set(ch, (counts.get(ch) ?? 0) + 1)
  return counts
})()
const maxCount = Math.max(...charCounts.values())

function VocabTable() {
  const { setHighlight } = useCodeSync()
  return (
    <div className="not-prose my-4">
      <div
        className="grid grid-cols-9 gap-1.5 md:grid-cols-14"
        style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(44px, 1fr))' }}
        onMouseEnter={() => setHighlight([24, 25, 26])}
      >
        {tok.uchars.map((ch, id) => (
          <div
            key={ch}
            className="rounded-md border border-ink/15 bg-white/60 p-1.5 text-center"
            title={`'${ch}' appears ${charCounts.get(ch)?.toLocaleString()} times in the dataset`}
          >
            <div className="font-mono text-base font-semibold">{ch}</div>
            <div className="font-mono text-[10px] text-muted">{id}</div>
            <div className="mx-auto mt-1 flex h-6 w-1.5 items-end rounded-sm bg-ink/10">
              <div
                className="w-full rounded-sm"
                style={{
                  height: `${Math.max(4, Math.round(((charCounts.get(ch) ?? 0) / maxCount) * 100))}%`,
                  background: 'var(--neg)',
                }}
              />
            </div>
          </div>
        ))}
        <div className="rounded-md border p-1.5 text-center" style={{ borderColor: 'var(--neg)', background: 'rgba(25,113,194,0.07)' }}>
          <div className="font-mono text-base font-semibold" style={{ color: 'var(--neg)' }}>
            ·
          </div>
          <div className="font-mono text-[10px]" style={{ color: 'var(--neg)' }}>
            26
          </div>
          <div className="mt-1 font-mono text-[9px] text-muted">BOS</div>
        </div>
      </div>
      <p className="mt-2 font-mono text-xs text-muted">
        the full vocabulary — bars show how often each letter appears across all{' '}
        {names.length.toLocaleString()} names
      </p>
    </div>
  )
}

function RoundTrip() {
  const example = useAppStore((s) => s.example)
  const setExample = useAppStore((s) => s.setExample)
  const [raw, setRaw] = useState(example)
  const { setHighlight } = useCodeSync()

  const cleaned = useMemo(
    () => [...raw.toLowerCase()].filter((c) => tok.isInVocab(c)).join('').slice(0, 14),
    [raw],
  )
  const rejected = cleaned.length < [...raw].length
  const tokens = useMemo(() => tok.encodeDoc(cleaned), [cleaned])

  return (
    <div className="not-prose my-4 rounded-lg border border-ink/15 bg-white/60 p-4">
      <label htmlFor="roundtrip" className="font-mono text-xs text-muted">
        type anything (a–z survive; everything else is not in the vocabulary):
      </label>
      <input
        id="roundtrip"
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value)
          const c = [...e.target.value.toLowerCase()].filter((ch) => tok.isInVocab(ch)).join('').slice(0, 14)
          if (c.length > 0) setExample(c)
          setHighlight([157])
        }}
        onFocus={() => setHighlight([24, 25, 26])}
        className="mt-1 w-full rounded border border-ink/20 bg-white px-3 py-2 font-mono text-lg focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
        maxLength={24}
        spellCheck={false}
      />
      {rejected && (
        <p className="mt-1 font-mono text-xs" style={{ color: 'var(--pos)' }}>
          dropped characters outside the 26-letter vocabulary — the file would crash on
          them (uchars.index raises ValueError)
        </p>
      )}
      <div className="mt-3">
        <div className="font-mono text-xs text-muted">
          tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        </div>
        <TokenTape tokens={tokens} labelOf={labelOf} />
      </div>
      <div className="mt-2 font-mono text-sm">
        decode back: <span className="font-semibold">{cleaned || '∅'}</span>
        <span className="text-muted"> — a lossless round trip</span>
      </div>
      <p className="mt-2 text-xs text-muted">
        this word is now your running example everywhere in the app (chapters 4, 5 and 10 will use it)
      </p>
    </div>
  )
}

function BosStory() {
  const emma = tok.encodeDoc('emma')
  return (
    <div className="not-prose my-4">
      <TokenTape tokens={emma} labelOf={labelOf} showTargets />
      <p className="mt-2 text-sm text-muted">
        training pairs for &quot;emma&quot; — each token must predict the <em>next</em> one
      </p>
    </div>
  )
}

const chapter = CHAPTERS[1]!

export default function Ch01() {
  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Everything starts with <code>input.txt</code>: {names.length.toLocaleString()} real
        names ({names.slice(0, 5).join(', ')}, …), one per line. Line 20 shuffles them
        once — and with <code>random.seed(42)</code> fixed on line 12, everyone who runs
        the file gets the same shuffle, the same initial weights, the same 20 generated
        names at the end. That determinism is what lets this app check every number it
        shows you against the real thing.
      </p>

      <h2>The vocabulary is discovered, not designed</h2>
      <p>
        Line 24 is the whole tokenizer:{' '}
        <code>uchars = sorted(set(&apos;&apos;.join(docs)))</code>. Join every name into one
        long string, keep the unique characters, sort them. The table below is computed
        live from the real dataset with exactly that recipe:
      </p>
      <VocabTable />
      <p>
        A character&apos;s <Term t="token">token</Term> id is just its position in this
        sorted list — <code>a</code> is 0 because nothing sorts before it. Then line 25
        mints one extra id: <code>BOS = len(uchars)</code>, a 27th token that appears in
        no name.
      </p>

      <h2>Round trip: chars ⇄ ids</h2>
      <RoundTrip />

      <h2>Why BOS brackets both sides</h2>
      <p>
        Line 157 wraps every training document as <code>[BOS] + chars + [BOS]</code>. The
        two ends do different jobs. Read the arrows:
      </p>
      <BosStory />
      <p>
        The first pair (<code>· → e</code>) teaches the model <em>how names begin</em>:
        given only the start signal, predict a plausible first letter. The last pair
        (<code>a → ·</code>) teaches it <em>when to stop</em>: after seeing
        e‑m‑m‑a, the right prediction is <Term t="BOS">BOS</Term>. At sampling time
        (chapter 10) that emission is exactly what ends a generated name — there is no
        length counter anywhere in the file.
      </p>

      <Aside kind="wild" title="Real GPTs tokenize subwords, not characters">
        Production models use byte-pair encoding (BPE): frequent character chunks
        (&quot;the&quot;, &quot;ing&quot;, &quot; name&quot;) become single tokens, giving
        vocabularies of 50k–200k ids and much shorter sequences. Same idea — text becomes
        integer ids from a fixed vocabulary — just a smarter dictionary. GPT-2 would carve
        &quot;emma&quot; into one or two tokens, not four. Real models also carry several
        special tokens (end-of-text, message separators), not a single BOS.
      </Aside>

      <PredictReveal
        qid="ch1-id-of-m"
        question={<>Without counting on your fingers: which id does &apos;m&apos; get?</>}
        options={['11', '12', '13']}
        answerIndex={1}
        explanation={
          <>
            &apos;m&apos; is the 13th letter and ids start at 0 — so 12. Verify in the
            round-trip box: emma is [26, <strong>4, 12, 12, 0</strong>, 26]. Position in
            the sorted unique-character list <em>is</em> the id; there is no other lookup
            table anywhere.
          </>
        }
      />
      <PredictReveal
        qid="ch1-digits-vocab"
        question={<>If the dataset also contained the ten digits 0–9, what would vocab_size be?</>}
        options={['27', '36', '37']}
        answerIndex={2}
        explanation={
          <>
            26 letters + 10 digits = 36 unique characters → ids 0–35 (digits sort before
            letters!), and BOS becomes id 36, so{' '}
            <code>vocab_size = len(uchars) + 1 = 37</code>. Everything downstream — wte
            rows, lm_head rows, the softmax width, the ln(vocab_size) starting loss —
            resizes automatically from this one number.
          </>
        }
      />
      <PredictReveal
        qid="ch1-no-right-bos"
        question={<>Suppose line 157 only prepended BOS: tokens = [BOS] + chars. What breaks?</>}
        options={['training crashes on short names', 'generated names never learn to end', 'the vocabulary shrinks to 26']}
        answerIndex={1}
        explanation={
          <>
            The model would never see one example of &quot;emit BOS when the name is
            done.&quot; At sampling time the stop condition{' '}
            <code>if token_id == BOS: break</code> would almost never fire, and every
            sample would run into the 16-token block_size wall. Stopping is not built in —
            it is <em>learned</em>, from the right-hand BOS, like any other prediction.
          </>
        }
      />

      <Recap
        chapterId={1}
        points={[
          <>The tokenizer is one line: sorted unique characters; an id is a position in that list.</>,
          <>BOS (id 26) appears in no name; it is both the start prompt and the learned stop signal — hence [BOS] + chars + [BOS].</>,
          <>seed(42) makes the file fully deterministic, which is why this app can be parity-tested against it.</>,
        ]}
      />
    </ChapterFrame>
  )
}
