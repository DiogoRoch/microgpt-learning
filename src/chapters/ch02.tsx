/**
 * Chapter 2 — Autograd. Live computation graphs built with the real engine
 * (the exact twin of class Value), node-by-node backward stepping, and a
 * free-form expression sandbox.
 */
import { useEffect, useMemo, useState } from 'react'
import { CHAPTERS } from '../app/chapters.ts'
import { ChapterFrame } from '../components/ChapterFrame.tsx'
import { PredictReveal, Recap } from '../components/Quiz.tsx'
import { Term } from '../components/Term.tsx'
import { Aside } from '../components/Aside.tsx'
import { K } from '../components/Katex.tsx'
import { StepPlayer, useStepPlayer } from '../components/StepPlayer.tsx'
import { useCodeSync } from '../components/CodeSync.tsx'
import { GraphView, OP_LINE, partialGrads, planBackward } from '../viz/GraphView.tsx'
import { V } from '../engine/graph.ts'
import { parseExpr } from './exprParser.ts'

/** Name every node: leaves get their variable names, intermediates n₁, n₂ …, root gets outName. */
function autoNames(root: V, leaves: Map<string, V>, outName: string): Map<V, string> {
  const names = new Map<V, string>()
  for (const [name, v] of leaves) names.set(v, name)
  let i = 1
  for (const v of root.topo()) {
    if (v.children.length === 0 || names.has(v)) continue
    names.set(v, v === root ? outName : `n${i++}`)
  }
  names.set(root, outName)
  return names
}

/** Interactive backward walk over a real graph. */
function BackwardExplorer({
  root, names, showGradsFromStart = false,
}: {
  root: V
  names: Map<V, string>
  showGradsFromStart?: boolean
}) {
  const topo = useMemo(() => root.topo(), [root])
  const nameOf = useMemo(() => (v: V) => names.get(v) ?? (v.op === 'const' ? String(v.data) : v.op), [names])
  const plan = useMemo(() => planBackward(topo, nameOf), [topo, nameOf])
  const player = useStepPlayer(plan.captions.length, 0.7)
  const { setHighlight } = useCodeSync()
  const grads = useMemo(
    () => partialGrads(topo, plan.rawProcessed[player.index]!),
    [topo, plan, player.index],
  )
  const started = showGradsFromStart || player.index > 0

  useEffect(() => {
    setHighlight(player.index === 0 ? [69] : [70, 71, 72])
  }, [player.index, setHighlight])

  return (
    <div className="not-prose my-4 space-y-3">
      <GraphView
        root={root}
        names={names}
        grads={started ? grads : null}
        focus={plan.focus[player.index]}
        onHoverNode={(v) => {
          if (v && v.op !== 'const' && OP_LINE[v.op]) setHighlight(OP_LINE[v.op]!)
        }}
      />
      <StepPlayer player={player} length={plan.captions.length} label="backward stepper" format={(i) => `step ${i}`} />
      <p className="min-h-10 rounded bg-ink/5 px-3 py-2 font-mono text-[13px]">{plan.captions[player.index]}</p>
    </div>
  )
}

// --- the three story examples, built once with the real engine -------------

function exampleMul() {
  const a = new V(2)
  const b = new V(-3)
  const c = a.mul(b)
  return { root: c, names: autoNames(c, new Map([['a', a], ['b', b]]), 'c'), a, b }
}

function exampleChain() {
  const a = new V(2)
  const b = new V(-3)
  const c = new V(4)
  const d = a.mul(b).add(c.pow(2))
  return { root: d, names: autoNames(d, new Map([['a', a], ['b', b], ['c', c]]), 'd') }
}

function exampleShared() {
  const x = new V(3)
  const y = x.mul(x).add(x)
  return { root: y, names: autoNames(y, new Map([['x', x]]), 'y') }
}

function Sandbox() {
  const [src, setSrc] = useState('a*b + relu(c) - 1')
  const [vals, setVals] = useState<Record<string, number>>({ a: 2, b: -3, c: 4 })
  const parsed = useMemo(() => parseExpr(src, vals), [src, vals])
  const ok = !('error' in parsed)

  return (
    <div className="not-prose my-4 rounded-lg border border-ink/15 bg-white/60 p-4">
      <label htmlFor="expr" className="font-mono text-xs text-muted">
        your expression — + − × ÷, ** (literal exponents, like Value.__pow__), log, exp, relu:
      </label>
      <input
        id="expr"
        value={src}
        onChange={(e) => setSrc(e.target.value)}
        spellCheck={false}
        className="mt-1 w-full rounded border border-ink/20 bg-white px-3 py-2 font-mono focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
      />
      {!ok ? (
        <p className="mt-2 font-mono text-sm" style={{ color: 'var(--pos)' }}>
          {parsed.error}
        </p>
      ) : (
        <>
          {parsed.vars.size > 0 && (
            <div className="mt-3 flex flex-wrap gap-3">
              {[...parsed.vars.keys()].map((name) => (
                <label key={name} className="flex items-center gap-1.5 font-mono text-sm">
                  {name} =
                  <input
                    type="number"
                    step="any"
                    value={vals[name] ?? 1}
                    onChange={(e) => setVals((v) => ({ ...v, [name]: Number(e.target.value) }))}
                    className="w-20 rounded border border-ink/20 bg-white px-2 py-1 focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
                    aria-label={`value of ${name}`}
                  />
                </label>
              ))}
            </div>
          )}
          <div className="mt-3">
            <BackwardExplorer
              key={src + JSON.stringify(vals)}
              root={parsed.root}
              names={autoNames(parsed.root, parsed.vars, 'out')}
            />
          </div>
        </>
      )}
    </div>
  )
}

const chapter = CHAPTERS[2]!

export default function Ch02() {
  const ex1 = useMemo(exampleMul, [])
  const ex2 = useMemo(exampleChain, [])
  const ex3 = useMemo(exampleShared, [])

  return (
    <ChapterFrame chapter={chapter}>
      <p>
        Training needs one thing above all: for each of the 4,192 parameters, the answer
        to <em>&quot;if this number grew a little, would the loss go up or down, and how
        fast?&quot;</em> — the <Term t="gradient">gradient</Term>. The 40-line{' '}
        <code>Value</code> class computes all of them, exactly, by remembering how every
        number was made.
      </p>

      <h2>Numbers that remember their history</h2>
      <p>
        A <code>Value</code> wraps one number (<code>data</code>) plus three bookkeeping
        fields: which Values produced it (<code>_children</code>), the <em>local</em>{' '}
        derivative with respect to each (<code>_local_grads</code>), and a slot for the
        eventual answer (<code>grad</code>). Look at line 45: the moment{' '}
        <code>a * b</code> runs, the product node stores{' '}
        <code>(other.data, self.data)</code> — because ∂(ab)/∂a = b and ∂(ab)/∂b = a.
        The derivative rule is captured <strong>at construction time</strong>, while both
        inputs are right there. Below: a real graph from this app&apos;s engine, a = 2,
        b = −3. Edge labels are the stored local grads. Hover any node to see the exact
        line of the file that created it.
      </p>
      <BackwardExplorer root={ex1.root} names={ex1.names} />

      <h2>backward() is a walk through time, reversed</h2>
      <p>
        <code>backward()</code> (lines 59–72) does exactly two things: it lists every
        node so that children always come before parents (a topological sort), then walks
        that list <em>backwards</em>, applying one rule at every node:{' '}
        <code>child.grad += local_grad * v.grad</code>. That single line is the chain
        rule. Step through it on a bigger expression — d = a·b + c², so you can watch
        the grad of c become 2c = 8:
      </p>
      <BackwardExplorer root={ex2.root} names={ex2.names} />

      <Aside kind="math" title="Why child.grad += local · parent.grad is the chain rule">
        <p>
          If d depends on n, and n depends on c, then{' '}
          <K tex="\frac{\partial d}{\partial c} = \frac{\partial d}{\partial n}\cdot\frac{\partial n}{\partial c}" />
          . The walk keeps <K tex="\partial d/\partial n" /> in <code>n.grad</code> (parents
          are always finished first, thanks to the topological order), and{' '}
          <K tex="\partial n/\partial c" /> is the stored local grad. When several paths
          lead from c to d, the total derivative is the <em>sum</em> over paths:{' '}
          <K
            block
            tex="\frac{\partial d}{\partial c} = \sum_{\text{paths } p} \prod_{\text{edges} \in p} (\text{local grad})"
          />
          — which is exactly what <code>+=</code> accumulates, one edge at a time.
        </p>
      </Aside>

      <h2>The += that will power residuals</h2>
      <p>
        Why <code>+=</code> and not <code>=</code>? Because a node can feed into several
        places, and each route contributes derivative. Watch x&apos;s grad build up in{' '}
        <strong>y = x·x + x</strong> at x = 3: the multiply sends it 3 <em>twice</em>{' '}
        (once as each factor), the add sends 1. Total: 7 = 2x + 1.
      </p>
      <BackwardExplorer root={ex3.root} names={ex3.names} />
      <p>
        Keep this picture. In chapter 6, the transformer&apos;s{' '}
        <Term t="residual stream">residual stream</Term> does{' '}
        <code>x = block(x) + x</code> — the same shared-node pattern, at scale. Without
        this <code>+=</code>, gradients flowing through the shortcut would overwrite the
        ones flowing through the block, and training would silently break.
      </p>

      <h2>Sandbox: differentiate anything</h2>
      <p>
        The parser below builds graphs with this app&apos;s <code>Value</code>-twin —
        including Python&apos;s desugarings (<code>-x</code> becomes <code>x * -1</code>,{' '}
        <code>a / b</code> becomes <code>a * b**-1</code>; find them on lines 51–57).
      </p>
      <Sandbox />

      <Aside kind="wild" title="PyTorch is this class, industrialized">
        Swap &quot;one number&quot; for &quot;a tensor of millions of numbers&quot;, the
        stored local grads for closed-form backward functions per op, and the recursive
        topo sort for an iterative one, and you have the autograd at the heart of PyTorch
        and JAX. This app&apos;s fast engine (which trains the model in chapter 9) makes
        exactly that swap — and is tested to agree with the scalar graph to five decimal
        places.
      </Aside>

      <PredictReveal
        qid="ch2-shared-grad"
        question={<>After y.backward() on y = x·x + x with x = 3, what is x.grad?</>}
        options={['6', '7', '1']}
        answerIndex={1}
        explanation={
          <>
            Three contributions accumulate: the multiply node&apos;s stored local grads are
            (3, 3) — x is both factors — and the add contributes 1. So 3 + 3 + 1 ={' '}
            <strong>7</strong>, matching the calculus answer 2x + 1. You just watched it
            happen step by step above.
          </>
        }
      />
      <PredictReveal
        qid="ch2-relu-dead"
        question={<>relu(-2) stores what local gradient (line 50)?</>}
        options={['0', '1', '-2']}
        answerIndex={0}
        explanation={
          <>
            <code>float(self.data &gt; 0)</code> → 0.0 for a negative input. Whatever
            gradient arrives at a dead ReLU stops there — nothing flows to its input. In
            the MLP (chapter 6) that means a neuron that stays negative for an input
            simply doesn&apos;t learn from that example.
          </>
        }
      />
      <PredictReveal
        qid="ch2-mul-grad"
        question={<>c = a·b with a = 2, b = −3. After c.backward(), a.grad is…</>}
        options={['−3', '2', '−6']}
        answerIndex={0}
        explanation={
          <>
            ∂(ab)/∂a = b = −3 — the value stored in the node&apos;s local grads the moment
            it was created (line 45). Increase a slightly and c decreases at 3× the rate:
            the gradient is a sensitivity, and here it&apos;s negative.
          </>
        }
      />

      <Recap
        chapterId={2}
        points={[
          <>Every op stores its local derivatives at construction — the graph remembers how each number was made.</>,
          <>backward() = topological sort + one rule in reverse: child.grad += local_grad × parent.grad. That line is the chain rule.</>,
          <>+= accumulation is what makes shared nodes (like the upcoming residual stream) receive gradient from every path.</>,
        ]}
      />
    </ChapterFrame>
  )
}
