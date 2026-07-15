/**
 * Scalar autograd engine — an exact TypeScript twin of microgpt.py's `Value`.
 *
 * Semantics mirrored precisely (see PLAN.md "Engine semantics"):
 * - Local gradients are computed and stored AT CONSTRUCTION TIME of each node.
 * - `backward()` does a DFS post-order topological sort from the output node,
 *   seeds `grad = 1`, then walks the topo order in reverse accumulating
 *   `child.grad += local_grad * v.grad`. The `+=` is what makes nodes with
 *   multiple parents (residual connections!) receive gradient from every path.
 * - Python operator desugaring is reproduced literally:
 *     -x      →  x * -1                    (a mul node with a constant)
 *     a - b   →  a + (b * -1)  [Value b]   /  a + (-b)  [number b]
 *     a / b   →  a * b**-1     [Value b: a pow node]  /  a * (1/b)  [number b]
 *     sum(xs) →  starts from a constant 0 node (Python's sum() + __radd__)
 * - Only number exponents are supported by pow, exactly like `Value.__pow__`.
 *
 * The one deliberate difference: the topo sort is ITERATIVE (explicit stack,
 * same visit order as the Python recursion) so huge graphs can't overflow the
 * JS call stack. Values and gradients are identical.
 *
 * This engine exists for pedagogy — chapter 2's live computation graphs and
 * small worked examples — and as the high-precision (f64) parity reference.
 * The full model runs on the vectorized twin in tensor.ts.
 */

export type Op = 'const' | 'add' | 'mul' | 'pow' | 'log' | 'exp' | 'relu'

let NEXT_ID = 0

export class V {
  /** scalar value of this node, calculated during the forward pass */
  data: number
  /** derivative of the loss w.r.t. this node, calculated in backward() */
  grad = 0
  /** children of this node in the computation graph */
  readonly children: readonly V[]
  /** local derivative of this node w.r.t. each child, stored at construction */
  readonly localGrads: readonly number[]
  /** which operation produced this node — for drawing graphs, not for math */
  readonly op: Op
  /** stable id so UI code can key nodes */
  readonly id: number

  constructor(data: number, children: readonly V[] = [], localGrads: readonly number[] = [], op: Op = 'const') {
    this.data = data
    this.children = children
    this.localGrads = localGrads
    this.op = op
    this.id = NEXT_ID++
  }

  add(other: V | number): V {
    const o = other instanceof V ? other : new V(other)
    return new V(this.data + o.data, [this, o], [1, 1], 'add')
  }

  mul(other: V | number): V {
    const o = other instanceof V ? other : new V(other)
    return new V(this.data * o.data, [this, o], [o.data, this.data], 'mul')
  }

  /** number exponents only, exactly like Value.__pow__ */
  pow(exponent: number): V {
    return new V(this.data ** exponent, [this], [exponent * this.data ** (exponent - 1)], 'pow')
  }

  log(): V {
    return new V(Math.log(this.data), [this], [1 / this.data], 'log')
  }

  exp(): V {
    const e = Math.exp(this.data)
    return new V(e, [this], [e], 'exp')
  }

  relu(): V {
    return new V(Math.max(0, this.data), [this], [this.data > 0 ? 1 : 0], 'relu')
  }

  /** Python: __neg__ = self * -1 (a real mul node, not a special case) */
  neg(): V {
    return this.mul(-1)
  }

  /** Python: __sub__ = self + (-other) */
  sub(other: V | number): V {
    return other instanceof V ? this.add(other.mul(-1)) : this.add(-other)
  }

  /** Python: __truediv__ = self * other**-1 (pow node for Value denominators) */
  div(other: V | number): V {
    return other instanceof V ? this.mul(other.pow(-1)) : this.mul(other ** -1)
  }

  /**
   * Topological sort (DFS post-order, children first), identical visit order
   * to the reference's recursive build_topo, but with an explicit stack.
   */
  topo(): V[] {
    const topo: V[] = []
    const visited = new Set<V>()
    const stack: Array<{ node: V; i: number }> = []
    visited.add(this)
    stack.push({ node: this, i: 0 })
    while (stack.length > 0) {
      const frame = stack[stack.length - 1]!
      if (frame.i < frame.node.children.length) {
        const child = frame.node.children[frame.i]!
        frame.i++
        if (!visited.has(child)) {
          visited.add(child)
          stack.push({ node: child, i: 0 })
        }
      } else {
        stack.pop()
        topo.push(frame.node)
      }
    }
    return topo
  }

  backward(): void {
    const topo = this.topo()
    this.grad = 1
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i]!
      for (let c = 0; c < v.children.length; c++) {
        v.children[c]!.grad += v.localGrads[c]! * v.grad
      }
    }
  }
}

/**
 * Python's sum() over Values: `0 + x0` dispatches to `x0.__radd__(0)` which
 * builds `x0 + 0` (x0 as first child, constant 0 second); every later step is
 * a plain left-fold `acc + xi`. Mirrored exactly so graph shape, topo order
 * and float accumulation order all match the reference.
 */
export function vsum(xs: readonly V[]): V {
  if (xs.length === 0) return new V(0)
  let acc = xs[0]!.add(0)
  for (let i = 1; i < xs.length; i++) acc = acc.add(xs[i]!)
  return acc
}

/** number - Value, Python's __rsub__: other + (-self) */
export function rsub(other: number, x: V): V {
  return new V(other).add(x.mul(-1))
}

/** number / Value, Python's __rtruediv__: other * self**-1 */
export function rdiv(other: number, x: V): V {
  return new V(other).mul(x.pow(-1))
}
