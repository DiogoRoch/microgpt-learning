/**
 * The 27-way probability distribution over the vocabulary — the shape the
 * model's every prediction takes. Bars animate via CSS so temperature
 * changes and softmax sharpening read as motion of probability mass.
 */
import { fmt } from './color.ts'

export interface BarDistributionProps {
  probs: ArrayLike<number>
  /** bar labels — the vocab chars, BOS rendered as '·' */
  labels: readonly string[]
  /** index drawn in --hot (e.g. the sampled token or the target) */
  highlight?: number | null
  /** optional marker index (e.g. the target under teacher forcing) */
  marker?: number | null
  height?: number
  onClickBar?: (index: number) => void
  /** fix the y scale to [0,1] instead of max(probs) (default true) */
  fullScale?: boolean
}

export function BarDistribution({
  probs, labels, highlight, marker, height = 140, onClickBar, fullScale = true,
}: BarDistributionProps) {
  const n = probs.length
  let max = 0
  for (let i = 0; i < n; i++) max = Math.max(max, probs[i] as number)
  const scale = fullScale ? 1 : max || 1
  const bars = []
  for (let i = 0; i < n; i++) {
    const p = probs[i] as number
    const h = Math.max(1, (p / scale) * (height - 4))
    const isHot = highlight === i
    bars.push(
      <div key={i} className="flex flex-1 flex-col items-center justify-end gap-1" style={{ height }}>
        <span className="font-mono text-[9px] tabular-nums text-muted opacity-0 transition-opacity group-hover:opacity-100">
          {p >= 0.005 ? (p * 100).toFixed(0) + '%' : ''}
        </span>
        <button
          type="button"
          aria-label={`p(${labels[i]}) = ${fmt(p)}`}
          title={`p(${labels[i] ?? i}) = ${fmt(p)}`}
          onClick={onClickBar ? () => onClickBar(i) : undefined}
          tabIndex={onClickBar ? 0 : -1}
          className="w-full rounded-t-[2px] transition-[height] duration-300 motion-reduce:transition-none"
          style={{
            height: h,
            background: isHot ? 'var(--hot)' : marker === i ? 'var(--neg)' : 'rgba(22,24,29,0.55)',
            cursor: onClickBar ? 'pointer' : 'default',
          }}
        />
        <span
          className="font-mono text-[11px]"
          style={{ color: isHot || marker === i ? 'var(--ink)' : 'var(--muted)', fontWeight: isHot ? 600 : 400 }}
        >
          {labels[i]}
        </span>
      </div>,
    )
  }
  return <div className="group flex items-end gap-[2px]">{bars}</div>
}
