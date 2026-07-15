/**
 * The scrubbable loss curve. Draws the training trajectory (live or
 * precomputed), the Python reference overlay, the ln(27) cold-start line,
 * and a scrub cursor that reports the hovered/dragged step.
 */
import { useRef, useState } from 'react'
import { fmt } from './color.ts'

export interface LossCurveProps {
  losses: readonly number[]
  /** total x-axis width in steps (so a live, partial curve draws left-aligned) */
  totalSteps: number
  /** optional second curve (the Python reference run) */
  overlay?: readonly number[]
  /** current scrub position */
  step?: number | null
  onScrub?: (step: number) => void
  height?: number
}

const LN27 = Math.log(27)

export function LossCurve({ losses, totalSteps, overlay, step, onScrub, height = 180 }: LossCurveProps) {
  const W = 640
  const H = height
  const PAD = { l: 34, r: 8, t: 8, b: 18 }
  const [hover, setHover] = useState<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  const yMax = 3.6
  const yMin = 1.6
  const x = (s: number) => PAD.l + (s / Math.max(1, totalSteps - 1)) * (W - PAD.l - PAD.r)
  const y = (l: number) => PAD.t + (1 - (l - yMin) / (yMax - yMin)) * (H - PAD.t - PAD.b)

  const path = (data: readonly number[]) =>
    data.map((l, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(Math.min(yMax, Math.max(yMin, l))).toFixed(1)}`).join(' ')

  const stepFromEvent = (e: React.PointerEvent) => {
    const rect = svgRef.current!.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const s = Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (totalSteps - 1))
    return Math.max(0, Math.min(totalSteps - 1, s))
  }

  const cursor = step ?? hover

  return (
    <svg
      ref={svgRef}
      width="100%"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="training loss curve, scrubbable"
      style={{ touchAction: 'none', cursor: onScrub ? 'crosshair' : 'default' }}
      onPointerMove={(e) => {
        const s = stepFromEvent(e)
        setHover(s)
        if (e.buttons === 1) onScrub?.(s)
      }}
      onPointerDown={(e) => onScrub?.(stepFromEvent(e))}
      onPointerLeave={() => setHover(null)}
    >
      {/* axes + ln27 marker */}
      {[2.0, 2.5, 3.0, 3.5].map((l) => (
        <g key={l}>
          <line x1={PAD.l} y1={y(l)} x2={W - PAD.r} y2={y(l)} stroke="rgba(22,24,29,0.08)" />
          <text x={PAD.l - 4} y={y(l) + 3} textAnchor="end" fontSize={9} className="font-mono" fill="var(--muted)">
            {l.toFixed(1)}
          </text>
        </g>
      ))}
      <line x1={PAD.l} y1={y(LN27)} x2={W - PAD.r} y2={y(LN27)} stroke="var(--pos)" strokeDasharray="4 3" strokeWidth={1} />
      <text x={W - PAD.r} y={y(LN27) - 4} textAnchor="end" fontSize={9} className="font-mono" fill="var(--pos)">
        ln 27 ≈ 3.296 — uniform guessing
      </text>
      {/* overlay (python) then main curve */}
      {overlay && overlay.length > 1 && <path d={path(overlay)} fill="none" stroke="rgba(22,24,29,0.25)" strokeWidth={2.5} />}
      {losses.length > 1 && <path d={path(losses)} fill="none" stroke="var(--neg)" strokeWidth={1.5} />}
      {/* cursor */}
      {cursor != null && cursor < losses.length && (
        <g>
          <line x1={x(cursor)} y1={PAD.t} x2={x(cursor)} y2={H - PAD.b} stroke="var(--hot)" strokeWidth={1.5} />
          <circle cx={x(cursor)} cy={y(Math.min(yMax, Math.max(yMin, losses[cursor]!)))} r={3.5} fill="var(--hot)" />
          <text x={Math.min(x(cursor) + 6, W - 130)} y={PAD.t + 10} fontSize={10} className="font-mono" fill="var(--ink)">
            step {cursor + 1} · loss {fmt(losses[cursor]!, 4)}
          </text>
        </g>
      )}
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize={9} className="font-mono" fill="var(--muted)">
        step (drag to scrub)
      </text>
    </svg>
  )
}
