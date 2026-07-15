/**
 * Generic step-through controller: play/pause/step/scrub, keyboard-driven
 * (←/→ step, space toggles). Everything animated in the app is scrubbable;
 * with prefers-reduced-motion, autoplay is disabled and stepping is manual.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

export interface StepPlayerState {
  index: number
  playing: boolean
  seek: (i: number) => void
  next: () => void
  prev: () => void
  toggle: () => void
  stop: () => void
}

export function useStepPlayer(length: number, stepsPerSecond = 1): StepPlayerState {
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const raf = useRef<number>(0)
  const lastTick = useRef<number>(0)

  const seek = useCallback(
    (i: number) => setIndex(Math.max(0, Math.min(length - 1, i))),
    [length],
  )
  const next = useCallback(() => setIndex((i) => Math.min(length - 1, i + 1)), [length])
  const prev = useCallback(() => setIndex((i) => Math.max(0, i - 1)), [])
  const stop = useCallback(() => setPlaying(false), [])
  const toggle = useCallback(() => {
    setPlaying((p) => {
      if (!p && index >= length - 1) setIndex(0) // restart from the beginning
      return !p
    })
  }, [index, length])

  useEffect(() => {
    if (!playing) return
    const stepMs = 1000 / stepsPerSecond
    lastTick.current = performance.now()
    const loop = (t: number) => {
      if (t - lastTick.current >= stepMs) {
        lastTick.current = t
        setIndex((i) => {
          if (i + 1 >= length) {
            setPlaying(false)
            return i
          }
          return i + 1
        })
      }
      raf.current = requestAnimationFrame(loop)
    }
    raf.current = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf.current)
  }, [playing, length, stepsPerSecond])

  return { index, playing, seek, next, prev, toggle, stop }
}

const btn =
  'rounded border border-ink/20 px-2.5 py-1 font-mono text-xs hover:bg-ink/5 ' +
  'focus-visible:outline-2 focus-visible:outline-[var(--hot)] disabled:opacity-30'

export function StepPlayer({
  player, length, label, format,
}: {
  player: StepPlayerState
  length: number
  /** accessible name for this player, e.g. "position stepper" */
  label: string
  /** format the current index for display, e.g. (i) => `pos ${i}` */
  format?: (i: number) => string
}) {
  const reduced =
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowRight') {
      e.preventDefault()
      player.next()
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault()
      player.prev()
    } else if (e.key === ' ') {
      e.preventDefault()
      if (!reduced) player.toggle()
    }
  }

  return (
    <div
      role="group"
      aria-label={label}
      onKeyDown={onKey}
      className="flex flex-wrap items-center gap-2 rounded-lg border border-ink/10 bg-white/50 px-3 py-2"
    >
      <button type="button" className={btn} onClick={() => player.seek(0)} disabled={player.index === 0} aria-label="restart">
        ⏮
      </button>
      <button type="button" className={btn} onClick={player.prev} disabled={player.index === 0} aria-label="step back">
        ←
      </button>
      {!reduced && (
        <button type="button" className={btn} onClick={player.toggle} aria-label={player.playing ? 'pause' : 'play'}>
          {player.playing ? '⏸' : '▶'}
        </button>
      )}
      <button
        type="button"
        className={btn}
        onClick={player.next}
        disabled={player.index >= length - 1}
        aria-label="step forward"
      >
        →
      </button>
      <input
        type="range"
        min={0}
        max={length - 1}
        value={player.index}
        onChange={(e) => player.seek(Number(e.target.value))}
        aria-label={`${label} scrubber`}
        className="h-1 min-w-24 flex-1 accent-[var(--neg)]"
      />
      <span className="min-w-16 text-right font-mono text-xs tabular-nums text-muted">
        {format ? format(player.index) : `${player.index + 1}/${length}`}
      </span>
    </div>
  )
}
