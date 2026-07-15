/**
 * The untrained ↔ trained toggle (brief §8). Sets the app-wide checkpoint so
 * every live visualization in the chapter re-renders from the other weights.
 */
import { useAppStore } from '../app/store.ts'

export function CompareToggle() {
  const step = useAppStore((s) => s.checkpointStep)
  const setStep = useAppStore((s) => s.setCheckpointStep)
  const options: Array<{ label: string; value: number }> = [
    { label: 'untrained (step 0)', value: 0 },
    { label: 'trained (step 1000)', value: 1000 },
  ]
  return (
    <div role="radiogroup" aria-label="model checkpoint" className="inline-flex rounded-lg border border-ink/20 p-0.5">
      {options.map((o) => {
        const active = step === o.value || (o.value === 1000 && step > 0)
        return (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => setStep(o.value)}
            className="rounded-md px-3 py-1 font-mono text-xs transition-colors focus-visible:outline-2 focus-visible:outline-[var(--hot)]"
            style={{
              background: active ? 'var(--ink)' : 'transparent',
              color: active ? 'var(--paper)' : 'var(--muted)',
            }}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
