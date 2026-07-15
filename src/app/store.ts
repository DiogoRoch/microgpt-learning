/**
 * Cross-chapter state. Persisted to localStorage so progress (the minimap
 * filling in) survives reloads.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface AppState {
  /** chapter ids marked complete (all quiz reveals seen or manually advanced) */
  completed: number[]
  markCompleted: (id: number) => void
  resetProgress: () => void
  /** the user's running example word (must be lowercase a–z) */
  example: string
  setExample: (word: string) => void
  /** global model checkpoint the visualizations use: a step into the precomputed run */
  checkpointStep: number
  setCheckpointStep: (step: number) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      completed: [],
      markCompleted: (id) =>
        set((s) => (s.completed.includes(id) ? s : { completed: [...s.completed, id].sort((a, b) => a - b) })),
      resetProgress: () => set({ completed: [] }),
      example: 'emma',
      setExample: (word) => set({ example: word }),
      checkpointStep: 1000,
      setCheckpointStep: (step) => set({ checkpointStep: step }),
    }),
    { name: 'microgpt-learning' },
  ),
)
