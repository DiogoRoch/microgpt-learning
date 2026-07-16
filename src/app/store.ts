/**
 * Cross-chapter state. Persisted to localStorage so progress (the minimap
 * filling in, every answered checkpoint) survives reloads.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/** Outcome of one checkpoint (quiz question / hands-on task). */
export interface QuizResult {
  /** 'correct' = solved it; 'revealed' = gave up and asked for the answer */
  status: 'correct' | 'revealed'
  /** how many wrong tries before resolving (0 = first try) */
  misses: number
}

interface AppState {
  /** chapter ids marked complete (all checkpoints resolved or manually advanced) */
  completed: number[]
  markCompleted: (id: number) => void
  /** per-checkpoint results, keyed by globally-unique qid (e.g. "ch5-no-mask") */
  quiz: Record<string, QuizResult>
  resolveQuiz: (qid: string, result: QuizResult) => void
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
      quiz: {},
      resolveQuiz: (qid, result) =>
        set((s) => (s.quiz[qid] ? s : { quiz: { ...s.quiz, [qid]: result } })),
      resetProgress: () => set({ completed: [], quiz: {} }),
      example: 'emma',
      setExample: (word) => set({ example: word }),
      checkpointStep: 1000,
      setCheckpointStep: (step) => set({ checkpointStep: step }),
    }),
    { name: 'microgpt-learning' },
  ),
)
