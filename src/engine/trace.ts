/**
 * Trace types — the shared shape for "every named intermediate of one gpt()
 * call". These mirror golden/step0_trace.json exactly (same names as the
 * variables in microgpt.py), so the same types serve parity tests, the trace
 * recorder, and every chapter UI. Record on demand only: tracing every step
 * would burn memory for nothing.
 */

export interface HeadTrace {
  /** scaled dot products q_h·k_h[t] / √head_dim, one per visible position */
  attn_logits: number[]
  /** softmax(attn_logits) — at pos 0 this is exactly [1.0] */
  attn_weights: number[]
  /** Σ_t attn_weights[t] · v_h[t] */
  head_out: number[]
}

export interface LayerTrace {
  x_ln_attn: number[]
  q: number[]
  k: number[]
  v: number[]
  heads: HeadTrace[]
  x_attn: number[]
  x_wo: number[]
  x_after_attn: number[]
  x_ln_mlp: number[]
  fc1: number[]
  relu: number[]
  fc2: number[]
  x_after_mlp: number[]
}

export interface GptCallTrace {
  token_id: number
  pos_id: number
  tok_emb: number[]
  pos_emb: number[]
  x_emb_sum: number[]
  x_emb_norm: number[]
  layers: LayerTrace[]
  logits: number[]
}

export interface PositionLoss {
  token_id: number
  target_id: number
  probs: number[]
  loss_t: number
}

/** One traced forward pass over a whole document. */
export interface DocTrace {
  tokens: number[]
  n: number
  gpt_calls: GptCallTrace[]
  positions: PositionLoss[]
  loss: number
}
