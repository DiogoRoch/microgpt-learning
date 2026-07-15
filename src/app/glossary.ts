/**
 * Glossary for Term popovers. Definitions speak this file's language —
 * "in microgpt.py, X is …" — never generic textbook prose.
 */
export const GLOSSARY: Record<string, string> = {
  token:
    'An integer standing for a character. microgpt has 27: ids 0–25 are the sorted letters a–z, id 26 is BOS.',
  BOS:
    "The Beginning-of-Sequence token, id 26 — one id past the last letter. Every training doc is [BOS] + chars + [BOS]: the left one gives the model a start signal, the right one teaches it to say 'stop'.",
  logits:
    'The 27 raw scores the model outputs at each position (linear(x, lm_head)), one per vocab token. Softmax turns them into probabilities.',
  softmax:
    'exp of each score divided by the sum of exps — scores become probabilities that sum to 1. microgpt subtracts the max first for numerical stability; that shift cancels out exactly.',
  loss:
    'The average of -log p(correct next token) over the document. Guessing uniformly gives ln(27) ≈ 3.30; a perfect model would approach 0.',
  gradient:
    "The derivative of the loss with respect to a number — 'if this parameter grew a little, how would the loss change?' backward() computes one for all 4,192 parameters.",
  rmsnorm:
    'Rescales a vector to unit root-mean-square: x · (mean(x²)+1e-5)^-0.5. The GPT-2 recipe uses layernorm with learnable scale; microgpt deliberately uses this simpler cousin.',
  'residual stream':
    "The running vector x that each block reads from and adds back into (x = block(x) + x). It's why gradients need += accumulation: two paths write to the same node.",
  embedding:
    "A learned 16-number vector for each token (wte) or position (wpe). 'emma' starts as wte[4] + wpe[1] at position 1.",
  'KV cache':
    'The growing lists keys[] and values[] that gpt() appends to at each position. Attention looks only at what these lists already contain — the future simply is not in them yet.',
  head:
    'One of 4 independent attention patterns. Each head works on its own 4-dim slice of the 16-dim q/k/v vectors (head_dim = n_embd / n_head).',
  temperature:
    'Divides the logits before softmax at sampling time. Below 1 it sharpens the distribution toward the argmax; near 0 it becomes deterministic; at 1 it samples the raw distribution.',
  Adam:
    'The optimizer: per-parameter running means of gradient (m) and squared gradient (v), bias-corrected, used to scale each update. microgpt uses lr=0.01 with linear decay to 0.',
  parameters:
    'The 4,192 numbers in state_dict that training changes. Everything the model "knows" lives in them.',
  'cross-entropy':
    '-log p(target): the loss for one position. Low when the model gave the true next token high probability.',
}
