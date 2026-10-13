"""
Compare your own GPT-2 forward pass against HuggingFace, token by token.

Setup:  pip install torch transformers numpy
Edit the one import below so it points at YOUR forward function.
Your forward(ids) must take a list of token ids and return logits of
shape (len(ids), vocab_size)  -- one row of scores per input position.
"""
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from miniengine.model.gpt2 import forward   

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "def fibonacci(n):",
    "In 2020, scientists discovered that",
]
N_NEW_TOKENS = 20
TOL = 1e-3

tok = GPT2TokenizerFast.from_pretrained("gpt2")
hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()   # float32 by default


def to_np(x):
    """Accept numpy arrays or torch tensors from your forward()."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@torch.no_grad()
def hf_logits(ids):
    return hf(torch.tensor([ids])).logits[0].numpy()    


# ---------------------------------------------------------------
# CHECK 1: one forward pass, compare logits at EVERY position
# ---------------------------------------------------------------
print("=" * 70)
print("CHECK 1: logits at every position (single forward pass)")
print("=" * 70)
for p in PROMPTS:
    ids = tok.encode(p)
    a, b = to_np(forward(ids)), hf_logits(ids)
    if a.shape != b.shape:
        print(f"{p!r}: SHAPE MISMATCH mine={a.shape} hf={b.shape}")
        continue
    diff = np.abs(a - b).max()
    argmax_ok = (a.argmax(-1) == b.argmax(-1)).all()
    status = "OK " if diff < TOL and argmax_ok else "BAD"
    print(f"[{status}] {p!r:35} max|diff|={diff:.2e}  argmax all positions match={argmax_ok}")

# ---------------------------------------------------------------
# CHECK 2: greedy generation, one token at a time
# ---------------------------------------------------------------
print()
print("=" * 70)
print("CHECK 2: greedy generation, token by token")
print("=" * 70)
for p in PROMPTS:
    ids = tok.encode(p)
    print(f"\nPrompt: {p!r}")
    print(f"{'step':>4} {'HF tok':>8} {'mine':>8} {'last-pos diff':>14} {'top2 gap':>9}  text")
    first_bad = None
    for step in range(N_NEW_TOKENS):
        a = to_np(forward(ids))[-1]     # scores for the NEXT token
        b = hf_logits(ids)[-1]
        t_hf, t_me = int(b.argmax()), int(a.argmax())
        top2 = np.sort(b)[-2:]
        gap = top2[1] - top2[0]         # how decisive HF's choice was
        flag = "" if t_hf == t_me else "  <-- MISMATCH"
        if t_hf != t_me and first_bad is None:
            first_bad = step
        print(f"{step:>4} {t_hf:>8} {t_me:>8} {np.abs(a - b).max():>14.2e} {gap:>9.3f}  {tok.decode([t_hf])!r}{flag}")
        ids.append(t_hf)
    print("Result:", "all tokens match" if first_bad is None else f"first mismatch at step {first_bad}")

# ---------------------------------------------------------------
# CHECK 3: free-running. My own loop vs HF's generate().
# ---------------------------------------------------------------
print()
print("=" * 70)
print("CHECK 3: free-running greedy vs hf.generate()")
print("=" * 70)
for p in PROMPTS:
    ids = tok.encode(p)
    mine = list(ids)
    for _ in range(N_NEW_TOKENS):
        mine.append(int(to_np(forward(mine))[-1].argmax()))
    ref = hf.generate(torch.tensor([ids]), max_new_tokens=N_NEW_TOKENS,
                      do_sample=False, pad_token_id=tok.eos_token_id)[0].tolist()
    print(f"[{'OK ' if mine == ref else 'BAD'}] {p!r}")
    if mine != ref:
        print("   mine:", tok.decode(mine))
        print("   hf  :", tok.decode(ref))