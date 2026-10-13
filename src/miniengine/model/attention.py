import numpy as np
from transformers import GPT2LMHeadModel

hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
W = {k: v.detach().numpy().astype(np.float32) for k, v in hf.state_dict().items()}

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def attention(x, W, layer, n_heads=12):
    p = f"transformer.h.{layer}.attn."
    T, C = x.shape              #5, 768
    d = C // n_heads            #64

    qkv = x @ W[p + "c_attn.weight"] + W[p +"c_attn.bias"]
    q, k, v = np.split(qkv, 3, axis=-1)

    def split_heads(t):
        return t.reshape(T, n_heads, d).transpose(1, 0, 2)
    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d)           
    future = np.triu(np.ones((T, T), dtype=bool), k=1)       
    scores = np.where(future, -np.inf, scores)               
    weights = softmax(scores)                                

    out = weights @ v                                        
    out = out.transpose(1, 0, 2).reshape(T, C)               
    return out @ W[p + "c_proj.weight"] + W[p + "c_proj.bias"]