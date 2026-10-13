import numpy as np
from transformers import GPT2LMHeadModel

hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
W = {k: v.detach().numpy().astype(np.float32) for k, v in hf.state_dict().items()}

def embed(ids):
    return W["transformer.wte.weight"][ids] + W["transformer.wpe.weight"][np.arange(len(ids))]

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean)/np.sqrt(var + eps)
    return x_hat * gamma + beta

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def mlp(x, W, layer):
    p = f"transformer.h.{layer}.mlp."
    h = x @ W[p + "c_fc.weight"] + W[p + "c_fc.bias"]        
    h = gelu(h)
    return h @ W[p + "c_proj.weight"] + W[p + "c_proj.bias"]  