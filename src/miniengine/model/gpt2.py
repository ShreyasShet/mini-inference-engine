from .layers import layer_norm, mlp, embed
from .attention import attention
from transformers import GPT2LMHeadModel
import numpy as np

hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
W = {k : v.detach().numpy().astype(np.float32) for k, v in hf.state_dict().items()}

def block(x, W, layer):
    p = f"transformer.h.{layer}."
    x = x + attention(layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"]), W, layer)
    x = x + mlp(layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"]), W, layer)
    return x

def forward(ids):
    x = embed(ids)                      
    for i in range(12):
        x = block(x, W, i)                 
    x = layer_norm(x, W["transformer.ln_f.weight"], W["transformer.ln_f.bias"])
    return x @ W["transformer.wte.weight"].T   