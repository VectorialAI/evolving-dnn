"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

# THIS FILE IS COPIED FROM https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# MINOR CHANGES MADE TO BE USED WITH torch.fx and be more readable

import logging
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import CfgNode as CN

# -----------------------------------------------------------------------------

@torch.fx.wrap  # TODO remove this if we want it expanded so activation function itself can evolve
def new_gelu_function(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return new_gelu_function(x)

# Wrapped to avoid output errors immediately following masked_fill which adds -inf to the logits
@torch.fx.wrap
def _masked_fill_softmax(input: torch.Tensor, mask: torch.Tensor, value: float, dim: int) -> torch.Tensor:
    return F.softmax(input.masked_fill(mask, value), dim=dim)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.is_proxy_for_fx = config.is_proxy_for_fx
        self.block_size = config.block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        if self.is_proxy_for_fx:
            sequence_length = self.block_size

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        attn_output = self.c_attn(x)
        q = attn_output.narrow(2, 0, self.n_embd)
        k = attn_output.narrow(2, self.n_embd, self.n_embd)
        v = attn_output.narrow(2, 2 * self.n_embd, self.n_embd)
        k = k.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = _masked_fill_softmax(att, self.bias[:,:,:sequence_length,:sequence_length] == 0, float('-inf'), dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = _transpose_contiguous(y, 1, 2).view(batch_size, sequence_length, embedding_dim) # re-assemble all head outputs side by side
        # y = y.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

@torch.fx.wrap
def _transpose_contiguous(x: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return x.transpose(dim0, dim1).contiguous()

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.is_proxy_for_fx = False
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.block_size = config.block_size

        assert all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        self.is_proxy_for_fx = config.is_proxy_for_fx

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        device = idx.device
        batch_size, sequence_length = idx.size()
        if self.is_proxy_for_fx:
            device = torch.device("cpu")
            sequence_length = self.block_size
        assert sequence_length <= self.block_size, f"Cannot forward sequence of length {sequence_length}, block size is only {self.block_size}"
        pos = torch.arange(0, sequence_length, dtype=torch.long, device=device).unsqueeze(0) # shape (1, sequence_length)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (batch_size, sequence_length, embedding_dim)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, sequence_length, embedding_dim)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx