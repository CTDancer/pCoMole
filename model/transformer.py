# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/facebookresearch/DiT
# which is released under NonCommercial-4.0 license
# Part of this implementation is adapted from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
# which is released under MIT license
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from einops import rearrange
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from torch import nn, Tensor
from transformers import EsmModel

from . import rotary

import sys
sys.path.append('/scratch/pranamlab/tong/cope/editflows/flow_matching')

import pdb

def lengths_to_offsets(lengths: torch.Tensor) -> torch.Tensor:
    # lengths: (B,) long -> offsets: (B+1,) long
    return F.pad(lengths.cumsum(0), (1, 0))

def build_seq_ids(lengths: torch.Tensor, device=None) -> torch.Tensor:
    # lengths: (B,) -> seq_ids: (T,) where T=sum(lengths)
    device = device or lengths.device
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)

def make_score_mod_for_intra_sequence_only(lengths: torch.Tensor):
    seq_ids = build_seq_ids(lengths, device=lengths.device)  # (T,)

    def score_mod(scores, b, h, q_idx, k_idx):
        # scores: (..., q_block, k_block)
        same = (seq_ids[q_idx] == seq_ids[k_idx])            # bool, broadcastable to scores
        # Set cross-sequence scores to a very negative value *without* in-place ops
        neg_large = torch.finfo(scores.dtype).min            # dtype-safe "−large" (no -inf issues)
        return torch.where(same, scores, torch.full_like(scores, neg_large))

    return score_mod
    
# def make_score_mod_for_intra_sequence_only(lengths: torch.Tensor):
#     """
#     Returns a score_mod callback for FlexAttention that sets logits to -inf
#     when query and key belong to different sequences (ragged, no padding).
#     """
#     seq_ids = build_seq_ids(lengths, device=lengths.device)  # (T,)

#     def score_mod(scores, b, h, q_idx, k_idx):
#         # scores: (..., q_block, k_block); q_idx/k_idx: flat token indices in [0..T-1]
#         original_type = scores.dtype
#         scores = scores.float()
#         same = (seq_ids[q_idx] == seq_ids[k_idx]).to(scores.dtype)
#         scores += (same - 1) * 1e9   # subtract a large number for cross-seq pairs
#         return scores.to(original_type)

#     return score_mod



def bias_dropout_add_scale(
    x: Tensor, scale: Tensor, residual: Optional[Tensor], prob: float, training: bool
) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            y = F.layer_norm(x.float(), [self.dim])
        if y.dim() == 3:   # (B,S,H)
            scale = self.weight[None, None, :]
        elif y.dim() == 2: # (T,H)
            scale = self.weight[None, :]
        else:
            raise ValueError(f"LayerNorm expects 2D/3D, got {y.shape}")
        return y * scale



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout

        self.head_dim = self.dim // self.n_heads

        self.norm1 = LayerNorm(dim=dim)

        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)

        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(
                self,
                x: Tensor,                    # (T, H) flat tokens; T = sum(lengths)
                rotary_caches: tuple,        # (cos, sin)
                c: Tensor                     # (B, cond_dim)
                ) -> Tensor:
        
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2) 

        seq_ids = build_seq_ids(lengths, device=x.device)  # (T,)

        shift_msa_b = shift_msa.squeeze(1)
        scale_msa_b = scale_msa.squeeze(1)
        gate_msa_b  = gate_msa.squeeze(1)
        shift_mlp_b = shift_mlp.squeeze(1)
        scale_mlp_b = scale_mlp.squeeze(1)
        gate_mlp_b  = gate_mlp.squeeze(1)

        # Token-wise (T,H)
        shift_msa_t = shift_msa_b[seq_ids]
        scale_msa_t = scale_msa_b[seq_ids]
        gate_msa_t  = gate_msa_b[seq_ids]
        shift_mlp_t = shift_mlp_b[seq_ids]
        scale_mlp_t = scale_mlp_b[seq_ids]
        gate_mlp_t  = gate_mlp_b[seq_ids]

        x_skip = x
        x = modulate(self.norm1(x), shift=shift_msa_t, scale=scale_msa_t)   # (T, H)

        q = self.qw(x); k = self.kw(x); v = self.vw(x)                  # (T, H)
        T = x.shape[0]
        q = q.view(T, self.n_heads, self.head_dim)
        k = k.view(T, self.n_heads, self.head_dim)
        v = v.view(T, self.n_heads, self.head_dim)

        with torch.amp.autocast("cuda", enabled=False):
            ##################
            # TODO:
            ##################
            q = q.to(dtype); k = k.to(dtype)

        # fold heads into batch
        q = q.transpose(0, 1).contiguous()  # (Hh, T, Dh)
        k = k.transpose(0, 1).contiguous()
        v = v.transpose(0, 1).contiguous()

        q = q.unsqueeze(0)  # -> (1, Hh, T, Dh)
        k = k.unsqueeze(0)  # -> (1, Hh, T, Dh)
        v = v.unsqueeze(0)  # -> (1, Hh, T, Dh)
        
        score_mod = make_score_mod_for_intra_sequence_only(lengths)
        attn_out = flex_attention(q, k, v, score_mod=score_mod)     # (Hh, T, Dh)
        attn_out = attn_out.squeeze(0)  # -> (Hh, T, Dh)
        
        x = attn_out.transpose(0, 1).contiguous().view(T, self.dim)     # (T,H)
        x = bias_dropout_add_scale(self.attn_out(x), gate_msa_t, x_skip, self.dropout, self.training)

        x = bias_dropout_add_scale(
            self.mlp(modulate(self.norm2(x), shift=shift_mlp_t, scale=scale_mlp_t)),
            gate_mlp_t, x, self.dropout, self.training
        )
        return x



class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x

# -------------------------------------------------------------

# NEW: small helper to keep λ ≥ 0
class Positive(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x + self.bias)

# NEW: project token hidden states to "slot" hidden states (n+1)
class SlotProjector(nn.Module):
    """
    Builds n+1 slot states from n token states using learnable boundaries.
    slot i uses left h_{i-1} and right h_i (with learned BOS/EOS).
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.bos = nn.Parameter(torch.zeros(hidden_size))
        self.eos = nn.Parameter(torch.zeros(hidden_size))
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, h_tok: Tensor) -> Tensor:
        # h_tok: (B, n, H)
        B, n, H = h_tok.shape
        bos = self.bos.expand(B, 1, H)               # (B, 1, H)
        eos = self.eos.expand(B, 1, H)               # (B, 1, H)

        # left/right neighbors for the n+1 between-token slots
        left  = torch.cat([bos, h_tok], dim=1)       # (B, n+1, H)
        right = torch.cat([h_tok, eos], dim=1)       # (B, n+1, H)

        slots = torch.cat([left, right], dim=-1)     # (B, n+1, 2H)
        return self.act(self.proj(slots))            # (B, n+1, H)

# NEW: multi-head output layer for edit flows
class EditFlowsHead(nn.Module):
    """
    Produces:
      - λ_ins: (B, n+1), Q_ins: (B, n+1, V)
      - λ_del: (B, n),   λ_sub: (B, n),   Q_sub: (B, n, V)
    """
    def __init__(self, hidden_size: int, vocab_size: int, cond_dim: int):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.cond = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.cond.weight.data.zero_()
        self.cond.bias.data.zero_()

        # token-position heads (n)
        self.lambda_del = nn.Linear(hidden_size, 1, bias=True)
        self.lambda_sub = nn.Linear(hidden_size, 1, bias=True)
        self.q_sub = nn.Linear(hidden_size, vocab_size, bias=True)

        # slot-position heads (n+1)
        self.slot_proj = SlotProjector(hidden_size)
        self.lambda_ins = nn.Linear(hidden_size, 1, bias=True)
        self.q_ins = nn.Linear(hidden_size, vocab_size, bias=True)

        self.to_positive = Positive()

        # init: keep outputs small at start
        for m in [self.lambda_del, self.lambda_sub, self.lambda_ins, self.q_sub, self.q_ins]:
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, h_tok: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # FiLM-style modulation like your DDitFinalLayer
        shift, scale = self.cond(c)[:, None].chunk(2, dim=-1)
        h_tok = modulate(self.norm(h_tok), shift=shift, scale=scale)  # (B, n, H)

        # token positions (n)
        lam_del = self.to_positive(self.lambda_del(h_tok)).squeeze(-1)  # (B, n)
        lam_sub = self.to_positive(self.lambda_sub(h_tok)).squeeze(-1)  # (B, n)
        q_sub   = self.q_sub(h_tok)        # (B, n, V) -- removed softmax here, may need to adjust tensor shape

        # slot positions (n+1)
        h_slot  = self.slot_proj(h_tok)                                 # (B, n+1, H)
        lam_ins = self.to_positive(self.lambda_ins(h_slot)).squeeze(-1) # (B, n+1)
        q_ins   = self.q_ins(h_slot)                                  # (B, n+1, V) -- removed softmax here, may need to adjust tensor shape

        return lam_ins, q_ins, lam_del, lam_sub, q_sub


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, masked: bool, config: DictConfig):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size
        self.pad_id =  getattr(config, "pad_id", 0)

        add_token = 1 if masked else 0   # keep if you need a mask token elsewhere

        # ESM-2 embedding approach (similar to gpm_model.py)
        esm_model_name = getattr(config, "esm_model_name", "facebook/esm2_t33_650M_UR50D")
        freeze_esm = getattr(config, "freeze_esm", True)
        
        if esm_model_name is not None:
            self.tok_embedder = EsmModel.from_pretrained(esm_model_name)
            tok_embed_dim = self.tok_embedder.config.hidden_size
            
            if freeze_esm:
                for param in self.tok_embedder.parameters():
                    param.requires_grad = False
                self.tok_embedder.eval()
            
            # Project from ESM hidden size to model hidden size
            self.tok_embed_to_hidden = nn.Linear(tok_embed_dim, config.hidden_size)
            self.vocab_embed = None  # Not needed when using ESM
        else:
            # Fallback to embedding layer if ESM is not used
            self.tok_embedder = None
            self.tok_embed_to_hidden = None  # Not needed when using standard embedding
            self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)
        
        self.time_embedding = TimestepEmbedder(hidden_size=config.cond_dim)
        self.rotary_emb = rotary.Rotary(dim=config.hidden_size // config.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=config.hidden_size,
                    n_heads=config.n_heads,
                    cond_dim=config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

        # CHANGED: use EditFlowsHead instead of DDitFinalLayer
        self.output_layer = EditFlowsHead(
            hidden_size=config.hidden_size,
            vocab_size=vocab_size + add_token,
            cond_dim=config.cond_dim,
        )
    
    def _embed_ragged(self, x_t, mask):
        # Handle device - use ESM device or vocab_embed device
        if self.tok_embedder is not None:
            device = next(self.tok_embedder.parameters()).device
        else:
            device = self.vocab_embed.weight.device

        if self.tok_embedder is not None:
            x_t_emb = self.tok_embedder(x_t, mask).last_hidden_state
            x = self.tok_embed_to_hidden(x_t_emb) 
        else:
            # Fallback to standard embedding
            x = self.vocab_embed(x_t)                      # (T, H)
        
        # positions from RoPE helper:
        B, S = x_t.shape
        lengths = torch.full((B,), S, device=device, dtype=torch.long)
        positions = self.rotary_emb.positions_like(lengths)  # (T,)
        return x, lengths, positions


    def forward(self, x_t, time: torch.Tensor):
        """
        x_t: List[LongTensor] of variable lengths  OR a (B,S) tensor
        time: (B,) float in [0,1]
        returns lists (ragged): lam_ins, q_ins, lam_del, lam_sub, q_sub
        """
        # 0) pad x_t
        x_t = torch.nn.utils.rnn.pad_sequence(x_t, batch_first=True, padding_value=self.pad_id)
        mask = (x_t != self.pad_id).long()

        # 1) embed ragged
        x_emb = self._embed_ragged(x_t, mask)

        # 2) time conditioning
        B = x_t.shape[0]
        t_emb = F.silu(self.time_embedding(time))  # (B, cond_dim)

        # 3) rotary caches (you can keep your existing cache logic)
        ###############
        # TODO:
        ###############

        # 4) transformer blocks
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for blk in self.blocks:
                x = blk(
                    x=x_emb,
                    mask=mask,
                    rotary_caches=rotary_caches,
                    c=t_emb,
                )

            #############
            # TODO:
            #############
            lam_ins, q_ins, lam_del, lam_sub, q_sub = zip(*outs)

        # 6) return ragged lists (keep padding outside the model if needed)
        return list(lam_ins), list(q_ins), list(lam_del), list(lam_sub), list(q_sub)
