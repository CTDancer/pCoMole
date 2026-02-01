# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py#L20
# which is released under BSD-3 license
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

from typing import Tuple, Optional

import torch
from einops import repeat
from torch import Tensor


class Rotary(torch.nn.Module):
    """
    From: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
    """

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    
    def positions_like(self, lengths: torch.Tensor) -> torch.Tensor:
        # returns (T,) positions 0..len_i-1 concatenated across sequences
        return torch.cat([torch.arange(int(L), device=lengths.device) for L in lengths], dim=0)

    def forward(self, x: Tensor, seq_dim: int = 1) -> Tuple[Tensor, Tensor]:
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)

            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)

        return self.cos_cached, self.sin_cached


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_torch(x, cos, sin, interleaved: bool = False, rotary_dim: Optional[int] = None):
    # cos/sin cached as in your Rotary.forward
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2].to(dtype=x.dtype, device=x.device)  # (S, d/2)
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2].to(dtype=x.dtype, device=x.device)  # (S, d/2)

    # Decide how many head dims get RoPE
    if rotary_dim is None:
        rotary_dim = cos.shape[-1] * 2  # default: full cached rotary span
    rotary_dim = min(rotary_dim, x.shape[-1])
    assert rotary_dim % 2 == 0, "rotary_dim must be even"

    # Broadcast to (..., S, 1, rotary_dim)
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")

    x_rot = x[..., :rotary_dim]
    x_tail = x[..., rotary_dim:]

    x_rot = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_rot, x_tail], dim=-1)

def apply_rotary_emb_ragged(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    positions: torch.Tensor, head_dim: Optional[int] = None,
    interleaved: bool = False, inplace: bool = True
):
    # q,k: (T, H, Dh), positions: (T,)
    cos_half = cos[0, :, 0, 0, : cos.shape[-1] // 2].to(dtype=q.dtype, device=q.device)  # (S, d/2)
    sin_half = sin[0, :, 0, 0, : sin.shape[-1] // 2].to(dtype=q.dtype, device=q.device)
    assert positions.max().item() < cos_half.size(0), "position index exceeds cached length"

    cos_sel = cos_half.index_select(0, positions)  # (T, d/2)
    sin_sel = sin_half.index_select(0, positions)  # (T, d/2)

    cos_sel = repeat(cos_sel, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")  # (T,1,d)
    sin_sel = repeat(sin_sel, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")

    ro_dim = head_dim or cos_sel.shape[-1]
    ro_dim = min(ro_dim, q.shape[-1])
    assert ro_dim % 2 == 0, "rotary_dim/head_dim must be even"

    if not inplace:
        q = q.clone()
        k = k.clone()

    def rot(x):
        x1, x2 = x[..., : ro_dim // 2], x[..., ro_dim // 2: ro_dim]
        return torch.cat([-x2, x1], dim=-1)

    q_head = q[..., :ro_dim]
    k_head = k[..., :ro_dim]
    q[..., :ro_dim] = q_head * cos_sel + rot(q_head) * sin_sel
    k[..., :ro_dim] = k_head * cos_sel + rot(k_head) * sin_sel
    return q, k

