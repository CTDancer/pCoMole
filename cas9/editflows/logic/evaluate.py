# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import math
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------
# Helpers for ragged handling
# ---------------------------

def _batch_iter(seq: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def _pad_ragged_to_dense(
    seqs: Sequence[Tensor],
    pad_id: int,
) -> Tuple[Tensor, Tensor]:
    """
    Pad a list of 1-D LongTensors -> (B, Lmax) LongTensor and an attention mask (B, Lmax) bool.
    pad_id is written into padded positions; attention_mask == 1 for real tokens.
    """
    if len(seqs) == 0:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.bool)

    device = seqs[0].device
    lens = [int(s.numel()) for s in seqs]
    Lmax = max(lens)
    B = len(seqs)

    out = torch.full((B, Lmax), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, Lmax), dtype=torch.bool, device=device)

    for i, s in enumerate(seqs):
        L = s.numel()
        if L > 0:
            out[i, :L] = s
            attn[i, :L] = True

    return out, attn


# ---------------------------
# Perplexity (ragged-friendly)
# ---------------------------

@torch.no_grad()
def compute_perplexity(
    samples: List[Tensor],
    *,
    lm_model,                      # a causal LM compatible with your tokenizer/vocab (e.g., HF AutoModelForCausalLM)
    pad_id: int,
    batch_size: int = 16,
) -> Tensor:
    """
    Compute perplexity on ragged samples using a provided *compatible* causal LM.

    Args:
        samples: list of 1-D LongTensors (ragged, token ids in the *same* vocab the LM expects)
        lm_model: a causal LM (e.g., transformers.AutoModelForCausalLM) on the right device
        pad_id: integer id used for padding during evaluation
        batch_size: eval batch size

    Returns:
        Scalar tensor: mean perplexity across sequences.
    """
    if len(samples) == 0:
        return torch.tensor(float("nan"))

    device = samples[0].device
    lm_model.eval()

    ppl_vals: List[Tensor] = []
    for chunk in _batch_iter(samples, batch_size):
        # pad ragged batch
        x, attn = _pad_ragged_to_dense(chunk, pad_id=pad_id)  # (B, L), (B, L)

        # shift for next-token prediction
        x_in  = x[:, :-1]
        x_tgt = x[:, 1:]
        attn_in = attn[:, :-1]  # (B, L-1)

        # forward LM
        out = lm_model(input_ids=x_in, attention_mask=attn_in, use_cache=False)
        logits = out.logits  # (B, L-1, V)

        # compute token-wise NLL with ignore on pads
        # set ignored targets to -100 per HF convention
        labels = x_tgt.clone()
        labels[~attn[:, 1:]] = -100

        # cross-entropy averaged over *valid* tokens; we want per-sequence mean → then exp
        ce = F.cross_entropy(
            logits.transpose(-1, -2),  # (B, V, L-1)
            labels,
            reduction="none",
            ignore_index=-100,
        )  # (B, L-1)

        # per-sequence mean over valid positions
        valid = (labels != -100).float()
        token_counts = valid.sum(dim=1).clamp_min(1.0)  # avoid div-by-zero
        ce_seq = (ce * valid).sum(dim=1) / token_counts  # (B,)
        ppl_seq = ce_seq.exp()                           # (B,)

        ppl_vals.append(ppl_seq)

    ppl_all = torch.cat(ppl_vals, dim=0)  # (N,)
    return ppl_all.mean()


# ---------------------------
# Entropy (ragged-friendly)
# ---------------------------

def _sample_entropy_1d(sample: Tensor) -> float:
    """
    Entropy (base-2) of a 1-D LongTensor by empirical token histogram.
    """
    if sample.numel() == 0:
        return 0.0
    histogram = Counter(sample.tolist())
    total = float(sum(histogram.values()))
    ent = 0.0
    for c in histogram.values():
        p = c / total
        ent -= p * math.log2(max(p, 1e-12))
    return ent


@torch.no_grad()
def compute_entropy(samples: List[Tensor]) -> Tensor:
    """
    Mean per-sequence token entropy (base-2) for ragged samples.
    """
    if len(samples) == 0:
        return torch.tensor(float("nan"))
    device = samples[0].device
    vals = [_sample_entropy_1d(s) for s in samples]
    return torch.tensor(sum(vals) / len(vals), device=device)


# ---------------------------
# Likelihood / ELBO (EF note)
# ---------------------------

@torch.no_grad()
def estimate_likelihood(
    model: torch.nn.Module,
    dataloader: DataLoader,
    source_distribution,
    path,
    n_discretization: int,
    device: torch.device,
    batch_size: int = 32,
    epsilon: float = 1e-3,
) -> Tensor:
    """
    Placeholder: ELBO-style likelihood estimation used for DFM does not
    directly apply to Edit Flows (different objective / dynamics).
    If you need a quantitative likelihood-like diagnostic for EF,
    consider CTMC pathwise estimators or reverse-time simulators instead.

    We raise NotImplementedError to avoid silently reporting a mismatched metric.
    """
    raise NotImplementedError(
        "ELBO/likelihood estimation used for discrete flow matching is not applicable to Edit Flows. "
        "Use task metrics or CTMC-based diagnostics instead."
    )
