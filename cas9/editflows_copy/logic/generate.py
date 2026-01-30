# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn, Tensor

from transformers.tokenization_utils import PreTrainedTokenizer

from flow_matching.path import ProbPath  # adapter exposing .scheduler(t)
from flow_matching.solver import EditFlowsEulerSolver
from .flow import SourceDistribution


class WrappedEFModel:
    """
    Thin wrapper so the solver has a stable interface:
      forward(x_list, t_scalar or t_vec) -> lam_ins, logits_ins, lam_del, lam_sub, logits_sub
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @torch.no_grad()
    def __call__(self, x_list: Sequence[Tensor], t: Tensor):
        # Model already returns ragged lists (lam_ins, logits_ins, lam_del, lam_sub, logits_sub)
        return self.model(x_t=list(x_list), time=t)


def _rows_to_ragged(x_dense: Tensor) -> List[Tensor]:
    # Start from a (B, L) dense init -> ragged list of (L,)
    B, L = x_dense.shape
    return [x_dense[b].clone() for b in range(B)]


def _ragged_to_text(
    seqs: Sequence[Tensor],
    tokenizer: PreTrainedTokenizer,
) -> List[str]:
    # Decode each sequence independently (no padding required)
    out = []
    for s in seqs:
        ids = s.detach().tolist()
        out.append(tokenizer.decode(ids, skip_special_tokens=False))
    return out


def generate_samples(
    model: nn.Module,
    step: int,
    vocab_size: int,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    path: ProbPath,                      # adapter or object exposing .scheduler(t)
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    time_epsilon: float = 0.0,           # not used in Option A; kept for API parity
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
) -> List[Tensor]:
    """
    EditFlows ragged generation with Euler thinning (≤1 jump per step).
    Returns list[LongTensor] of final sequences.
    """
    wrapped = WrappedEFModel(model=model)

    # Initial sequences (dense) -> ragged
    x_init_dense = source_distribution.sample(
        tensor_size=(sample_batch_size, sequence_length), device=device
    ).long()
    x_list = _rows_to_ragged(x_init_dense)

    # New ragged solver
    solver = EditFlowsEulerSolver(
        model=wrapped,
        scheduler=path.scheduler,
        vocab_size=vocab_size,
        dtype_categorical=dtype_categorical,
    )

    # Simulate from t=0 → t=1
    x_final = solver.sample(
        x_list=x_list,
        n_steps=sampling_steps,
        t0=0.0,
        t1=1.0,
        verbose=False,
    )

    # Decode to text (optional, for logging/inspection)
    sentences = _ragged_to_text(x_final, tokenizer)
    if sample_dir is not None:
        file_name = sample_dir / f"iter_{step}" / f"sample_{rank}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)
        with open(file_name, "w") as f:
            for s in sentences:
                f.write(s + "\n" + "=" * 20 + " New sample " + "=" * 20 + "\n")

    return x_final
