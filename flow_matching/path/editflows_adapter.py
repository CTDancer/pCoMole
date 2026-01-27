from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import torch
from torch import Tensor
import pdb

from flow_matching.path import MixtureDiscreteProbPath


@dataclass
class EditFlowsPathSampleRagged:
    # aligned inputs/outputs (ragged)
    z0_list: List[Tensor]      # list[(T_i,)]
    z1_list: List[Tensor]      # list[(T_i,)]
    zt_list: List[Tensor]      # list[(T_i,)]
    # projected current sequences (ragged X_t)
    x_t_list: List[Tensor]     # list[(n_i,)]
    # targets/masks for EF loss (ragged)
    need_delete_list: List[Tensor]        # list[(n_i,)]   bool
    need_substitute_list: List[Tensor]    # list[(n_i,)]   bool
    sub_target_list: List[Tensor]         # list[(n_i,)]   long
    ins_slot_idx_list: List[Tensor]       # list[(K_i,)]   long  indices in [0..n_i]
    ins_target_list: List[Tensor]         # list[(K_i,)]   long
    # time
    t: Tensor                              # (B,)


class EditFlowsPathAdapter:
    """
    Ragged adapter for Edit Flows:
      - Accepts aligned Z sequences as tensors (B, N) (may include eps_id).
      - Applies MixtureDiscreteProbPath's scheduler (σ_t) to sample z_t by
        flipping z0→z1 with prob (1-σ_t) per aligned token.
      - You can later project z_t -> x_t by removing ε.
    """
    def __init__(self, mixture_path, eps_id: int):
        self.path = mixture_path
        self.eps_id = int(eps_id)

    # expose scheduler so training can precompute weights κ̇/(1-κ)
    def scheduler(self, t: Tensor):
        return self.path.scheduler(t)

    @torch.no_grad()
    def sample(
        self,
        z0: Tensor,   # (B, N)
        z1: Tensor,   # (B, N)
        t: Tensor,    # () or (B,)
    ) -> Tensor:
        """
        Sample z_t from (z0, z1) using the discrete mixture path:
          - σ_t = scheduler(t)
          - with prob σ_t keep z0
          - with prob (1 - σ_t) take z1
        We do this per-column.
        BOS/EOS are already aligned -> z0 == z1 there -> safe.
        """
        device = z0.device
        B, N = z0.shape

        # get sigma_t from the path
        sigma = self.path.scheduler(t).sigma_t  # could be scalar or (B,)
        if sigma.dim() == 0:
            sigma = sigma.expand(B)          # (B,)
        else:
            # ensure shape is (B,)
            sigma = sigma.view(B)
        # reshape to broadcast over columns
        sigma = sigma.view(B, 1)             # (B, 1)

        # uniform noise per position
        u = torch.rand(B, N, device=device)

        # we only need to flip where z0 != z1
        diff_mask = (z0 != z1)               # (B, N)

        # flip where u > sigma (i.e. prob 1 - sigma) AND tokens differ
        flip = (u > sigma) & diff_mask       # (B, N)

        # choose from z1 when flip, else z0
        z_t = torch.where(flip, z1, z0)      # (B, N)

        return z_t