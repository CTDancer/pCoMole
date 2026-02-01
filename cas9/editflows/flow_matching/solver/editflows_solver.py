# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

from typing import Callable, List, Sequence, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class EditFlowsEulerSolver:
    """
    Ragged EditFlows CTMC solver with Euler thinning (≤1 jump per step).

    At each step h:
      - query model for (λ_ins[slots], Q_ins[slots], λ_del[tokens], λ_sub[tokens], Q_sub[tokens])
      - total intensity Λ = sum(λ_ins) + sum(λ_del) + sum(λ_sub)
      - with prob 1 - exp(-h Λ), take ONE jump:
           * sample which event from concatenated rates
           * if ins/sub: sample token from the Q at that position/slot
           * apply edit to the ragged sequence (insert/delete/replace)
      - advance t ← t + h

    This is faithful to the CTMC in the small-h regime used by Euler schemes.
    """

    def __init__(
        self,
        model,                                        # callable: (x_list, t_vec) -> ragged heads
        scheduler: Callable[[Tensor], object],        # path.scheduler(t) providing alpha_t, d_alpha_t, sigma_t if needed
        vocab_size: int,
        dtype_categorical: torch.dtype = torch.float64,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.vocab_size = vocab_size
        self.dtype_cat = dtype_categorical
        self.gen = generator

    @torch.no_grad()
    def _sample_categorical_logits(self, logits: Tensor) -> Tensor:
        # logits: (V,) -> sample id
        probs = F.softmax(logits, dim=-1, dtype=self.dtype_cat)
        return torch.multinomial(probs, num_samples=1, replacement=True, generator=self.gen).squeeze(0)

    @torch.no_grad()
    def _maybe_one_jump(
        self,
        x: Tensor,                     # (n,)
        lam_ins: Tensor,               # (n+1,)
        logits_ins: Tensor,            # (n+1, V)
        lam_del: Tensor,               # (n,)
        lam_sub: Tensor,               # (n,)
        logits_sub: Tensor,            # (n, V)
        h: float,
        device: torch.device,
    ) -> Tensor:
        """
        Apply at most one edit event to x with step size h.
        Returns possibly modified x (1-D LongTensor).
        """
        n = x.numel()
        # Total intensity
        sum_ins = lam_ins.sum() if lam_ins.numel() else torch.tensor(0.0, device=device)
        sum_del = lam_del.sum() if lam_del.numel() else torch.tensor(0.0, device=device)
        sum_sub = lam_sub.sum() if lam_sub.numel() else torch.tensor(0.0, device=device)
        total = (sum_ins + sum_del + sum_sub).clamp(min=0.0)

        if total.item() <= 0:
            return x  # no possible jumps

        # Bernoulli: does a jump occur?
        p_jump = 1.0 - torch.exp(torch.tensor(-h, device=device) * total)
        if torch.rand((), device=device, generator=self.gen) >= p_jump:
            return x  # no jump this step

        # Build event vector: [ins(0..n), del(0..n-1), sub(0..n-1)]
        parts = []
        if lam_ins.numel(): parts.append(lam_ins)
        if lam_del.numel(): parts.append(lam_del)
        if lam_sub.numel(): parts.append(lam_sub)
        lam_cat = torch.cat(parts, dim=0)  # (n+1 + n + n,)
        probs = (lam_cat / total).to(self.dtype_cat)

        # Sample which event
        e_idx = torch.multinomial(probs, num_samples=1, generator=self.gen).item()

        # Decode which branch
        offset = 0
        if e_idx < lam_ins.numel():
            # INSERT at slot s
            s = e_idx
            # sample token from slot distribution
            y = self._sample_categorical_logits(logits_ins[s])  # id ∈ [0..V-1]
            # perform insertion at slot s: x' = [x[:s], y, x[s:]]
            if s == 0:
                x_new = torch.cat([y.view(1), x], dim=0)
            elif s == n:
                x_new = torch.cat([x, y.view(1)], dim=0)
            else:
                x_new = torch.cat([x[:s], y.view(1), x[s:]], dim=0)
            return x_new

        e_idx -= lam_ins.numel()
        if e_idx < lam_del.numel():
            # DELETE token j
            j = e_idx
            if n == 0:
                return x
            if j == 0:
                return x[1:]
            elif j == n - 1:
                return x[:-1]
            else:
                return torch.cat([x[:j], x[j + 1 :]], dim=0)

        e_idx -= lam_del.numel()
        # SUBSTITUTE at token j
        j = e_idx
        if n == 0:
            return x
        y = self._sample_categorical_logits(logits_sub[j])
        x_new = x.clone()
        x_new[j] = y
        return x_new

    @torch.no_grad()
    def sample(
        self,
        x_list: Sequence[Tensor],   # ragged: list[(n_i,)]
        n_steps: int,
        t0: float = 0.0,
        t1: float = 1.0,
        verbose: bool = False,
    ) -> List[Tensor]:
        """
        Evolve all sequences independently with shared time grid t_k.
        """
        assert n_steps >= 1
        B = len(x_list)
        device = x_list[0].device if B > 0 else torch.device("cpu")

        x = [xi.clone() for xi in x_list]
        t = torch.full((B,), float(t0), device=device)
        h = (t1 - t0) / float(n_steps)

        for k in range(n_steps):
            # Query model once at time t_k (vector of size B)
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub = self.model(x, t)

            # Apply at most one jump per sample
            for i in range(B):
                x[i] = self._maybe_one_jump(
                    x=x[i],
                    lam_ins=lam_ins[i].squeeze(0) if lam_ins[i].dim() == 2 else lam_ins[i],
                    logits_ins=logits_ins[i].squeeze(0) if logits_ins[i].dim() == 3 else logits_ins[i],
                    lam_del=lam_del[i].squeeze(0) if lam_del[i].dim() == 2 else lam_del[i],
                    lam_sub=lam_sub[i].squeeze(0) if lam_sub[i].dim() == 2 else lam_sub[i],
                    logits_sub=logits_sub[i].squeeze(0) if logits_sub[i].dim() == 3 else logits_sub[i],
                    h=h,
                    device=device,
                )

            # advance time
            t = t + h

        return x
