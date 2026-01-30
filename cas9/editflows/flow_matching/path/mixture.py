# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torch import Tensor

from flow_matching.path.path import ProbPath

from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like, unsqueeze_to_match


class MixtureDiscreteProbPath(ProbPath):
    r"""The ``MixtureDiscreteProbPath`` class defines a factorized discrete probability path.

    This path remains constant at the source data point :math:`X_0` until a random time, determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\sigma_t`, which is a function of time `t`. Specifically, :math:`\sigma_t` represents the probability of remaining at :math:`X_0`, while :math:`1 - \sigma_t` is the probability of flipping to :math:`X_1`:

    .. math::

        P(X_t = X_0) = \sigma_t \quad \text{and} \quad  P(X_t = X_1) = 1 - \sigma_t,

    where :math:`\sigma_t` is provided by the scheduler.

    Example:

    .. code-block:: python

        >>> x_0 = torch.zeros((1, 3, 3))
        >>> x_1 = torch.ones((1, 3, 3))

        >>> path = MixtureDiscreteProbPath(PolynomialConvexScheduler(n=1.0))
        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.1])).x_t
        >>> result
        tensor([[[0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, 0.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([0.5])).x_t
        >>> result
        tensor([[[1.0, 0.0, 1.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 1.0, 0.0]]])

        >>> result = path.sample(x_0, x_1, t=torch.tensor([1.0])).x_t
        >>> result
        tensor([[[1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0],
                 [1.0, 1.0, 1.0]]])

    Args:
        scheduler (ConvexScheduler): The scheduler that provides :math:`\sigma_t`.
    """

    def __init__(self, scheduler: ConvexScheduler):
        assert isinstance(
            scheduler, ConvexScheduler
        ), "Scheduler for ConvexProbPath must be a ConvexScheduler."

        self.scheduler = scheduler

    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        r"""Sample from the affine probability path:
            | given :math:`(X_0,X_1) \sim \pi(X_0,X_1)` and a scheduler :math:`(\alpha_t,\sigma_t)`.
            | return :math:`X_0, X_1, t`, and :math:`X_t \sim p_t`.
        Args:
            x_0 (Tensor): source data point, shape (batch_size, ...).
            x_1 (Tensor): target data point, shape (batch_size, ...).
            t (Tensor): times in [0,1], shape (batch_size).

        Returns:
            DiscretePathSample: a conditional sample at :math:`X_t ~ p_t`.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t

        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        source_indices = torch.rand(size=x_1.shape, device=x_1.device) < sigma_t
        x_t = torch.where(condition=source_indices, input=x_0, other=x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)
    
    @torch.no_grad()
    def sample_localized(
        self,
        z0: torch.Tensor,              # (B, N) aligned (ε allowed)
        z1: torch.Tensor,              # (B, N) aligned (ε allowed)
        t: torch.Tensor,               # () or (B,)
        lambda_prop: float | torch.Tensor,  # λ_prop (scalar or per-batch)
        return_M: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Localized sampling per Appendix C.1 using the ONE-STEP inverse-CDF sampler:

          1) U ~ Uniform(0,1), t*_all = kappa^{-1}(U)
             seeds = (U <= kappa(t)), t* = seeds ? t*_all : t
          2) Propagate left/right with Poisson(λ_prop * (t - t*))
          3) Collapse M_t (row-wise CTMCs) -> m_t (columnwise OR)
          4) z_t = where(m_t & (z0!=z1), z1, z0)

        Returns:
          z_t, (optional) M_t (B,N,N) bool, (optional) m_t (B,N) bool
        """
        device = z0.device
        B, N = z0.shape

        # Normalize shapes
        if t.dim() == 0:
            t = t.expand(B)                         # (B,)
        kappa_t = self.scheduler.kappa(t)           # (B,)
        if not torch.is_tensor(lambda_prop):
            lambda_prop = torch.tensor(lambda_prop, dtype=kappa_t.dtype, device=device)
        if lambda_prop.dim() == 0:
            lambda_prop = lambda_prop.expand(B)     # (B,)

        # ========= ONE-STEP SAMPLER (CHANGED BLOCK) =========
        # U ~ Uniform(0,1): unconditional draw for the first-activation time via inverse-CDF
        U = torch.rand(B, N, device=device)                         # (B, N) in [0,1]
        t_star_all = self.scheduler.kappa_inverse(U)                # (B, N) unconditional T*

        # Seed indicator: "activated by time t?"  <=>  T* <= t  <=>  U <= κ(t)
        seeds = (U <= kappa_t.view(B, 1))                           # (B, N) bool

        # Truncated activation time: if not seeded, set t* = t so Δt=0
        t_star = torch.where(seeds, t_star_all, t.view(B, 1))       # (B, N)
        # =====================================================

        # Propagation window and Poisson spreads
        delta = (t.view(B, 1) - t_star).clamp_min(0.0)              # (B, N)
        rate = (lambda_prop.view(B, 1) * delta).to(torch.float32)   # (B, N)
        left_ext  = torch.poisson(rate)                              # (B, N)
        right_ext = torch.poisson(rate)                              # (B, N)

        # Zero extents where there is no seed
        left_ext  = (left_ext  * seeds.float()).to(torch.int64)
        right_ext = (right_ext * seeds.float()).to(torch.int64)

        # Paint M_t row-by-row as contiguous intervals around each diagonal
        M_t = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        for b in range(B):
            for i in range(N):
                if not seeds[b, i]:
                    continue
                L = max(0, i - int(left_ext[b, i].item()))
                R = min(N - 1, i + int(right_ext[b, i].item()))
                M_t[b, i, L : R + 1] = True

        # Collapse to m_t (columnwise OR across rows)
        m_t = M_t.any(dim=1)                                       # (B, N) bool

        # Build z_t (apply mask only where z0 != z1)
        diff = (z0 != z1)
        z_t = torch.where(m_t & diff, z1, z0)                      # (B, N)

        if return_M:
            return z_t, M_t, m_t
        return z_t, None, None

    def posterior_to_velocity(
        self, posterior_logits: Tensor, x_t: Tensor, t: Tensor
    ) -> Tensor:
        r"""Convert the factorized posterior to velocity.

        | given :math:`p(X_1|X_t)`. In the factorized case: :math:`\prod_i p(X_1^i | X_t)`.
        | return :math:`u_t`.

        Args:
            posterior_logits (Tensor): logits of the x_1 posterior conditional on x_t, shape (..., vocab size).
            x_t (Tensor): path sample at time t, shape (...).
            t (Tensor): time in [0,1].

        Returns:
            Tensor: velocity.
        """
        posterior = torch.softmax(posterior_logits, dim=-1)
        vocabulary_size = posterior.shape[-1]
        x_t = F.one_hot(x_t, num_classes=vocabulary_size)
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)

        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)
