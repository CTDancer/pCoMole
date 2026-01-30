# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from abc import ABC
from typing import List, Optional, Tuple, Union

import torch
from flow_matching.loss import MixturePathGeneralizedKL, EditFlowsLoss
from flow_matching.path import MixtureDiscreteProbPath  # for the scheduler only
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.path.editflows_adapter import EditFlowsPathAdapter  # <-- NEW import
from torch import Tensor
from torch.nn.modules.loss import _Loss


class SourceDistribution(ABC):
    def __init__(self) -> None:
        ...

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        ...

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        ...


class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()


import torch
from typing import List, Tuple, Optional, Union
Tensor = torch.Tensor

class UniformSourceDistribution:
    def __init__(self, vocab_size, special_token_ids = None):
        self.vocab_size = vocab_size
        self.special_token_ids = set(special_token_ids) if special_token_ids is not None else set()
        # Compute allowed tokens by removing all special tokens from vocab
        self._allowed_tokens = [i for i in range(vocab_size) if i not in self.special_token_ids]
        if len(self._allowed_tokens) == 0:
            raise ValueError(f"All tokens are special tokens: {special_token_ids}")

    @property
    def masked(self) -> bool:
        return False

    @torch.no_grad()
    def _sample_from_allowed(self, shape, device, allowed_tokens = None, generator = None):
        """Sample uniformly from allowed tokens with given shape."""
        if allowed_tokens is None:
            allowed_tokens = self._allowed_tokens
        if len(allowed_tokens) == 0:
            raise ValueError("No allowed tokens provided")

        allowed_tensor = torch.tensor(allowed_tokens, dtype=torch.long, device=device)
        num_allowed = len(allowed_tokens)

        if shape.numel() == 0:
            # Return an empty tensor (length-0 sequence)
            return torch.empty(shape, dtype=torch.long, device=device)

        # Sample indices in [0, num_allowed)
        indices = torch.randint(
            low=0, high=num_allowed, size=shape, device=device, generator=generator
        )
        return allowed_tensor[indices]

    @torch.no_grad()
    def sample(self, tensor_size, device, allowed_tokens = None, generator = None):
        return self._sample_from_allowed(tensor_size, device, allowed_tokens, generator)

    @torch.no_grad()
    def sample_like(self, tensor_like, allowed_tokens = None, generator = None):
        """
        Keep original semantics: sample tokens with the SAME SHAPE(S) as tensor_like.
        """
        if isinstance(tensor_like, (list, tuple)):
            return [
                self._sample_from_allowed(seq.shape, seq.device, allowed_tokens, generator)
                for seq in tensor_like
            ]
        return self._sample_from_allowed(tensor_like.shape, tensor_like.device, allowed_tokens, generator)

    @torch.no_grad()
    def sample_like(
        self,
        tensor_like,
        allowed_tokens: Optional[List[int]] = None,
        min_len_factor: float = 0.0,
        max_len_factor: float = 2.0,
        generator: Optional[torch.Generator] = None,
    ):
        """
        For each reference x1, sample x0 with LENGTH L ~ Uniform{ floor(min_len_factor*N) .. floor(max_len_factor*N) },
        defaulting to [0, 2*N]. Returns an empty tensor if L == 0.

        Supports Tensor (1D) or List[Tensor] (ragged). If you pass a 2D tensor, we
        interpret N as the last dimension and return a 1D sequence for that tensor.
        """
        def _one(x1):
            assert x1.dim() >= 1, "x1 must be at least 1D"
            device = x1.device
            dtype = torch.long

            N = int(x1.shape[-1])  # use last-dim length as reference
            lo = int(max(0, int(min_len_factor * N)))
            hi = int(max(0, int(max_len_factor * N)))
            # randint is [low, high), so make high inclusive with +1
            L = int(torch.randint(low=lo, high=hi + 1, size=(1,), device=device, generator=generator).item())
            if L == 0:
                return torch.empty((0,), dtype=dtype, device=device)
            return self._sample_from_allowed(torch.Size([L]), device, allowed_tokens, generator)

        if isinstance(tensor_like, (list, tuple)):
            return [_one(seq) for seq in tensor_like]
        else:
            return _one(tensor_like)
    
    def sample_x0_from_x1(self, x1, pad_id, allowed_tokens, scale_size = 2, bos_id = 0, eos_id = 2, sample_type='regular', eps_id = -1):
        """
        For each sequence in x1, sample an x0 whose *core* length (excluding BOS/EOS)
        is in [max(0, (1-scale_size)*len_valid(x1)), (1+scale_size)*len_valid(x1)], 
        where len_valid(x1) counts only tokens that are NOT {BOS, EOS, PAD}.

        Rules:
        - x0 always starts with BOS and ends with EOS
        - x0 core tokens are sampled uniformly from vocab excluding {BOS, EOS, PAD}
        - We batch and pad x0 to a common length (B, L0) with pad_id
        - "length of x0 does not account for BOS and EOS" = the sampled core length

        Returns:
            x0: (B, L0) Long, padded with pad_id
        """
        device = x1.device
        B, L1 = x1.shape

        # compute valid length of x1 per sequence
        # valid = not pad, not BOS, not EOS
        valid_mask_x1 = (x1 != pad_id) & (x1 != bos_id) & (x1 != eos_id)
        valid_len = valid_mask_x1.sum(dim=1)  # (B,)

        # we will store all sequences here before padding
        x0_seqs = []

        for b in range(B):
            max_core_len = int((1 + scale_size) * valid_len[b].item())  # may be 0
            # sample core length in [max(0, (1-scale_size)*len_valid(x1)), (1+scale_size)*len_valid(x1)]
            min_core_len = max(0, int((1 - scale_size) * valid_len[b].item()))
            # print(f"x1 length: {valid_len[b].item()}, min_core_len: {min_core_len}, max_core_len: {max_core_len}")
            core_len = int(torch.randint(low=min_core_len, high=max_core_len + 1, size=(1,), device=device).item())

            # 4) sample core tokens
            if core_len > 0:
                idx = torch.randint(0, allowed_tokens.size(0), (core_len,), device=device)
                core_tokens = allowed_tokens[idx]  # (core_len,)
            else:
                core_tokens = torch.empty(0, dtype=torch.long, device=device)

            # 5) build full x0: [BOS] + core + [EOS]
            seq = torch.cat([
                torch.tensor([bos_id], device=device, dtype=torch.long),
                core_tokens,
                torch.tensor([eos_id], device=device, dtype=torch.long),
            ], dim=0)  # (1 + core_len + 1,)

            x0_seqs.append(seq)

        x0 = torch.nn.utils.rnn.pad_sequence(x0_seqs, batch_first=True, padding_value=pad_id)

        return x0


# class UniformSourceDistribution(SourceDistribution):
#     def __init__(self, vocab_size: int, special_token_ids: Optional[List[int]] = None) -> None:
#         self.vocab_size = vocab_size
#         self.special_token_ids = set(special_token_ids) if special_token_ids is not None else set()
#         # Compute allowed tokens by removing all special tokens from vocab
#         self._allowed_tokens = [i for i in range(vocab_size) if i not in self.special_token_ids]
        
#         if len(self._allowed_tokens) == 0:
#             raise ValueError(f"All tokens are special tokens: {special_token_ids}")

#     @property
#     def masked(self) -> bool:
#         return False

#     def _sample_from_allowed(
#         self, shape: torch.Size, device: torch.device, allowed_tokens: Optional[List[int]] = None
#     ) -> Tensor:
#         """Sample uniformly from allowed tokens."""
#         if allowed_tokens is None:
#             # Use default allowed tokens (vocab minus special tokens)
#             allowed_tokens = self._allowed_tokens
        
#         if len(allowed_tokens) == 0:
#             raise ValueError(f"No allowed tokens provided")
        
#         # print(f"allowed_tokens: {allowed_tokens}")

#         # Sample indices into allowed_tokens, then map to actual token IDs
#         allowed_tensor = torch.tensor(allowed_tokens, dtype=torch.long, device=device)
#         num_allowed = len(allowed_tokens)
        
#         # Sample indices in [0, num_allowed)
#         indices = torch.randint(size=shape, high=num_allowed, device=device)
#         return allowed_tensor[indices]

#     def sample(
#         self, 
#         tensor_size: Tuple[int, ...], 
#         device: torch.device, 
#         allowed_tokens: Optional[List[int]] = None
#     ) -> Tensor:
#         return self._sample_from_allowed(tensor_size, device, allowed_tokens)

#     def sample_like(
#         self, 
#         tensor_like: Union[Tensor, List[Tensor]], 
#         allowed_tokens: Optional[List[int]] = None
#     ) -> Union[Tensor, List[Tensor]]:
#         """
#         Sample uniform tokens matching the shape of tensor_like.
#         Supports both Tensor and List[Tensor] for ragged inputs.
        
#         Args:
#             tensor_like: Either a Tensor or List[Tensor] (for ragged inputs)
#             allowed_tokens: Optional list of allowed token IDs. If None, uses vocab minus special tokens.
            
#         Returns:
#             Tensor or List[Tensor] matching the input shape(s)
#         """
#         # Handle ragged input (list of tensors)
#         if isinstance(tensor_like, (list, tuple)):
#             return [
#                 self._sample_from_allowed(seq.shape, seq.device, allowed_tokens)
#                 for seq in tensor_like
#             ]
        
#         # Handle regular tensor input
#         return self._sample_from_allowed(tensor_like.shape, tensor_like.device, allowed_tokens)


class EmpiricalSourceDistribution(SourceDistribution):
    def __init__(self, vocab_size: int, probs: torch.Tensor, length: int):
        self.vocab_size = vocab_size
        self.registered = probs / probs.sum()
        self.length = length  # e.g., 100 tokens

    @property
    def masked(self) -> bool:
        return False

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        B = tensor_size[0]
        idx = torch.multinomial(self.registered.to(device), num_samples=self.length, replacement=True)
        return idx.view(1, self.length).repeat(B, 1)

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        B = tensor_like.shape[0]
        idx = torch.multinomial(self.registered.to(tensor_like.device), num_samples=self.length, replacement=True)
        return idx.view(1, self.length).repeat(B, 1)


# NOTE: return type changed to the adapter (ragged). We only rely on .scheduler(t) and .sample(...).
def get_path(scheduler_type: str, exponent: Optional[float] = None, eps_id: int = -1) -> EditFlowsPathAdapter:
    if scheduler_type == "polynomial":
        # paper uses cubic => exponent=3
        scheduler = PolynomialConvexScheduler(n=exponent)
    else:
        raise ValueError(f"{scheduler_type} is not supported")

    # MixtureDiscreteProbPath carries the scheduler; the adapter will sample ragged z_t itself
    mixture = MixtureDiscreteProbPath(scheduler=scheduler)
    return EditFlowsPathAdapter(mixture_path=mixture, eps_id=eps_id)


def get_source_distribution(
    source_distribution: str,
    p_emp: Optional[Tensor] = None,
    length: Optional[int] = None,
    vocab_size: Optional[int] = None,
    special_token_ids: Optional[List[int]] = None,
) -> SourceDistribution:
    if p_emp is not None:
        assert vocab_size is not None and length is not None, "Empirical source requires vocab_size and length"
        return EmpiricalSourceDistribution(vocab_size=vocab_size, probs=p_emp, length=length)

    if source_distribution == "mask":
        assert vocab_size is not None, "Masked source requires vocab_size"
        return MaskedSourceDistribution(mask_token=vocab_size)
    elif source_distribution == "uniform":
        assert vocab_size is not None, "Uniform source requires vocab_size"
        return UniformSourceDistribution(vocab_size=vocab_size, special_token_ids=special_token_ids)
    else:
        raise ValueError(f"{source_distribution} is not supported")


def get_loss_function(loss_function: str, path: Optional[Union[MixtureDiscreteProbPath, EditFlowsPathAdapter]] = None) -> _Loss:
    if loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_function == "generalized_kl":
        assert path is not None
        # Generalized KL still expects a (dense) path; fine for DFM experiments
        return MixturePathGeneralizedKL(path=path)
    elif loss_function == "editflows":
        # Ragged EF loss does NOT need the path; training.step precomputes the weight
        return EditFlowsLoss(reduction="mean")
    else:
        raise ValueError(f"{loss_function} is not supported")


# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the CC-by-NC license found in the
# # LICENSE file in the root directory of this source tree.

# from abc import ABC
# from typing import Optional, Tuple

# import torch
# from flow_matching.loss import MixturePathGeneralizedKL, EditFlowsLoss
# from flow_matching.path import MixtureDiscreteProbPath, ProbPath, EditFlowsPathAdapter
# from flow_matching.path.scheduler import PolynomialConvexScheduler
# from torch import Tensor
# from torch.nn.modules.loss import _Loss


# class SourceDistribution(ABC):
#     def __init__(
#         self,
#     ) -> None:
#         ...

#     def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
#         ...

#     def sample_like(self, tensor_like: Tensor) -> Tensor:
#         ...


# class MaskedSourceDistribution(SourceDistribution):
#     def __init__(self, mask_token: int) -> None:
#         self.mask_token = mask_token

#     @property
#     def masked(self) -> bool:
#         return True

#     def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
#         return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

#     def sample_like(self, tensor_like: Tensor) -> Tensor:
#         return torch.zeros_like(tensor_like).fill_(self.mask_token).long()


# class UniformSourceDistribution(SourceDistribution):
#     def __init__(self, vocab_size: int) -> None:
#         self.vocab_size = vocab_size

#     @property
#     def masked(self) -> bool:
#         return False

#     def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
#         return torch.randint(size=tensor_size, high=self.vocab_size, device=device)

#     def sample_like(self, tensor_like: Tensor) -> Tensor:
#         return torch.randint_like(tensor_like, high=self.vocab_size)
    
# class EmpiricalSourceDistribution(SourceDistribution):
#     def __init__(self, vocab_size: int, probs: torch.Tensor, length: int):
#         self.vocab_size = vocab_size
#         self.registered = probs / probs.sum()
#         self.length = length  # e.g., 100 tokens as in the paper’s variant

#     @property
#     def masked(self) -> bool: return False

#     def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
#         B = tensor_size[0]
#         idx = torch.multinomial(self.registered.to(device), num_samples=self.length, replacement=True)
#         return idx.view(1, self.length).repeat(B, 1)

#     def sample_like(self, tensor_like: Tensor) -> Tensor:
#         B = tensor_like.shape[0]
#         idx = torch.multinomial(self.registered.to(tensor_like.device), num_samples=self.length, replacement=True)
#         return idx.view(1, self.length).repeat(B, 1)


# def get_path(scheduler_type: str, exponent: Optional[float] = None) -> ProbPath:
#     if scheduler_type == "polynomial":
#         scheduler = PolynomialConvexScheduler(n=exponent)
#     else:
#         raise ValueError(f"{scheduler_type} is not supported")

#     return EditFlowsPathAdapter(path=MixtureDiscreteProbPath(scheduler=scheduler)) # still need to decide (1) whether to implement the z_0, z_1 creation here, and (2) how to pass the eps_id to the adapter


# def get_source_distribution(
#     source_distribution: str, p_emp: Optional[Tensor] = None, length: Optional[int] = None, vocab_size: int = None
# ) -> SourceDistribution:
#     if p_emp is not None:
#         return EmpiricalSourceDistribution(vocab_size=vocab_size, probs=p_emp, length=length)
#     if source_distribution == "mask":
#         return MaskedSourceDistribution(mask_token=vocab_size)
#     elif source_distribution == "uniform":
#         return UniformSourceDistribution(vocab_size=vocab_size)
#     else:
#         raise ValueError(f"{source_distribution} is not supported")


# def get_loss_function(loss_function: str, path: Optional[ProbPath] = None) -> _Loss:
#     if loss_function == "cross_entropy":
#         return torch.nn.CrossEntropyLoss()
#     elif loss_function == "generalized_kl":
#         assert path is not None

#         return MixturePathGeneralizedKL(path=path)
#     elif loss_function == "editflows":
#         assert path is not None

#         return EditFlowsLoss(path=path)
#     else:
#         raise ValueError(f"{loss_function} is not supported")
