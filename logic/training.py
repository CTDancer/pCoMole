import math
from contextlib import nullcontext
from typing import Optional

import torch
from flow_matching.loss import MixturePathGeneralizedKL, EditFlowsLoss
from flow_matching.path import ProbPath
from omegaconf.dictconfig import DictConfig
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader
from utils.logging import TrainLogger

from .flow import SourceDistribution
from .state import TrainState
from ..model.utils import build_z0_z1_with_alignment, remove_eps

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from torch import Tensor

import pdb


def step(
    state: TrainState,
    loss_fn: nn.Module,                 # EditFlowsLoss
    path: ProbPath,                     # EditFlowsPathAdapter (exposes .scheduler)
    scaler: GradScaler,
    iterator: DataLoader,
    device: torch.device,
    source_distribution: SourceDistribution,
    logger: TrainLogger,
    training: bool,
    optim_params: Optional[DictConfig],
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> Tensor:
    assert (training and (optim_params is not None)) or (not training)
    state.train() if training else state.eval()

    batch = next(iterator)
    # x_1 = pad_sequence(batch['input_ids'], batch_first=True, padding_value=pad_id).to(device)
    x_1 = torch.tensor(batch["input_ids"]).to(device)
    B = x_1.shape[0]

    # === Source & time ===
    with torch.no_grad():
        eps_id = getattr(path, "eps_id", -1)
        allowed_tokens = torch.tensor([tok for tok in source_distribution._allowed_tokens if tok != eps_id]).to(device)
        
        x_0 = source_distribution.sample_x0_from_x1(x_1, pad_id=pad_id, allowed_tokens=allowed_tokens, scale_size=2, bos_id = bos_id, eos_id = eos_id)
        t = torch.rand(B, device=device)

        sched = path.scheduler(t)
        precomputed_weight = sched.d_alpha_t / sched.sigma_t     # (B,)

        z_0, z_1 = build_z0_z1_with_alignment(x_0, x_1, eps_id, pad_id, bos_id, eos_id, p_optimal=0.6)

        z_t = path.sample(z_0, z_1, t=t)
        x_t, mask = remove_eps(z_t, eps_id, pad_id)

    ctx = torch.amp.autocast('cuda', dtype=torch.float16) if training else torch.no_grad()
    with ctx:
        # pdb.set_trace()
        lam_ins, logits_ins, lam_del, lam_sub, logits_sub = state.model(x_t=x_t, mask=mask,t=t)

        loss = loss_fn(lam_ins, logits_ins, lam_del, lam_sub, logits_sub, 
                       z_t, z_1, x_t, mask, precomputed_weight, eps_id, bos_id, eos_id)

    if training:
        optimization_step(
            state=state,
            loss=loss,
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

    return loss.detach()



def _get_lr(lr: float, step: int, warmup: int, n_iters: int, eta_min_ratio: float):
    if step < warmup:
        # Linear warmup
        return lr * (step / warmup)
    else:
        # Cosine annealing
        total_steps = n_iters
        eta_min = eta_min_ratio * lr
        cosine_decay = 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup))
        )
        return eta_min + (lr - eta_min) * cosine_decay

def optimization_step(
    state: TrainState,
    scaler: GradScaler,
    loss: Tensor,
    optim_params: DictConfig,
    logger: TrainLogger,
) -> None:
    scaler.scale(loss).backward()
    scaler.unscale_(state.optimizer)

    lr = _get_lr(
        lr=optim_params.lr,
        step=state.step,
        warmup=optim_params.warmup,
        n_iters=optim_params.n_iters,
        eta_min_ratio=optim_params.eta_min_ratio,
    )

    # Update learning rate in optimizer
    for g in state.optimizer.param_groups:
        g["lr"] = lr

    if state.step % optim_params.log_lr_every == 0:
        logger.log_lr(value=lr, step=state.step)

    if optim_params.grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), max_norm=optim_params.grad_clip
        )

    scaler.step(state.optimizer)
    scaler.update()

    state.optimizer.zero_grad()
