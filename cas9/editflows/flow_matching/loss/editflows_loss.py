# Copyright (c) Meta Platforms, Inc.
# All rights reserved.

from typing import List, Optional, Sequence, Tuple, Dict
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class EditFlowsLoss(_Loss):
    """
    Edit Flows loss (Eq. 23), ragged version + optional auxiliary CE.

    Base EF (unchanged):
      L_i =  (sum_j λ_ins[i][j] + sum_j λ_del[i][j] + sum_j λ_sub[i][j])
            - w_i * ( sum_{ins events e} [log λ_ins[i][slot_e] + log Q_ins[i](y_e)]
                    + sum_{del j}       [log λ_del[i][j]]
                    + sum_{sub j}       [log λ_sub[i][j] + log Q_sub[i](y_j)] )

    Optional auxiliary CE (no-sampling, log-space):
      For each aligned column i where z_t != z_1, construct unnormalized log-masses
      over tokens∪{ε} from existing heads and apply CE to the normalized mixture:
        - token column:
              log m_i(a) = log λ_sub + log Q_sub(a)
              log m_i(ε) = log λ_del
        - gap column:
              log m_i(a) = log λ_ins + log Q_ins(a)
              log m_i(ε) = -inf
      The per-site weight w_i matches the base edit term (precomputed_weight or lambda_eff).
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(None, None, reduction)

    # -------------------------------
    # VANILLA (Eq. 23) + AUX CE
    # -------------------------------
    def forward(
        self,
        lam_ins: torch.Tensor,     # (B, L)
        logits_ins: torch.Tensor,  # (B, L, V)
        lam_del: torch.Tensor,     # (B, L)
        lam_sub: torch.Tensor,     # (B, L)
        logits_sub: torch.Tensor,  # (B, L, V)
        z_t: torch.Tensor,         # (B, N) aligned, with eps_id
        z_1: torch.Tensor,         # (B, N) aligned target, with eps_id
        x_t: torch.Tensor,         # (B, L)
        valid_mask: torch.Tensor,  # (B, L) bool, False==padding, True==valid
        precomputed_weight: torch.Tensor,  # (B,) or ()  = kappa_dot/(1-kappa)
        eps_id: int,
        bos_id: int,
        eos_id: int,

        # ---- NEW ----
        use_aux_ce: bool = False,
        aux_ce_weight: float = 0.1,
    ) -> torch.Tensor:
        device = lam_ins.device
        B, L = x_t.shape
        _, N = z_t.shape
        V = logits_ins.size(-1)

        # 1) ----- RATE TERM -----
        valid_f = valid_mask.to(lam_ins.dtype)              # (B, L)
        total_rate_pos = (lam_ins + lam_del + lam_sub) * valid_f
        loss_rate = total_rate_pos.sum(dim=1)               # (B,)

        # 2) ----- EDIT TERM (Eq. 23) -----
        logp_ins = F.log_softmax(logits_ins, dim=-1)        # (B, L, V)
        logp_sub = F.log_softmax(logits_sub, dim=-1)        # (B, L, V)

        if precomputed_weight.dim() == 0:
            precomputed_weight = precomputed_weight.view(1).expand(B).to(device)
        else:
            precomputed_weight = precomputed_weight.to(device)

        loss_edit = torch.zeros(B, dtype=torch.float32, device=device)
        loss_aux  = torch.zeros(B, dtype=torch.float32, device=device)  # NEW

        for b in range(B):
            valid_len = int(valid_mask[b].sum().item())
            prefix_non_eps = 0  # number of non-ε seen so far in z_t[b, :i]

            for i in range(N):
                zt = int(z_t[b, i].item())
                z1 = int(z_1[b, i].item())

                # map aligned column i -> x_t position
                if zt != eps_id:
                    x_pos = prefix_non_eps
                    is_token = True
                    prefix_non_eps += 1
                else:
                    x_pos = prefix_non_eps   # gap column sits between tokens
                    is_token = False

                # If already matched, nothing to supervise
                if zt == z1:
                    continue

                # Skip BOS/EOS on target (should already align)
                if z1 == bos_id or z1 == eos_id:
                    continue

                # ======== Base EF edit supervision (log-space, unchanged) ========
                if is_token and (z1 == eps_id):
                    # DELETE
                    if x_pos >= valid_len:
                        raise NotImplementedError
                    x_token = int(x_t[b, x_pos].item())
                    if x_token == bos_id or x_token == eos_id:
                        continue
                    lam = lam_del[b, x_pos].clamp_min(1e-12)
                    log_u_req = torch.log(lam)

                elif is_token and (z1 != eps_id) and (zt != z1):
                    # SUBSTITUTE
                    if x_pos >= valid_len:
                        raise NotImplementedError
                    x_token = int(x_t[b, x_pos].item())
                    if x_token == bos_id or x_token == eos_id:
                        continue
                    lam = lam_sub[b, x_pos].clamp_min(1e-12)
                    logp_tok = logp_sub[b, x_pos, z1]
                    log_u_req = torch.log(lam) + logp_tok

                elif (not is_token) and (z1 != eps_id):
                    # INSERT (in gap after position x_pos-1, clamped to [0..valid_len-1])
                    ins_pos = x_pos - 1
                    if ins_pos < 0:
                        ins_pos = 0
                    if valid_len == 0:
                        ins_pos = 0
                    elif ins_pos >= valid_len:
                        ins_pos = valid_len - 1
                    x_token = int(x_t[b, ins_pos].item())
                    if x_token == eos_id:
                        continue
                    lam = lam_ins[b, ins_pos].clamp_min(1e-12)
                    logp_tok = logp_ins[b, ins_pos, z1]
                    log_u_req = torch.log(lam) + logp_tok

                else:
                    raise NotImplementedError

                w = precomputed_weight[b]
                loss_edit[b] += - w * log_u_req

                # ======== NEW: Auxiliary CE over local mixture (log-space) ========
                if use_aux_ce and aux_ce_weight > 0.0:
                    if is_token:
                        if x_pos >= valid_len:
                            continue
                        # token classes (V): log λ_sub + log Q_sub
                        log_m_tokens = torch.log(lam_sub[b, x_pos].clamp_min(1e-12)).unsqueeze(-1) \
                                       + logp_sub[b, x_pos]                                  # (V,)
                        # ε class: log λ_del
                        log_m_eps = torch.log(lam_del[b, x_pos].clamp_min(1e-12))            # ()
                    else:
                        # gap column uses insertion at ins_pos
                        ins_pos = x_pos - 1
                        if ins_pos < 0:
                            ins_pos = 0
                        if valid_len == 0:
                            ins_pos = 0
                        elif ins_pos >= valid_len:
                            ins_pos = valid_len - 1
                        if int(x_t[b, ins_pos].item()) == eos_id:
                            # skip if the gap maps "after EOS"
                            continue
                        # token classes (V): log λ_ins + log Q_ins
                        log_m_tokens = torch.log(lam_ins[b, ins_pos].clamp_min(1e-12)).unsqueeze(-1) \
                                       + logp_ins[b, ins_pos]                                 # (V,)
                        # no deletion mass in a pure gap column
                        log_m_eps = torch.tensor(float("-inf"), device=device, dtype=log_m_tokens.dtype)

                    log_m = torch.cat([log_m_tokens, log_m_eps.view(1)], dim=-1)              # (V+1,)
                    target_idx = z1 if z1 != eps_id else V
                    ce_logits = log_m.unsqueeze(0)                                            # (1, V+1)
                    ce_target = torch.tensor([target_idx], device=device, dtype=torch.long)
                    ce_val = F.cross_entropy(ce_logits, ce_target, reduction="sum")
                    loss_aux[b] += w * ce_val

        # 3) ----- COMBINE -----
        loss_base = loss_rate + loss_edit  # non-auxiliary loss (B,)
        loss_aux_weighted_batch = aux_ce_weight * loss_aux if use_aux_ce and aux_ce_weight > 0.0 else torch.zeros_like(loss_base)
        loss_total = loss_base + loss_aux_weighted_batch  # (B,)
        
        # Compute unweighted and weighted aux losses for logging (scalars)
        # Note: We take mean to get a scalar for logging (consistent with other components)
        loss_aux_unweighted = loss_aux.mean() if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)
        loss_aux_weighted = aux_ce_weight * loss_aux_unweighted if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)
        
        # Return total loss for optimization and components for logging
        # Detach components from graph since they're only used for logging
        loss_components = {
            "loss_base": loss_base.mean().detach(),
            "loss_aux_unweighted": loss_aux_unweighted.detach(),
            "loss_aux_weighted": loss_aux_weighted.detach(),
            "loss_rate": loss_rate.mean().detach(),
            "loss_edit": loss_edit.mean().detach(),
        }
        return loss_total.mean(), loss_components
    
    # --------------------------------
    # LOCALIZED (App. C.1) + AUX CE
    # --------------------------------
    def forward_localized(
        self,
        lam_ins: torch.Tensor,     # (B, L)
        logits_ins: torch.Tensor,  # (B, L, V)
        lam_del: torch.Tensor,     # (B, L)
        lam_sub: torch.Tensor,     # (B, L)
        logits_sub: torch.Tensor,  # (B, L, V)
        z_t: torch.Tensor,         # (B, N) aligned, with eps_id
        z_1: torch.Tensor,         # (B, N) aligned target, with eps_id
        x_t: torch.Tensor,         # (B, L)
        valid_mask: torch.Tensor,  # (B, L) bool, False==padding, True==valid

        # Localized weights
        lambda_indep: torch.Tensor,        # (B,) or ()
        M_t: torch.Tensor,                 # bool (B, N, N)
        lambda_prop: torch.Tensor | float, # () or (B,)

        eps_id: int,
        bos_id: int,
        eos_id: int,

        # ---- NEW ----
        use_aux_ce: bool = False,
        aux_ce_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = lam_ins.device
        B, L = x_t.shape
        _, N = z_t.shape
        V = logits_ins.size(-1)

        # 1) ----- RATE TERM -----
        valid_f = valid_mask.to(lam_ins.dtype)
        total_rate_pos = (lam_ins + lam_del + lam_sub) * valid_f
        loss_rate = total_rate_pos.sum(dim=1)               # (B,)

        # 2) ----- PREP -----
        logp_ins = F.log_softmax(logits_ins, dim=-1)
        logp_sub = F.log_softmax(logits_sub, dim=-1)

        if lambda_indep.dim() == 0:
            lambda_indep = lambda_indep.view(1).expand(B).to(device)
        else:
            lambda_indep = lambda_indep.to(device)

        if not torch.is_tensor(lambda_prop):
            lambda_prop = torch.tensor(lambda_prop, dtype=lambda_indep.dtype, device=device)
        if lambda_prop.dim() == 0:
            lambda_prop = lambda_prop.view(1).expand(B).to(device)
        else:
            lambda_prop = lambda_prop.to(device)

        assert M_t.dtype == torch.bool and M_t.shape[1] == N and M_t.shape[2] == N, \
            "M_t must be bool (B, N, N) with N equal to aligned length"

        neighbor_or = torch.zeros_like(M_t, dtype=torch.bool)
        neighbor_or[:, :, 1:] |= M_t[:, :, :-1]
        neighbor_or[:, :, :-1] |= M_t[:, :, 1:]
        neighbor_count = neighbor_or.sum(dim=1).to(dtype=lambda_indep.dtype)   # (B, N)
        lambda_eff = lambda_indep.view(B, 1) + lambda_prop.view(B, 1) * neighbor_count

        # 3) ----- EDIT + AUX CE -----
        loss_edit = torch.zeros(B, dtype=torch.float32, device=device)
        loss_aux  = torch.zeros(B, dtype=torch.float32, device=device)  # NEW

        for b in range(B):
            valid_len = int(valid_mask[b].sum().item())
            prefix_non_eps = 0

            for i in range(N):
                zt = int(z_t[b, i].item())
                z1 = int(z_1[b, i].item())

                if zt != eps_id:
                    x_pos = prefix_non_eps
                    is_token = True
                    prefix_non_eps += 1
                else:
                    x_pos = prefix_non_eps
                    is_token = False

                if zt == z1:
                    continue
                if z1 == bos_id or z1 == eos_id:
                    continue

                # ======== Base localized EF supervision ========
                if is_token and (z1 == eps_id):
                    if x_pos >= valid_len:
                        raise NotImplementedError
                    x_token = int(x_t[b, x_pos].item())
                    if x_token == bos_id or x_token == eos_id:
                        continue
                    lam = lam_del[b, x_pos].clamp_min(1e-12)
                    log_u_req = torch.log(lam)

                elif is_token and (z1 != eps_id) and (zt != z1):
                    if x_pos >= valid_len:
                        raise NotImplementedError
                    x_token = int(x_t[b, x_pos].item())
                    if x_token == bos_id or x_token == eos_id:
                        continue
                    lam = lam_sub[b, x_pos].clamp_min(1e-12)
                    logp_tok = logp_sub[b, x_pos, z1]
                    log_u_req = torch.log(lam) + logp_tok

                elif (not is_token) and (z1 != eps_id):
                    ins_pos = x_pos - 1
                    if ins_pos < 0:
                        ins_pos = 0
                    if valid_len == 0:
                        ins_pos = 0
                    elif ins_pos >= valid_len:
                        ins_pos = valid_len - 1
                    x_token = int(x_t[b, ins_pos].item())
                    if x_token == eos_id:
                        continue
                    lam = lam_ins[b, ins_pos].clamp_min(1e-12)
                    logp_tok = logp_ins[b, ins_pos, z1]
                    log_u_req = torch.log(lam) + logp_tok

                else:
                    raise NotImplementedError

                w = lambda_eff[b, i]
                loss_edit[b] += - w * log_u_req

                # ======== NEW: Auxiliary CE over local mixture (log-space) ========
                if use_aux_ce and aux_ce_weight > 0.0:
                    if is_token:
                        if x_pos >= valid_len:
                            continue
                        log_m_tokens = torch.log(lam_sub[b, x_pos].clamp_min(1e-12)).unsqueeze(-1) \
                                       + logp_sub[b, x_pos]                                  # (V,)
                        log_m_eps = torch.log(lam_del[b, x_pos].clamp_min(1e-12))            # ()
                    else:
                        ins_pos = x_pos - 1
                        if ins_pos < 0:
                            ins_pos = 0
                        if valid_len == 0:
                            ins_pos = 0
                        elif ins_pos >= valid_len:
                            ins_pos = valid_len - 1
                        if int(x_t[b, ins_pos].item()) == eos_id:
                            continue
                        log_m_tokens = torch.log(lam_ins[b, ins_pos].clamp_min(1e-12)).unsqueeze(-1) \
                                       + logp_ins[b, ins_pos]                                 # (V,)
                        log_m_eps = torch.tensor(float("-inf"), device=device, dtype=log_m_tokens.dtype)

                    log_m = torch.cat([log_m_tokens, log_m_eps.view(1)], dim=-1)              # (V+1,)
                    target_idx = z1 if z1 != eps_id else V
                    ce_logits = log_m.unsqueeze(0)
                    ce_target = torch.tensor([target_idx], device=device, dtype=torch.long)
                    ce_val = F.cross_entropy(ce_logits, ce_target, reduction="sum")
                    loss_aux[b] += w * ce_val

        # 4) ----- COMBINE -----
        loss_base = loss_rate + loss_edit  # non-auxiliary loss (B,)
        loss_aux_weighted_batch = aux_ce_weight * loss_aux if use_aux_ce and aux_ce_weight > 0.0 else torch.zeros_like(loss_base)
        loss_total = loss_base + loss_aux_weighted_batch  # (B,)
        
        # Compute unweighted and weighted aux losses for logging (scalars)
        # Note: We take mean to get a scalar for logging (consistent with other components)
        loss_aux_unweighted = loss_aux.mean() if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)
        loss_aux_weighted = aux_ce_weight * loss_aux_unweighted if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)
        
        # Return total loss for optimization and components for logging
        # Detach components from graph since they're only used for logging
        loss_components = {
            "loss_base": loss_base.mean().detach(),
            "loss_aux_unweighted": loss_aux_unweighted.detach(),
            "loss_aux_weighted": loss_aux_weighted.detach(),
            "loss_rate": loss_rate.mean().detach(),
            "loss_edit": loss_edit.mean().detach(),
        }
        return loss_total.mean(), loss_components
        
    def reparameterized_forward(
        self,
        lam_total: torch.Tensor,   # (B, L) >=0 from model softplus
        logits_type: torch.Tensor, # (B, L, 3) over {ins, del, sub}
        logits_ins: torch.Tensor,  # (B, L, V)
        logits_sub: torch.Tensor,  # (B, L, V)
        z_t: torch.Tensor,         # (B, N) aligned, with eps_id
        z_1: torch.Tensor,         # (B, N) aligned target, with eps_id
        x_t: torch.Tensor,         # (B, L)
        valid_mask: torch.Tensor,  # (B, L) bool, True==valid
        precomputed_weight: torch.Tensor,  # (B,) or ()
        eps_id: int,
        bos_id: int,
        eos_id: int,
        gamma_rate: float = 1.0,   # weight for rate loss terms
        gamma_edit: float = 1.0,   # weight for edit loss terms
        use_aux_ce: bool = False,
        aux_ce_weight: float = 0.1,
    ):
        device = lam_total.device
        B, L = x_t.shape
        _, N = z_t.shape
        V = logits_ins.size(-1)

        lam_total = lam_total.clamp_min(1e-12)
        valid_f = valid_mask.to(lam_total.dtype)

        # π_type and token π
        pi_type     = F.softmax(logits_type, dim=-1)                # (B,L,3)
        log_pi_type = (pi_type.clamp_min(1e-12)).log()
        logp_ins    = F.log_softmax(logits_ins, dim=-1)             # (B,L,V)
        logp_sub    = F.log_softmax(logits_sub, dim=-1)

        # ---- loss_rate holds:  Σ_p λ_p  -  Σ_p W_p log λ_p
        # start with Σ_p λ_p
        loss_rate = (lam_total * valid_f).sum(dim=1)                # (B,)

        if precomputed_weight.dim() == 0:
            precomputed_weight = precomputed_weight.view(1).expand(B).to(device)
        else:
            precomputed_weight = precomputed_weight.to(device)

        loss_edit = torch.zeros(B, dtype=torch.float32, device=device)  # only the π term
        loss_aux  = torch.zeros(B, dtype=torch.float32, device=device)

        for b in range(B):
            valid_len = int(valid_mask[b].sum().item())
            prefix_non_eps = 0

            for i in range(N):
                zt = int(z_t[b, i])
                z1 = int(z_1[b, i])

                if zt != eps_id:
                    x_pos = prefix_non_eps
                    is_token = True
                    prefix_non_eps += 1
                else:
                    x_pos = prefix_non_eps
                    is_token = False

                if zt == z1:              # already matched
                    continue
                if z1 == bos_id or z1 == eos_id:
                    continue

                w = precomputed_weight[b]

                if is_token:
                    if x_pos >= valid_len:
                        continue
                    x_tok = int(x_t[b, x_pos])
                    if x_tok == bos_id or x_tok == eos_id:
                        continue

                    log_lambda = torch.log(lam_total[b, x_pos].clamp_min(1e-12))

                    # add -w * log λ_p to loss_rate  (second term)
                    loss_rate[b] += - w * log_lambda

                    if z1 == eps_id:
                        # DELETE: π term is log π_type(del)
                        log_pi = log_pi_type[b, x_pos, 1]
                    else:
                        # SUBSTITUTE: π term is log π_type(sub) + log π(token|sub)
                        log_pi = log_pi_type[b, x_pos, 2] + logp_sub[b, x_pos, z1]

                    # add -w * log π_p(e) to loss_edit  (third term)
                    loss_edit[b] += - w * log_pi

                    # optional aux CE mixture (unchanged; includes λ+π by design)
                    if use_aux_ce and aux_ce_weight > 0.0:
                        log_m_tokens = (torch.log(lam_total[b, x_pos].clamp_min(1e-12))
                                        + log_pi_type[b, x_pos, 2]).unsqueeze(-1) + logp_sub[b, x_pos]
                        log_m_eps = (torch.log(lam_total[b, x_pos].clamp_min(1e-12))
                                    + log_pi_type[b, x_pos, 1])
                        log_m = torch.cat([log_m_tokens, log_m_eps.view(1)], dim=-1)
                        target_idx = z1 if z1 != eps_id else V
                        ce_val = F.cross_entropy(log_m.unsqueeze(0),
                                                torch.tensor([target_idx], device=device),
                                                reduction="sum")
                        loss_aux[b] += w * ce_val

                else:
                    # INSERT at gap after x_pos-1 (clamped)
                    ins_pos = max(0, min(valid_len - 1, x_pos - 1)) if valid_len > 0 else 0
                    if int(x_t[b, ins_pos]) == eos_id:
                        continue

                    log_lambda = torch.log(lam_total[b, ins_pos].clamp_min(1e-12))
                    loss_rate[b] += - w * log_lambda

                    log_pi = log_pi_type[b, ins_pos, 0] + logp_ins[b, ins_pos, z1]
                    loss_edit[b] += - w * log_pi

                    if use_aux_ce and aux_ce_weight > 0.0:
                        log_m_tokens = (torch.log(lam_total[b, ins_pos].clamp_min(1e-12))
                                        + log_pi_type[b, ins_pos, 0]).unsqueeze(-1) + logp_ins[b, ins_pos]
                        log_m_eps = torch.tensor(float("-inf"), device=device, dtype=log_m_tokens.dtype)
                        log_m = torch.cat([log_m_tokens, log_m_eps.view(1)], dim=-1)
                        target_idx = z1 if z1 != eps_id else V
                        ce_val = F.cross_entropy(log_m.unsqueeze(0),
                                                torch.tensor([target_idx], device=device),
                                                reduction="sum")
                        loss_aux[b] += w * ce_val

        # Store unweighted versions before applying gamma weights
        loss_rate_unweighted = loss_rate.clone()
        loss_edit_unweighted = loss_edit.clone()

        # weigh rate loss by gamma_rate and edit loss by gamma_edit
        loss_rate_weighted = gamma_rate * loss_rate
        loss_edit_weighted = gamma_edit * loss_edit

        loss_base_weighted = loss_rate_weighted + loss_edit_weighted
        loss_base_unweighted = loss_rate_unweighted + loss_edit_unweighted
        loss_aux_weighted = aux_ce_weight * loss_aux if use_aux_ce and aux_ce_weight > 0.0 else torch.zeros_like(loss_base_weighted)
        
        # Total loss with weights (used for optimization)
        loss_total_weighted = loss_base_weighted + loss_aux_weighted
        # Total loss without weights (for logging)
        loss_total_unweighted = loss_base_unweighted + loss_aux_weighted

        comps = {
            "loss_rate_unweighted": loss_rate_unweighted.mean().detach(),
            "loss_rate_weighted": loss_rate_weighted.mean().detach(),
            "loss_edit_unweighted": loss_edit_unweighted.mean().detach(),
            "loss_edit_weighted": loss_edit_weighted.mean().detach(),
            "loss_total_weighted": loss_total_weighted.mean().detach(),
            "loss_total_unweighted": loss_total_unweighted.mean().detach(),
            # Keep backward compatibility
            "loss_base": loss_base_weighted.mean().detach(),
            "loss_rate": loss_rate_weighted.mean().detach(),   # contains term1 + term2
            "loss_edit": loss_edit_weighted.mean().detach(),   # contains only term3
            "loss_aux_unweighted": (loss_aux.mean().detach()
                                    if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)),
            "loss_aux_weighted": (loss_aux_weighted.mean().detach()
                                if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)),
        }
        return loss_total_weighted.mean(), comps



    def reparameterized_forward_localized(
        self,
        lam_total: torch.Tensor,
        logits_type: torch.Tensor,
        logits_ins: torch.Tensor,
        logits_sub: torch.Tensor,
        z_t: torch.Tensor,
        z_1: torch.Tensor,
        x_t: torch.Tensor,
        valid_mask: torch.Tensor,
        lambda_indep: torch.Tensor,
        M_t: torch.Tensor,
        lambda_prop: torch.Tensor | float,
        eps_id: int,
        bos_id: int,
        eos_id: int,
        gamma_rate: float = 1.0,   # weight for rate loss terms
        gamma_edit: float = 1.0,   # weight for edit loss terms
        use_aux_ce: bool = False,
        aux_ce_weight: float = 0.1,
    ):
        device = lam_total.device
        B, L = x_t.shape
        _, N = z_t.shape
        V = logits_ins.size(-1)

        lam_total = lam_total.clamp_min(1e-12)
        valid_f = valid_mask.to(lam_total.dtype)

        pi_type     = F.softmax(logits_type, dim=-1)
        log_pi_type = (pi_type.clamp_min(1e-12)).log()
        logp_ins    = F.log_softmax(logits_ins, dim=-1)
        logp_sub    = F.log_softmax(logits_sub, dim=-1)

        # Σ_p λ_p
        loss_rate = (lam_total * valid_f).sum(dim=1)

        # localized weights
        if lambda_indep.dim() == 0:
            lambda_indep = lambda_indep.view(1).expand(B).to(device)
        else:
            lambda_indep = lambda_indep.to(device)

        if not torch.is_tensor(lambda_prop):
            lambda_prop = torch.tensor(lambda_prop, dtype=lambda_indep.dtype, device=device)
        if lambda_prop.dim() == 0:
            lambda_prop = lambda_prop.view(1).expand(B).to(device)
        else:
            lambda_prop = lambda_prop.to(device)

        assert M_t.dtype == torch.bool and M_t.shape[1] == N and M_t.shape[2] == N

        neighbor_or = torch.zeros_like(M_t, dtype=torch.bool)
        neighbor_or[:, :, 1:] |= M_t[:, :, :-1]
        neighbor_or[:, :, :-1] |= M_t[:, :, 1:]
        neighbor_count = neighbor_or.sum(dim=1).to(dtype=lambda_indep.dtype)  # (B,N)
        lambda_eff = lambda_indep.view(B,1) + lambda_prop.view(B,1) * neighbor_count

        loss_edit = torch.zeros(B, dtype=torch.float32, device=device)
        loss_aux  = torch.zeros(B, dtype=torch.float32, device=device)

        for b in range(B):
            valid_len = int(valid_mask[b].sum().item())
            prefix_non_eps = 0

            for i in range(N):
                zt = int(z_t[b, i]); z1 = int(z_1[b, i])

                if zt != eps_id:
                    x_pos = prefix_non_eps; is_token = True; prefix_non_eps += 1
                else:
                    x_pos = prefix_non_eps; is_token = False

                if zt == z1 or z1 == bos_id or z1 == eos_id:
                    continue

                w = lambda_eff[b, i]

                if is_token:
                    if x_pos >= valid_len:
                        continue
                    x_tok = int(x_t[b, x_pos])
                    if x_tok == bos_id or x_tok == eos_id:
                        continue

                    log_lambda = torch.log(lam_total[b, x_pos].clamp_min(1e-12))
                    loss_rate[b] += - w * log_lambda

                    if z1 == eps_id:
                        log_pi = log_pi_type[b, x_pos, 1]
                    else:
                        log_pi = log_pi_type[b, x_pos, 2] + logp_sub[b, x_pos, z1]
                    loss_edit[b] += - w * log_pi

                    if use_aux_ce and aux_ce_weight > 0.0:
                        log_m_tokens = (torch.log(lam_total[b, x_pos].clamp_min(1e-12))
                                        + log_pi_type[b, x_pos, 2]).unsqueeze(-1) + logp_sub[b, x_pos]
                        log_m_eps = (torch.log(lam_total[b, x_pos].clamp_min(1e-12))
                                    + log_pi_type[b, x_pos, 1])
                        log_m = torch.cat([log_m_tokens, log_m_eps.view(1)], dim=-1)
                        target_idx = z1 if z1 != eps_id else V
                        ce_val = F.cross_entropy(log_m.unsqueeze(0),
                                                torch.tensor([target_idx], device=device),
                                                reduction="sum")
                        loss_aux[b] += w * ce_val

                else:
                    ins_pos = max(0, min(valid_len - 1, x_pos - 1)) if valid_len > 0 else 0
                    if int(x_t[b, ins_pos]) == eos_id:
                        continue

                    log_lambda = torch.log(lam_total[b, ins_pos].clamp_min(1e-12))
                    loss_rate[b] += - w * log_lambda

                    log_pi = log_pi_type[b, ins_pos, 0] + logp_ins[b, ins_pos, z1]
                    loss_edit[b] += - w * log_pi

                    if use_aux_ce and aux_ce_weight > 0.0:
                        log_m_tokens = (torch.log(lam_total[b, ins_pos].clamp_min(1e-12))
                                        + log_pi_type[b, ins_pos, 0]).unsqueeze(-1) + logp_ins[b, ins_pos]
                        log_m_eps = torch.tensor(float("-inf"), device=device, dtype=log_m_tokens.dtype)
                        log_m = torch.cat([log_m_tokens, log_m_eps.view(1)], dim=-1)
                        target_idx = z1 if z1 != eps_id else V
                        ce_val = F.cross_entropy(log_m.unsqueeze(0),
                                                torch.tensor([target_idx], device=device),
                                                reduction="sum")
                        loss_aux[b] += w * ce_val

        # Store unweighted versions before applying gamma weights
        loss_rate_unweighted = loss_rate.clone()
        loss_edit_unweighted = loss_edit.clone()

        # weigh rate loss by gamma_rate and edit loss by gamma_edit
        loss_rate_weighted = gamma_rate * loss_rate
        loss_edit_weighted = gamma_edit * loss_edit
        loss_base_weighted = loss_rate_weighted + loss_edit_weighted
        loss_base_unweighted = loss_rate_unweighted + loss_edit_unweighted
        loss_aux_weighted = aux_ce_weight * loss_aux if use_aux_ce and aux_ce_weight > 0.0 else torch.zeros_like(loss_base_weighted)
        
        # Total loss with weights (used for optimization)
        loss_total_weighted = loss_base_weighted + loss_aux_weighted
        # Total loss without weights (for logging)
        loss_total_unweighted = loss_base_unweighted + loss_aux_weighted

        comps = {
            "loss_rate_unweighted": loss_rate_unweighted.mean().detach(),
            "loss_rate_weighted": loss_rate_weighted.mean().detach(),
            "loss_edit_unweighted": loss_edit_unweighted.mean().detach(),
            "loss_edit_weighted": loss_edit_weighted.mean().detach(),
            "loss_total_weighted": loss_total_weighted.mean().detach(),
            "loss_total_unweighted": loss_total_unweighted.mean().detach(),
            # Keep backward compatibility
            "loss_base": loss_base_weighted.mean().detach(),
            "loss_rate": loss_rate_weighted.mean().detach(),   # term1 + term2
            "loss_edit": loss_edit_weighted.mean().detach(),   # term3 only
            "loss_aux_unweighted": (loss_aux.mean().detach()
                                    if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)),
            "loss_aux_weighted": (loss_aux_weighted.mean().detach()
                                if use_aux_ce and aux_ce_weight > 0.0 else torch.tensor(0.0, device=device)),
        }
        return loss_total_weighted.mean(), comps



