from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


Tensor = torch.Tensor


def _optimal_align_core(core0: torch.Tensor, core1: torch.Tensor, eps_id: int):
    """
    Edit-distance alignment on the *core* (no BOS/EOS).
    Returns two python lists of ints of the same length, using eps_id for gaps.
    """
    L0 = core0.size(0)
    L1 = core1.size(0)

    dp = torch.zeros((L0 + 1, L1 + 1), dtype=torch.long, device=core0.device)
    for i in range(1, L0 + 1):
        dp[i, 0] = i
    for j in range(1, L1 + 1):
        dp[0, j] = j

    for i in range(1, L0 + 1):
        for j in range(1, L1 + 1):
            cost_sub = 0 if core0[i-1].item() == core1[j-1].item() else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,          # delete core0[i-1]
                dp[i, j-1] + 1,          # insert core1[j-1]
                dp[i-1, j-1] + cost_sub  # match/sub
            )

    z0_core = []
    z1_core = []
    i, j = L0, L1
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost_sub = 0 if core0[i-1].item() == core1[j-1].item() else 1
            if dp[i, j].item() == dp[i-1, j-1].item() + cost_sub:
                z0_core.append(int(core0[i-1].item()))
                z1_core.append(int(core1[j-1].item()))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i, j].item() == dp[i-1, j].item() + 1:
            z0_core.append(int(core0[i-1].item()))
            z1_core.append(eps_id)
            i -= 1
            continue
        if j > 0 and dp[i, j].item() == dp[i, j-1].item() + 1:
            z0_core.append(eps_id)
            z1_core.append(int(core1[j-1].item()))
            j -= 1
            continue

    z0_core.reverse()
    z1_core.reverse()
    return z0_core, z1_core


def _suboptimal_align_core(core0: torch.Tensor, core1: torch.Tensor, eps_id: int):
    """
    Left-align cores; pad the shorter core with eps_id.
    """
    L0 = core0.size(0)
    L1 = core1.size(0)
    N = max(L0, L1)
    z0_core, z1_core = [], []
    for k in range(N):
        tok0 = int(core0[k].item()) if k < L0 else eps_id
        tok1 = int(core1[k].item()) if k < L1 else eps_id
        z0_core.append(tok0)
        z1_core.append(tok1)
    return z0_core, z1_core


def build_z0_z1_with_alignment(
    x0: torch.Tensor,  # (B, L0), padded with pad_id, contains BOS/EOS
    x1: torch.Tensor,  # (B, L1), padded with pad_id, contains BOS/EOS
    eps_id: int,
    pad_id: int,
    bos_id: int,   
    eos_id: int,   
    p_optimal: float = 0.6,
    sample_type: str = 'regular',
):
    """
    Align x0 and x1 such that:
      - BOS aligns with BOS
      - EOS aligns with EOS
      - between BOS and EOS we align with eps_id
      - after EOS we pad with pad_id

    Returns:
      z0: (B, N_max)
      z1: (B, N_max)
    """
    device = x0.device
    B = x0.size(0)

    z0_list = []
    z1_list = []
    max_len = 0

    rand = torch.rand(B, device=device)

    for b in range(B):
        # strip pads
        seq0 = x0[b][x0[b] != pad_id]  # e.g. [BOS, ..., EOS]
        seq1 = x1[b][x1[b] != pad_id]

        # find BOS/EOS positions (assume 1 each, in order)
        # usually BOS is at index 0, but let's be safe
        bos_pos0 = (seq0 == bos_id).nonzero(as_tuple=False)[0, 0].item()
        bos_pos1 = (seq1 == bos_id).nonzero(as_tuple=False)[0, 0].item()
        eos_pos0 = (seq0 == eos_id).nonzero(as_tuple=False)[0, 0].item()
        eos_pos1 = (seq1 == eos_id).nonzero(as_tuple=False)[0, 0].item()

        # cores: everything between BOS and EOS
        core0 = seq0[bos_pos0 + 1 : eos_pos0]  # may be empty
        core1 = seq1[bos_pos1 + 1 : eos_pos1]

        # pick alignment strategy for the core
        if rand[b].item() < p_optimal:
            core0_aligned, core1_aligned = _optimal_align_core(core0, core1, eps_id)
        else:
            core0_aligned, core1_aligned = _suboptimal_align_core(core0, core1, eps_id)

        # rebuild full aligned sequences: [BOS] + core_aligned + [EOS]
        aligned0 = [bos_id] + core0_aligned + [eos_id]
        aligned1 = [bos_id] + core1_aligned + [eos_id]

        cur_len = len(aligned0)
        assert cur_len == len(aligned1)
        if cur_len > max_len:
            max_len = cur_len

        z0_list.append(aligned0)
        z1_list.append(aligned1)

    # pad with pad_id AFTER eos
    z0 = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    z1 = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)

    for b in range(B):
        cur = len(z0_list[b])
        z0[b, :cur] = torch.tensor(z0_list[b], device=device, dtype=torch.long)
        z1[b, :cur] = torch.tensor(z1_list[b], device=device, dtype=torch.long)

    return z0, z1

def remove_eps(
    z_t: torch.Tensor,   # (B, N)
    eps_id: int,
    pad_id: int,
    return_mask: bool = True,
):
    device = z_t.device
    B, N = z_t.shape

    x_t = []
    for b in range(B):
        seq = z_t[b]
        core = seq[seq != eps_id]  # remove eps
        x_t.append(core)

    x_t = pad_sequence(x_t, batch_first=True, padding_value=pad_id)
    mask = (x_t != pad_id).bool()

    if return_mask:
        return x_t, mask
    return x_t

@torch.no_grad()
def generate_from_x0(
    model,
    x0: torch.Tensor,          # (B, L) long, has BOS/EOS, padded with pad_id
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: torch.Tensor = None,  # 1D tensor of vocab ids we can generate
    num_steps: int = 32,
    max_len_cap: int = None,
    op_temperature: float = 1.0,          # temperature for choosing insert vs delete vs sub
    token_temperature: float = 1.0,       # temperature for choosing the token to insert/sub
    pos_temperature: float = 1.0,        # temperature for sampling position (reparameterized models only)
    device: torch.device = None,
    is_reparameterized: bool = None,      # If None, will auto-detect from model output
    convert_to_vanilla_outputs: bool = False,  # If False, use direct sampling for reparameterized models
):
    """
    Discrete edit sampler for Edit Flows with temperature on:
      - operation choice (insert/delete/sub)
      - token choice (for insert/sub)
      - position choice (for reparameterized models when convert_to_vanilla_outputs=False)

    At each step we apply at most ONE edit per sequence.
    
    Supports both base and reparameterized models:
    - Base: outputs (lam_ins, logits_ins, lam_del, lam_sub, logits_sub)
    - Reparameterized: outputs (lam_total, logits_type, logits_ins, logits_sub)
    
    For reparameterized models:
    - If convert_to_vanilla_outputs=True (default): converts to base format and uses
      best-position-per-operation approach
    - If convert_to_vanilla_outputs=False: uses direct sampling:
      1. Samples position from lam_total using pos_temperature
      2. Samples edit type from logits_type at sampled position using op_temperature
      3. Samples token if needed (insert/sub) using token_temperature
    """
    if device is None:
        device = x0.device
    x = x0.clone().to(device)
    B = x.size(0)

    def sample_token_from_logits(logits_row: torch.Tensor) -> int:
        """
        logits_row: (V,)
        Apply temperature + allowed_tokens filtering, then sample.
        """
        logit = logits_row
        if allowed_tokens is not None:
            mask = torch.zeros_like(logit, dtype=torch.bool)
            mask[allowed_tokens] = True
            logit = logit.masked_fill(~mask, -1e4)

        if token_temperature is not None and token_temperature > 0.0:
            logit = logit / token_temperature

        probs = F.softmax(logit, dim=-1)
        # multinomial expects probs >= 0 and sum=1
        idx = torch.multinomial(probs, num_samples=1)
        return int(idx.item())

    # Auto-detect model type if not specified
    if is_reparameterized is None:
        # Try to detect from model class name first (more efficient)
        model_class_name = model.__class__.__name__
        if "Reparameterized" in model_class_name:
            is_reparameterized = True
        else:
            # Fall back to test forward pass to detect model type
            test_t = torch.zeros(1, device=device)
            test_mask = torch.ones(1, x.size(1), dtype=torch.bool, device=device)
            test_output = model(x_t=x[:1], mask=test_mask, t=test_t)
            # Reparameterized models return 4 (SMILES) or 8 (Protein) values
            is_reparameterized = len(test_output) in (4, 8)

    for step in range(num_steps):
        # t in [0,1]
        t = torch.full((B,), float(step) / float(max(1, num_steps - 1)), device=device)

        # build mask: True = valid, False = pad
        mask = (x != pad_id)

        # forward through model
        model_output = model(x_t=x, mask=mask, t=t)
        
        if is_reparameterized:
            if len(model_output) == 8:
                # ReparameterizedProteinEditFlowModel: returns all decomposed rates + total + type info
                lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, logits_type, pi_type = model_output
                # pi_type is already computed, no need to recompute
            elif len(model_output) == 4:
                # ReparameterizedSMILESEditFlowModel: (lam_total, logits_type, logits_ins, logits_sub)
                lam_total, logits_type, logits_ins, logits_sub = model_output
                pi_type = F.softmax(logits_type, dim=-1)  # (B, L, 3) over {ins, del, sub}
                lam_ins = lam_total * pi_type[:, :, 0]   # (B, L)
                lam_del = lam_total * pi_type[:, :, 1]    # (B, L)
                lam_sub = lam_total * pi_type[:, :, 2]    # (B, L)
            else:
                raise ValueError(f"Unexpected reparameterized model output length: {len(model_output)}. Expected 4 or 8.")
            
            if convert_to_vanilla_outputs:
                # For ReparameterizedProteinEditFlowModel, we already have lam_ins/del/sub
                # For ReparameterizedSMILESEditFlowModel, we computed them above
                pass  # lam_ins, lam_del, lam_sub are already set
            else:
                # Use direct sampling approach: keep reparameterized outputs as-is
                # We'll sample position and edit type separately below
                lam_ins = None  # Not used in direct sampling mode
                lam_del = None
                lam_sub = None
        else:
            # Base model: (lam_ins, logits_ins, lam_del, lam_sub, logits_sub)
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub = model_output

        # collect new sequences
        new_seqs = []
        max_len_this_round = 0

        for b in range(B):
            seq = x[b]
            valid = (seq != pad_id)
            tokens = seq[valid].tolist()  # python list

            if len(tokens) == 0:
                new_seq = torch.tensor([], device=device, dtype=torch.long)
                new_seqs.append(new_seq)
                continue

            # find EOS pos
            try:
                eos_pos = tokens.index(eos_id)
            except ValueError:
                eos_pos = len(tokens) - 1

            length_b = valid.sum().item()

            if is_reparameterized and not convert_to_vanilla_outputs:
                # Direct sampling approach for reparameterized models
                lam_total_b = lam_total[b]  # (L,)
                logits_type_b = logits_type[b]  # (L, 3)
                logits_ins_b = logits_ins[b]  # (L, V)
                logits_sub_b = logits_sub[b]  # (L, V)
                
                # 1. Sample position from lam_total with pos_temperature
                # Mask out invalid positions (after EOS, or BOS/EOS for certain operations)
                # For now, we'll allow sampling from all valid positions, then filter based on edit type
                lam_total_valid = lam_total_b[:length_b].clone()  # Only consider valid positions
                
                # Apply temperature to position distribution
                if pos_temperature is not None and pos_temperature > 0.0:
                    pos_logits = lam_total_valid / pos_temperature
                    pos_probs = F.softmax(pos_logits, dim=-1)
                    sampled_pos = int(torch.multinomial(pos_probs, 1).item())
                elif pos_temperature == 0.0:
                    # Greedy sampling: choose position with highest lam_total
                    sampled_pos = int(torch.argmax(lam_total_valid).item())
                else:
                    # Default behavior when pos_temperature is None: use softmax without temperature scaling
                    pos_probs = F.softmax(lam_total_valid, dim=-1)
                    sampled_pos = int(torch.multinomial(pos_probs, 1).item())
                
                # 2. Sample edit type from logits_type at the sampled position with op_temperature
                edit_type_logits = logits_type_b[sampled_pos]  # (3,) for {ins, del, sub}
                
                if op_temperature is not None and op_temperature > 0.0:
                    edit_type_logits_scaled = edit_type_logits / op_temperature
                else:
                    edit_type_logits_scaled = edit_type_logits
                
                edit_type_probs = F.softmax(edit_type_logits_scaled, dim=-1)
                op_idx = int(torch.multinomial(edit_type_probs, 1).item())
                
                # 3. Apply the sampled edit
                # 0 -> insert, 1 -> delete, 2 -> sub
                if op_idx == 0:
                    # insertion: can insert at any position, but skip after EOS
                    if sampled_pos < eos_pos:
                        ins_tok = sample_token_from_logits(logits_ins_b[sampled_pos])
                        tokens = tokens[:sampled_pos + 1] + [ins_tok] + tokens[sampled_pos + 1:]
                    # else: skip insertion if position is at or after EOS
                elif op_idx == 1:
                    # deletion: skip BOS/EOS
                    if tokens[sampled_pos] != bos_id and tokens[sampled_pos] != eos_id:
                        tokens = tokens[:sampled_pos] + tokens[sampled_pos + 1:]
                    # else: skip deletion if position is BOS/EOS
                else:  # op_idx == 2
                    # substitution: skip BOS/EOS
                    if tokens[sampled_pos] != bos_id and tokens[sampled_pos] != eos_id:
                        sub_tok = sample_token_from_logits(logits_sub_b[sampled_pos])
                        tokens = tokens[:sampled_pos] + [sub_tok] + tokens[sampled_pos + 1:]
                    # else: skip substitution if position is BOS/EOS
            else:
                # Original approach: convert to vanilla outputs or use base model outputs
                lam_ins_b = lam_ins[b]
                lam_del_b = lam_del[b]
                lam_sub_b = lam_sub[b]
                logits_ins_b = logits_ins[b]
                logits_sub_b = logits_sub[b]

                # --- collect best candidate per op ---

                # insertion: pick position with highest lambda, but skip after EOS
                best_ins_pos = None
                best_ins_val = 0.0
                for i in range(length_b):
                    if tokens[i] == eos_id:
                        continue
                    val = lam_ins_b[i].item()
                    if val > best_ins_val:
                        best_ins_val = val
                        best_ins_pos = i

                # deletion: pick position with highest lambda, skip BOS/EOS
                best_del_pos = None
                best_del_val = 0.0
                for i in range(length_b):
                    if tokens[i] == bos_id or tokens[i] == eos_id:
                        continue
                    val = lam_del_b[i].item()
                    if val > best_del_val:
                        best_del_val = val
                        best_del_pos = i

                # substitution: pick position with highest lambda, skip BOS/EOS
                best_sub_pos = None
                best_sub_val = 0.0
                for i in range(length_b):
                    if tokens[i] == bos_id or tokens[i] == eos_id:
                        continue
                    val = lam_sub_b[i].item()
                    if val > best_sub_val:
                        best_sub_val = val
                        best_sub_pos = i

                # --- choose which operation to apply ---
                # we form a 3-vector of op "scores" = the lambdas
                op_scores = torch.tensor(
                    [best_ins_val, best_del_val, best_sub_val],
                    device=device,
                    dtype=torch.float32,
                )

                # if all zero-ish, just keep sequence
                if torch.all(op_scores <= 1e-6):
                    new_seq = torch.tensor(tokens, device=device, dtype=torch.long)
                    new_seqs.append(new_seq)
                    max_len_this_round = max(max_len_this_round, new_seq.size(0))
                    continue

                # temperature over ops
                if op_temperature is not None and op_temperature > 0.0:
                    op_logits = op_scores / op_temperature
                    op_probs = F.softmax(op_logits, dim=0)
                    op_idx = int(torch.multinomial(op_probs, 1).item())
                else:
                    op_idx = int(torch.argmax(op_scores).item())

                # 0 -> insert, 1 -> delete, 2 -> sub
                if op_idx == 0:
                    # insertion
                    pos = best_ins_pos
                    if pos is not None:
                        ins_tok = sample_token_from_logits(logits_ins_b[pos])
                        tokens = tokens[:pos + 1] + [ins_tok] + tokens[pos + 1:]

                elif op_idx == 1:
                    # deletion
                    pos = best_del_pos
                    if pos is not None:
                        tokens = tokens[:pos] + tokens[pos + 1:]

                else:
                    # substitution
                    pos = best_sub_pos
                    if pos is not None:
                        sub_tok = sample_token_from_logits(logits_sub_b[pos])
                        tokens = tokens[:pos] + [sub_tok] + tokens[pos + 1:]

            # make sure we still end with EOS
            if len(tokens) == 0 or tokens[-1] != eos_id:
                tokens.append(eos_id)

            # enforce max_len_cap
            if max_len_cap is not None and len(tokens) > max_len_cap:
                tokens = tokens[:max_len_cap]
                if tokens[-1] != eos_id:
                    tokens[-1] = eos_id

            new_seq = torch.tensor(tokens, device=device, dtype=torch.long)
            new_seqs.append(new_seq)
            max_len_this_round = max(max_len_this_round, new_seq.size(0))

        # pad batch back to tensor
        x = x.new_full((B, max_len_this_round), pad_id)
        for b, seq_b in enumerate(new_seqs):
            x[b, :seq_b.size(0)] = seq_b

    return x
    
@torch.no_grad()
def generate_from_x0_ctmc(
    model,
    x0: torch.Tensor,          # (B, L) long, has BOS/EOS, padded with pad_id
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: Optional[torch.Tensor] = None,  # 1D tensor of vocab ids we can generate
    num_steps: int = 32,
    max_len_cap: Optional[int] = None,
    op_temperature: float = 1.0,          # accepted but unused (for API compat)
    token_temperature: float = 1.0,
    pos_temperature: float = 1.0,         # accepted but unused (for API compat)
    is_reparameterized: Optional[bool] = None,
    convert_to_vanilla_outputs: bool = False,
    device: Optional[torch.device] = None,
):
    """
    CTMC-style discrete-time sampler for Edit Flows / DFM.

    At each step:
      - For each position j, we sample independent Bernoulli events:
          insert with prob h * λ_ins[t,j]
          delete/sub with prob h * (λ_del[t,j] + λ_sub[t,j])
        and, if a del/sub event occurs, choose delete vs sub proportional to λ_del vs λ_sub.
      - We apply all resulting edit operations simultaneously (left-to-right).

    Supports:
      - Base model:         (lam_ins, logits_ins, lam_del, lam_sub, logits_sub)
      - Reparameterized:    (lam_total, logits_type, logits_ins, logits_sub)
        * If convert_to_vanilla_outputs=True:
            lam_ins/del/sub = lam_total * softmax(logits_type)[..., k]
        * If convert_to_vanilla_outputs=False:
            probabilities are computed directly from lam_total and π_type.
    """

    if device is None:
        device = x0.device

    x = x0.clone().to(device)
    B = x.size(0)

    if num_steps <= 0:
        return x

    def sample_token_from_logits(logits_row: torch.Tensor) -> int:
        """
        logits_row: (V,). Apply allowed_tokens mask + temperature, then sample.
        """
        logit = logits_row
        if allowed_tokens is not None:
            mask = torch.zeros_like(logit, dtype=torch.bool)
            mask[allowed_tokens] = True
            logit = logit.masked_fill(~mask, -1e9)  # effectively remove disallowed tokens

        if token_temperature is not None and token_temperature > 0.0 and token_temperature != 1.0:
            logit = logit / token_temperature

        probs = F.softmax(logit, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        return int(idx.item())

    # Auto-detect reparameterized vs base model if not specified
    if is_reparameterized is None:
        model_class_name = model.__class__.__name__
        if "Reparameterized" in model_class_name:
            is_reparameterized = True
        else:
            # Fallback: look at forward output length
            with torch.no_grad():
                t_test = torch.zeros(1, device=device)
                mask_test = torch.ones(1, x.size(1), dtype=torch.bool, device=device)
                test_out = model(x_t=x[:1], mask=mask_test, t=t_test)
            # Reparameterized models return 4 (SMILES) or 8 (Protein) values
            is_reparameterized = (len(test_out) in (4, 8))

    # Time step size h; t_k = k * h, k = 0..num_steps-1
    if num_steps == 1:
        h = 1.0
    else:
        h = 1.0 / float(num_steps - 1)

    for step in range(num_steps):
        t_scalar = step * h
        t = torch.full((B,), t_scalar, device=device, dtype=torch.float32)

        mask = (x != pad_id)
        model_out = model(x_t=x, mask=mask, t=t)

        if is_reparameterized:
            if len(model_out) == 8:
                # ReparameterizedProteinEditFlowModel: returns all decomposed rates + total + type info
                lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, logits_type, pi_type = model_out
                # pi_type is already computed, no need to recompute
            elif len(model_out) == 4:
                # ReparameterizedSMILESEditFlowModel: (lam_total, logits_type, logits_ins, logits_sub)
                lam_total, logits_type, logits_ins, logits_sub = model_out
                pi_type = F.softmax(logits_type, dim=-1)  # (B, L, 3) over {ins, del, sub}
            else:
                raise ValueError(f"Unexpected reparameterized model output length: {len(model_out)}. Expected 4 or 8.")

            if convert_to_vanilla_outputs:
                if len(model_out) == 8:
                    # For ReparameterizedProteinEditFlowModel, we already have lam_ins/del/sub
                    pass  # lam_ins, lam_del, lam_sub are already set
                else:
                    # Convert to "vanilla" λ_ins/λ_del/λ_sub, then reuse base logic
                    lam_ins = lam_total * pi_type[..., 0]
                    lam_del = lam_total * pi_type[..., 1]
                    lam_sub = lam_total * pi_type[..., 2]
            else:
                # We'll use lam_total + pi_type directly in the loop
                lam_ins = lam_del = lam_sub = None  # not used in this branch
        else:
            # Base model: (lam_ins, logits_ins, lam_del, lam_sub, logits_sub)
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub = model_out
            pi_type = None  # not used for base

        new_batch = []
        max_len_this_round = 0

        for b in range(B):
            seq = x[b]
            valid = (seq != pad_id)
            tokens = seq[valid].tolist()

            # If sequence somehow became empty, reinsert BOS/EOS
            if len(tokens) == 0:
                tokens = [bos_id, eos_id]

            # Find EOS position (default to last if missing)
            try:
                eos_pos = tokens.index(eos_id)
            except ValueError:
                eos_pos = len(tokens) - 1

            Lb = len(tokens)

            delete_mask = [False] * Lb
            sub_tokens = [None] * Lb
            ins_tokens = [None] * Lb

            if is_reparameterized and not convert_to_vanilla_outputs:
                # ----- Reparameterized CTMC branch: use lam_total + π_type directly -----
                lam_total_b = lam_total[b, :Lb]          # (Lb,)
                pi_b = pi_type[b, :Lb, :]                # (Lb, 3)
                logits_ins_b = logits_ins[b, :Lb, :]     # (Lb, V)
                logits_sub_b = logits_sub[b, :Lb, :]     # (Lb, V)

                for j in range(Lb):
                    tok_j = tokens[j]

                    # π_type components
                    pi_ins = float(pi_b[j, 0].item())
                    pi_del = float(pi_b[j, 1].item())
                    pi_sub = float(pi_b[j, 2].item())
                    lam_tot_ij = float(lam_total_b[j].item())

                    # -------- Insertion event at position j --------
                    if j < eos_pos and lam_tot_ij > 0.0 and pi_ins > 0.0:
                        p_ins = h * lam_tot_ij * pi_ins
                        p_ins = min(p_ins, 1.0)
                        if p_ins > 0.0 and torch.rand(1, device=device).item() < p_ins:
                            ins_tok = sample_token_from_logits(logits_ins_b[j])
                            ins_tokens[j] = ins_tok

                    # -------- Delete/substitute event at position j --------
                    if tok_j == bos_id or tok_j == eos_id:
                        continue  # never delete/sub BOS/EOS

                    pi_ds = pi_del + pi_sub
                    if lam_tot_ij <= 0.0 or pi_ds <= 0.0:
                        continue

                    lam_ds = lam_tot_ij * pi_ds
                    p_ds = h * lam_ds
                    p_ds = min(p_ds, 1.0)
                    if p_ds <= 0.0:
                        continue

                    if torch.rand(1, device=device).item() < p_ds:
                        # A delete/sub event occurs; choose which
                        p_del_given = pi_del / pi_ds
                        choose_del = (torch.rand(1, device=device).item() < p_del_given)

                        if choose_del:
                            delete_mask[j] = True
                            ins_tokens[j] = None
                            sub_tokens[j] = None
                        else:
                            sub_tok = sample_token_from_logits(logits_sub_b[j])
                            sub_tokens[j] = sub_tok

            else:
                # ----- Base CTMC branch (or reparam+vanilla with lam_ins/lam_del/lam_sub) -----
                lam_ins_b = lam_ins[b, :Lb]
                lam_del_b = lam_del[b, :Lb]
                lam_sub_b = lam_sub[b, :Lb]
                logits_ins_b = logits_ins[b, :Lb, :]
                logits_sub_b = logits_sub[b, :Lb, :]

                for j in range(Lb):
                    tok_j = tokens[j]

                    # -------- Insertion event at position j --------
                    if j < eos_pos:
                        lam_ij = float(lam_ins_b[j].item())
                        if lam_ij > 0.0:
                            p_ins = h * lam_ij
                            p_ins = min(p_ins, 1.0)
                            if p_ins > 0.0 and torch.rand(1, device=device).item() < p_ins:
                                ins_tok = sample_token_from_logits(logits_ins_b[j])
                                ins_tokens[j] = ins_tok

                    # -------- Delete/substitute event at position j --------
                    if tok_j == bos_id or tok_j == eos_id:
                        continue

                    lam_del_ij = float(lam_del_b[j].item())
                    lam_sub_ij = float(lam_sub_b[j].item())
                    lam_ds = lam_del_ij + lam_sub_ij
                    if lam_ds <= 0.0:
                        continue

                    p_ds = h * lam_ds
                    p_ds = min(p_ds, 1.0)
                    if p_ds <= 0.0:
                        continue

                    if torch.rand(1, device=device).item() < p_ds:
                        if lam_del_ij == 0.0:
                            choose_del = False
                        elif lam_sub_ij == 0.0:
                            choose_del = True
                        else:
                            p_del_given = lam_del_ij / lam_ds
                            choose_del = (torch.rand(1, device=device).item() < p_del_given)

                        if choose_del:
                            delete_mask[j] = True
                            ins_tokens[j] = None
                            sub_tokens[j] = None
                        else:
                            sub_tok = sample_token_from_logits(logits_sub_b[j])
                            sub_tokens[j] = sub_tok

            # -------- Apply all edits simultaneously (left-to-right) --------
            new_tokens = []
            for j in range(Lb):
                tok_j = tokens[j]

                if delete_mask[j]:
                    pass
                elif sub_tokens[j] is not None:
                    new_tokens.append(sub_tokens[j])
                else:
                    new_tokens.append(tok_j)

                if ins_tokens[j] is not None:
                    new_tokens.append(ins_tokens[j])

            # Ensure EOS is present
            if eos_id not in new_tokens:
                new_tokens.append(eos_id)

            # Enforce max length cap
            if max_len_cap is not None and len(new_tokens) > max_len_cap:
                new_tokens = new_tokens[:max_len_cap]
                if new_tokens[-1] != eos_id:
                    new_tokens[-1] = eos_id

            new_seq = torch.tensor(new_tokens, device=device, dtype=torch.long)
            new_batch.append(new_seq)
            max_len_this_round = max(max_len_this_round, new_seq.size(0))

        x_next = x.new_full((B, max_len_this_round), pad_id)
        for b, seq_b in enumerate(new_batch):
            x_next[b, :seq_b.size(0)] = seq_b

        x = x_next

    return x

def generate_from_x0_multi_edit(
    model,
    x0: torch.Tensor,          # (B, L) long, has BOS/EOS, padded with pad_id
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: torch.Tensor = None,  # 1D tensor of vocab ids we can generate
    num_steps: int = 32,
    max_len_cap: int = None,
    op_temperature: float = 1.0,          # temperature for choosing insert vs delete vs sub
    token_temperature: float = 1.0,       # temperature for choosing the token to insert/sub
    device: torch.device = None,
):
    """
    Multi-edit discrete edit sampler for Edit Flows.

    At each step:
      - For each position i, independently "fire" an edit with probability
            p_i = 1 - exp(-delta * lambda_i),
        where lambda_i = lam_ins[i] + lam_del[i] + lam_sub[i] (after masking illegal ops).
      - If fired, sample ONE op type at that position (ins/del/sub) proportional to rates,
        with optional op_temperature.
      - For ins/sub, sample token from logits with optional token_temperature and allowed_tokens.
      - Apply edits in a single left-to-right pass (avoids index-shift headaches).
    """
    if device is None:
        device = x0.device
    x = x0.clone().to(device)
    B = x.size(0)

    # User-requested: delta = 1 / num_steps
    delta = 1.0 / float(max(1, num_steps))

    def sample_token_from_logits(logits_row: torch.Tensor) -> int:
        """
        logits_row: (V,)
        Apply temperature + allowed_tokens filtering, then sample.
        """
        logit = logits_row
        if allowed_tokens is not None:
            mask = torch.zeros_like(logit, dtype=torch.bool)
            mask[allowed_tokens] = True
            logit = logit.masked_fill(~mask, -1e4)

        if token_temperature is not None and token_temperature > 0.0:
            logit = logit / token_temperature

        probs = F.softmax(logit, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        return int(idx.item())

    for step in range(num_steps):
        # t in [0,1]
        t = torch.full((B,), float(step) / float(max(1, num_steps - 1)), device=device)

        # mask: True = valid (non-pad)
        mask = (x != pad_id)

        # forward
        model_output = model(x_t=x, mask=mask, t=t)
        
        # Handle both base models (5 values) and ReparameterizedProteinEditFlowModel (8 values)
        # ReparameterizedSMILESEditFlowModel returns 4 values, but we don't use it here
        if len(model_output) == 8:
            # ReparameterizedProteinEditFlowModel: returns all decomposed rates + total + type info
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, logits_type, pi_type = model_output
        elif len(model_output) == 5:
            # Base model: (lam_ins, logits_ins, lam_del, lam_sub, logits_sub)
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub = model_output
            lam_total = None  # Not used in base model path
            logits_type = None
            pi_type = None
        else:
            raise ValueError(f"Unexpected model output length: {len(model_output)}. Expected 5 (base) or 8 (ReparameterizedProteinEditFlowModel)")

        new_seqs = []
        max_len_this_round = 0

        for b in range(B):
            seq = x[b]
            valid = (seq != pad_id)
            tokens = seq[valid].tolist()

            if len(tokens) == 0:
                new_seq = torch.tensor([], device=device, dtype=torch.long)
                new_seqs.append(new_seq)
                continue

            # Ensure there's an EOS somewhere (fallback: append later)
            if eos_id not in tokens:
                tokens = tokens + [eos_id]

            Lb = len(tokens)

            lam_ins_b = lam_ins[b][:Lb].clone()
            lam_del_b = lam_del[b][:Lb].clone()
            lam_sub_b = lam_sub[b][:Lb].clone()
            logits_ins_b = logits_ins[b][:Lb]
            logits_sub_b = logits_sub[b][:Lb]

            # --- operation legality masks at current positions ---
            tok_tensor = torch.tensor(tokens, device=device, dtype=torch.long)
            is_bos = (tok_tensor == bos_id)
            is_eos = (tok_tensor == eos_id)

            # insertion not allowed at EOS
            lam_ins_b = lam_ins_b.masked_fill(is_eos, 0.0)
            # deletion/substitution not allowed at BOS/EOS
            lam_del_b = lam_del_b.masked_fill(is_bos | is_eos, 0.0)
            lam_sub_b = lam_sub_b.masked_fill(is_bos | is_eos, 0.0)

            lam_pos_total = lam_ins_b + lam_del_b + lam_sub_b

            # fire prob per position
            # p_i = 1 - exp(-delta * lambda_i)
            p_fire = 1.0 - torch.exp(-delta * lam_pos_total.clamp(min=0.0))

            # sample fired positions
            fired = (torch.rand(Lb, device=device) < p_fire) & (lam_pos_total > 1e-12)

            # sample op type (0=ins, 1=del, 2=sub) for ALL positions (we'll use only where fired)
            rates = torch.stack([lam_ins_b, lam_del_b, lam_sub_b], dim=-1)  # (Lb, 3)

            # temperature over ops: probs ∝ rate^(1/temp) == softmax(log(rate)/temp)
            if op_temperature is not None and op_temperature > 0.0:
                op_logits = torch.log(rates + 1e-20) / op_temperature
                op_probs = F.softmax(op_logits, dim=-1)
            else:
                # greedy: pick max-rate op; represent as one-hot probs for multinomial compatibility
                op_idx_greedy = torch.argmax(rates, dim=-1)  # (Lb,)
                op_probs = F.one_hot(op_idx_greedy, num_classes=3).float()

            # multinomial per row
            # torch.multinomial accepts (n, m) -> (n, num_samples)
            op_idx = torch.multinomial(op_probs, num_samples=1).squeeze(-1)  # (Lb,)

            # pre-sample tokens for fired ins/sub positions (loop only over fired positions)
            ins_tok_map = {}
            sub_tok_map = {}
            fired_idx = fired.nonzero(as_tuple=True)[0].tolist()
            for i in fired_idx:
                oi = int(op_idx[i].item())
                if oi == 0:
                    # insertion
                    ins_tok_map[i] = sample_token_from_logits(logits_ins_b[i])
                elif oi == 2:
                    # substitution
                    sub_tok_map[i] = sample_token_from_logits(logits_sub_b[i])

            # apply edits in one pass (left-to-right)
            out = []
            for i in range(Lb):
                tok = tokens[i]
                if fired[i]:
                    oi = int(op_idx[i].item())
                    if oi == 1:
                        # deletion (already masked for BOS/EOS)
                        continue
                    elif oi == 2:
                        # substitution
                        tok = sub_tok_map.get(i, tok)

                out.append(tok)

                # insertion happens AFTER this token (and never after EOS, due to masking)
                if fired[i] and int(op_idx[i].item()) == 0:
                    out.append(ins_tok_map.get(i))

            # ensure EOS at end
            if len(out) == 0 or out[-1] != eos_id:
                out.append(eos_id)

            # enforce max_len_cap
            if max_len_cap is not None and len(out) > max_len_cap:
                out = out[:max_len_cap]
                if out[-1] != eos_id:
                    out[-1] = eos_id

            new_seq = torch.tensor(out, device=device, dtype=torch.long)
            new_seqs.append(new_seq)
            max_len_this_round = max(max_len_this_round, new_seq.size(0))

        # pad batch
        x = x.new_full((B, max_len_this_round), pad_id)
        for b, seq_b in enumerate(new_seqs):
            x[b, :seq_b.size(0)] = seq_b

    return x
