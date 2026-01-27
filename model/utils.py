from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

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
    device: torch.device = None,
):
    """
    Discrete edit sampler for Edit Flows with temperature on:
      - operation choice (insert/delete/sub)
      - token choice (for insert/sub)

    At each step we apply at most ONE edit per sequence.
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

    for step in range(num_steps):
        # t in [0,1]
        t = torch.full((B,), float(step) / float(max(1, num_steps - 1)), device=device)

        # build mask: True = valid, False = pad
        mask = (x != pad_id)

        # forward through model
        lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type = model(x_t=x, mask=mask, t=t)
        # lam_ins, logits_ins, lam_del, lam_sub, logits_sub = model(x_t=x, mask=mask, t=t)

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
        lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type = model(x_t=x, mask=mask, t=t)

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