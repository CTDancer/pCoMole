import argparse
from typing import List, Callable, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
import time

from generate import build_model_and_stuff, tokenize_input_str, detokenize_output
from objectives import GFPExcitationPred, GFPBrightPred, GFPLength
from constraints import GFP, Length, GFPEmissionPred

import pdb

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def extract_objective_vector(seqs, objective_models, device):
    values = []

    for obj in objective_models:
        scores = obj(seqs) # list of shape B
        values.append(torch.tensor(scores, device=device, dtype=torch.float32))

    return torch.stack(values, dim=1)  # (B,m)


def compute_scores_print(seqs, objective_models, constraint_models, device):
    objective_scores = extract_objective_vector(seqs, objective_models, device).squeeze(0)
    scores = [score.item() for score in objective_scores]
    for constraint in constraint_models:
        scores.append(constraint(seqs)[0])
    print(scores)


# ---------------------------------------------------------------------------
# edit utilities
# ---------------------------------------------------------------------------
def _apply_single_edit(
    tokens: List[int],
    op: str,
    pos: int,
    tok: Optional[int],
    eos_id: int,
) -> List[int]:
    if op == "ins":
        return tokens[:pos + 1] + [tok] + tokens[pos + 1:]
    elif op == "del":
        return tokens[:pos] + tokens[pos + 1:]
    elif op == "sub":
        return tokens[:pos] + [tok] + tokens[pos + 1:]
    else:
        raise ValueError(op)


def apply_edit_padded_batch(x, edit_type, pos, tok, pad_id, eos_id):
    """
    x: (B, Lmax)
    edit_type: (B,) 0/1/2  (ins/del/sub)
    pos: (B,) position
    tok: (B,) token for ins/sub (ignored for del)
    Returns: (B, Lmax)
    """
    device = x.device
    B, Lmax = x.shape
    idx = torch.arange(Lmax, device=device).view(1, Lmax).expand(B, Lmax)  # (B,L)

    x_out = x.clone()

    # substitution: overwrite at pos
    sub = (edit_type == 2)
    if sub.any():
        rows = sub.nonzero(as_tuple=True)[0]
        x_out[rows, pos[rows]] = tok[rows]

    # deletion: shift left from pos+1 onward, put pad at end
    dele = (edit_type == 1)
    if dele.any():
        rows = dele.nonzero(as_tuple=True)[0]
        p = pos[rows].view(-1, 1)  # (K,1)
        idx_rows = idx[rows]
        # for columns >= p: take from col+1 else keep
        src = torch.where(idx_rows >= p, torch.clamp(idx_rows + 1, max=Lmax - 1), idx_rows)
        x_out[rows] = x_out[rows].gather(1, src)
        x_out[rows, -1] = pad_id

    # insertion: shift right from pos onward, write tok at pos
    ins = (edit_type == 0)
    if ins.any():
        rows = ins.nonzero(as_tuple=True)[0]
        p = pos[rows].view(-1, 1)
        idx_rows = idx[rows]
        src = torch.where(idx_rows > p, idx_rows - 1, idx_rows)
        x_tmp = x_out[rows].gather(1, src)
        x_tmp[torch.arange(x_tmp.size(0), device=device), pos[rows]] = tok[rows]
        x_out[rows] = x_tmp

    # optional: ensure eos somewhere by setting the last nonpad token to eos
    # (depends on your tokenizer / special tokens)

    return x_out


def _sample_single_edit_from_outputs(
    x: torch.Tensor,             # (L,)
    lam_ins: torch.Tensor,       # (1, L)
    logits_ins: torch.Tensor,    # (1, L, V)
    lam_del: torch.Tensor,       # (1, L)
    lam_sub: torch.Tensor,       # (1, L)
    logits_sub: torch.Tensor,    # (1, L, V)
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: Optional[torch.Tensor],
    max_len_cap: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    device = x.device
    tokens = x[x != pad_id].tolist()
    L = len(tokens)

    ins_rates, del_rates, sub_rates = [], [], []
    for i in range(L):
        t_i = tokens[i]
        # insertion
        ins_rates.append(lam_ins[0, i].item() if t_i != eos_id else 0.0)
        # deletion
        if t_i == bos_id or t_i == eos_id:
            del_rates.append(0.0)
        else:
            del_rates.append(lam_del[0, i].item())
        # substitution
        if t_i == bos_id or t_i == eos_id:
            sub_rates.append(0.0)
        else:
            sub_rates.append(lam_sub[0, i].item())

    rates = torch.tensor(ins_rates + del_rates + sub_rates, device=device)
    if rates.sum().item() <= 1e-8:
        return x[x != pad_id], 0.0

    probs = rates / (rates.sum() + 1e-12)
    idx = torch.multinomial(probs, 1).item()

    # your original little random tweak
    # p = torch.randn(1).item()
    # if idx < L and p < 0.5:
    #     idx += L

    if idx < L:
        # pdb.set_trace()
        # insertion
        pos = idx
        logits_row = logits_ins[0, pos]
        if allowed_tokens is not None:
            mask = torch.zeros_like(logits_row, dtype=torch.bool, device=device)
            mask[allowed_tokens] = True
            logits_row = logits_row.masked_fill(~mask, -1e4)
        q = F.softmax(logits_row, dim=-1)
        tok = torch.multinomial(q, 1).item()
        new_tokens = _apply_single_edit(tokens, "ins", pos, tok, eos_id)
        base_rate = lam_ins[0, pos].item() * q[tok].item()

    elif idx < 2 * L:
        # deletion
        pos = idx - L
        new_tokens = _apply_single_edit(tokens, "del", pos, None, eos_id)
        base_rate = lam_del[0, pos].item()

    else:
        # pdb.set_trace()
        # substitution
        pos = idx - 2 * L
        logits_row = logits_sub[0, pos]
        if allowed_tokens is not None:
            mask = torch.zeros_like(logits_row, dtype=torch.bool, device=device)
            mask[allowed_tokens] = True
            logits_row = logits_row.masked_fill(~mask, -1e4)
        q = F.softmax(logits_row, dim=-1)
        tok = torch.multinomial(q, 1).item()
        new_tokens = _apply_single_edit(tokens, "sub", pos, tok, eos_id)
        base_rate = lam_sub[0, pos].item() * q[tok].item()

    # ensure EOS
    if len(new_tokens) == 0 or new_tokens[-1] != eos_id:
        new_tokens.append(eos_id)

    # cap length
    if max_len_cap is not None and len(new_tokens) > max_len_cap:
        new_tokens = new_tokens[:max_len_cap]
        if new_tokens[-1] != eos_id:
            new_tokens[-1] = eos_id

    return torch.tensor(new_tokens, device=device, dtype=torch.long), float(base_rate)

@torch.no_grad()
def sample_single_edit_batch(
    x,                 # (B, Lmax) padded
    lam_ins,           # (B, Lmax)
    logits_ins,        # (B, Lmax, V)
    lam_del,           # (B, Lmax)
    lam_sub,           # (B, Lmax)
    logits_sub,        # (B, Lmax, V)
    pad_id, bos_id, eos_id,
    allowed_tokens=None,
    max_len_cap=None,
):
    device = x.device
    B, Lmax = x.shape
    V = logits_ins.shape[-1]

    # active positions
    nonpad = (x != pad_id)  # (B, Lmax)
    lengths = nonpad.sum(dim=1)  # (B,) number of real tokens

    is_bos = (x == bos_id)
    is_eos = (x == eos_id)

    # mask rates on invalid positions
    ins_rate = lam_ins.clone()
    ins_rate = ins_rate.masked_fill(~nonpad, 0.0)
    ins_rate = ins_rate.masked_fill(is_eos, 0.0)  # no insertion "at" EOS (same as your scalar)

    del_rate = lam_del.clone()
    del_rate = del_rate.masked_fill(~nonpad, 0.0)
    del_rate = del_rate.masked_fill(is_bos | is_eos, 0.0)

    sub_rate = lam_sub.clone()
    sub_rate = sub_rate.masked_fill(~nonpad, 0.0)
    sub_rate = sub_rate.masked_fill(is_bos | is_eos, 0.0)

    # if max_len_cap is set, disallow insertion when already at cap
    if max_len_cap is not None:
        at_cap = lengths >= max_len_cap
        # broadcast: zero all insertion rates for those rows
        ins_rate = ins_rate.masked_fill(at_cap.unsqueeze(1), 0.0)

    rates = torch.cat([ins_rate, del_rate, sub_rate], dim=1)  # (B, 3*Lmax)
    rate_sum = rates.sum(dim=1)                                # (B,)

    alive = rate_sum > 1e-8
    if not alive.any():
        return x

    # sample only for alive rows (avoid NaNs / weirdness)
    edit_type = torch.full((B,), 2, device=device, dtype=torch.long)  # default sub
    pos = torch.zeros((B,), device=device, dtype=torch.long)

    probs_alive = rates[alive] / (rate_sum[alive].unsqueeze(1) + 1e-12)
    edit_idx_alive = torch.multinomial(probs_alive, 1).squeeze(1)  # (K,)

    Lmax_const = Lmax
    edit_type_alive = edit_idx_alive // Lmax_const  # 0=ins,1=del,2=sub
    pos_alive = edit_idx_alive % Lmax_const

    alive_rows = alive.nonzero(as_tuple=True)[0]
    edit_type[alive_rows] = edit_type_alive
    pos[alive_rows] = pos_alive

    # token sampling for ins/sub
    tok = torch.full((B,), pad_id, device=device, dtype=torch.long)

    def _mask_logits(logits):
        # logits: (K, V)
        if allowed_tokens is None:
            return logits
        # allowed_tokens should be a 1D LongTensor/list of token ids
        mask = torch.full_like(logits, -1e9)
        mask[:, allowed_tokens] = 0.0
        return logits + mask

    ins_rows = alive & (edit_type == 0)
    if ins_rows.any():
        rows = ins_rows.nonzero(as_tuple=True)[0]
        logits = logits_ins[rows, pos[rows]]  # (K, V)
        logits = _mask_logits(logits)
        q = F.softmax(logits, dim=-1)
        tok[rows] = torch.multinomial(q, 1).squeeze(1)

    sub_rows = alive & (edit_type == 2)
    if sub_rows.any():
        rows = sub_rows.nonzero(as_tuple=True)[0]
        logits = logits_sub[rows, pos[rows]]  # (K, V)
        logits = _mask_logits(logits)
        q = F.softmax(logits, dim=-1)
        tok[rows] = torch.multinomial(q, 1).squeeze(1)

    # compute new lengths after edit (for repadding)
    new_lengths = lengths.clone()
    new_lengths = new_lengths + (edit_type == 0).long() - (edit_type == 1).long()

    # safety clamp (should already be ensured by rate-masking)
    if max_len_cap is not None:
        new_lengths = torch.clamp(new_lengths, max=max_len_cap)

    Lmax_new = int(new_lengths.max().item())
    # keep at least 1 column
    Lmax_new = max(Lmax_new, 1)

    # allocate output
    x_out = torch.full((B, Lmax_new), pad_id, device=device, dtype=x.dtype)

    # We apply edits using gather-based index maps per edit group.
    idx_new = torch.arange(Lmax_new, device=device).view(1, Lmax_new).expand(B, Lmax_new)

    # 1) start from copying as much of old as fits (for rows where length doesn't shrink)
    # We'll do group-wise writes below; this initial copy is optional.
    # Better: handle each group explicitly.

    # --- SUBSTITUTION (length unchanged) ---
    sub_group = alive & (edit_type == 2)
    if sub_group.any():
        rows = sub_group.nonzero(as_tuple=True)[0]
        # copy prefix up to Lmax_new from old (clamp old indices)
        src = torch.clamp(idx_new[rows], max=Lmax - 1)
        x_tmp = x[rows].gather(1, src)
        # apply substitution at pos (only if within new width)
        p = pos[rows]
        in_range = p < Lmax_new
        if in_range.any():
            rr = torch.arange(rows.numel(), device=device)[in_range]
            x_tmp[rr, p[in_range]] = tok[rows[in_range]]
        x_out[rows] = x_tmp

    # --- DELETION (length -1) ---
    del_group = alive & (edit_type == 1)
    if del_group.any():
        rows = del_group.nonzero(as_tuple=True)[0]
        p = pos[rows].view(-1, 1)  # (K,1)
        c = idx_new[rows]          # (K, Lmax_new)
        # delete token at p: positions < p copy same, positions >= p take from c+1
        src = torch.where(c < p, c, c + 1)
        src = torch.clamp(src, max=Lmax - 1)
        x_tmp = x[rows].gather(1, src)
        x_out[rows] = x_tmp  # tail already padded by construction if Lmax_new > new_lengths[row]

    # --- INSERTION (length +1) ---
    # Here we implement insertion *after* position pos (common convention):
    # new[pos+1] = tok, and old tokens from pos+1 onward shift right by 1.
    ins_group = alive & (edit_type == 0)
    if ins_group.any():
        rows = ins_group.nonzero(as_tuple=True)[0]
        p = pos[rows].view(-1, 1)  # (K,1)
        c = idx_new[rows]          # (K, Lmax_new)

        # For columns <= p: src=c
        # For columns > p+1: src=c-1
        # For column == p+1: we'll overwrite with tok after gather
        src = torch.where(c <= p, c, c - 1)
        src = torch.clamp(src, min=0, max=Lmax - 1)
        x_tmp = x[rows].gather(1, src)

        insert_col = (p.squeeze(1) + 1)
        in_range = insert_col < Lmax_new
        if in_range.any():
            rr = torch.arange(rows.numel(), device=device)[in_range]
            x_tmp[rr, insert_col[in_range]] = tok[rows[in_range]]

        x_out[rows] = x_tmp

    # Rows that were not alive: just copy (as much as fits) to new padded width
    dead_rows = (~alive).nonzero(as_tuple=True)[0]
    if dead_rows.numel() > 0:
        src = torch.clamp(idx_new[dead_rows], max=Lmax - 1)
        x_out[dead_rows] = x[dead_rows].gather(1, src)

    # Finally, for each row, ensure everything beyond its new length is pad_id
    # (since x_tmp may have copied extra pads/tokens if Lmax_new > needed)
    row_mask = idx_new >= new_lengths.view(B, 1)
    x_out = x_out.masked_fill(row_mask, pad_id)

    return x_out

# ---------------------------------------------------------------------------
# ATC + G_T
# ---------------------------------------------------------------------------
def _augmented_tchebycheff(
    f_vals: torch.Tensor,
    w: torch.Tensor,
    rho: float,
    z: torch.Tensor,
) -> torch.Tensor:
    diff = f_vals - z
    term1 = torch.min(w * diff, dim=1).values
    term2 = rho * torch.sum(w * diff, dim=1)
    return term1 + term2


def _G_T(
    x: torch.Tensor,
    objective_models: List[Callable[[torch.Tensor], Tuple[str, Any]]],
    constraint_models: List[Callable[[torch.Tensor], torch.Tensor]],
    w: torch.Tensor,
    rho: float,
    z: torch.Tensor,
    beta: float,
    tokenizer, ws_for_invalid=False
):
    device = x.device
    seqs = [seq.replace(' ', '') for seq in tokenizer.batch_decode(x, skip_special_tokens=True)]

    constraint_results = []
    for constraint in constraint_models:
        res = constraint(seqs)
        constraint_results.append(res)
    
    # if ws_for_invalid is False:
    #     pdb.set_trace()
    constraint_results = torch.tensor(constraint_results, device=device)

    survived_seq_indices = (constraint_results == 1).all(dim=0).nonzero(as_tuple=True)[0]
    survived_seqs = [seqs[idx] for idx in survived_seq_indices.tolist()] # (B')

    weighted_sum_full = torch.full((len(seqs),), float("-inf"), device=device)
    G_full = torch.zeros((len(seqs),), device=device)

    # objectives
    if ws_for_invalid:
        f_vals = extract_objective_vector(seqs, objective_models, x.device)
        weighted_sum_full = torch.sum(w * f_vals, dim=1) 
        u_atc = _augmented_tchebycheff(f_vals, w, rho, z)
        G = torch.exp(beta * u_atc)
        G_full[survived_seq_indices] = G[survived_seq_indices]
    else:
        if survived_seq_indices.numel() > 0:
            f_vals = extract_objective_vector(survived_seqs, objective_models, x.device)    # (B', m)
            u_atc = _augmented_tchebycheff(f_vals, w, rho, z)   # (B',)
            G = torch.exp(beta * u_atc) # (B',)
            weighted_sum = torch.sum(w * f_vals, dim=1)  # (B',)
            G_full[survived_seq_indices] = G
            weighted_sum_full[survived_seq_indices] = weighted_sum

    # return full-size tensors (B,)
    return G_full, weighted_sum_full



# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
@torch.no_grad()
def short_rollout_batch(
    model,
    x0: torch.Tensor,            # (B, Lmax) padded
    time_grid: torch.Tensor,
    start_idx: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: Optional[torch.Tensor],
    max_len_cap: Optional[int],
    num_rollouts: int = 1,
) -> torch.Tensor:
    """
    Returns:
      xT: (B*num_rollouts, Lmax)
    Grouping:
      xT[i*num_rollouts:(i+1)*num_rollouts] corresponds to candidate i.
    """
    device = x0.device
    B, Lmax = x0.shape

    # repeat each candidate num_rollouts times (grouped)
    x = x0.repeat_interleave(num_rollouts, dim=0)  # (B*num_rollouts, Lmax)

    # rollout in batch
    for j in range(start_idx + 1, time_grid.numel()):
        t_j = time_grid[j].view(1).to(device)

        mask = (x != pad_id) 

        lam_ins, logits_ins, lam_del, lam_sub, logits_sub, *_ = model(x_t=x, mask=mask, t=t_j)

        x = sample_single_edit_batch(
            x,
            lam_ins, logits_ins,
            lam_del, lam_sub, logits_sub,
            pad_id, bos_id, eos_id,
            allowed_tokens,
            max_len_cap=max_len_cap,
        )

    return x

# ---------------------------------------------------------------------------
# finalizer
# ---------------------------------------------------------------------------
def _finalize_from_last(
    model,
    x_last: torch.Tensor,
    time_grid: torch.Tensor,
    last_step: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: Optional[torch.Tensor],
    objective_models: List[Callable[[torch.Tensor], Tuple[str, Any]]],
    constraint_models: List[Callable[[torch.Tensor], torch.Tensor]],
    w: torch.Tensor,
    rho: float,
    ref_z: torch.Tensor,
    beta_final: float,
    max_len_cap: Optional[int] = None,
    num_final_rollouts: int = 16,
    cfg=None, tokenizer=None
) -> torch.Tensor:
    device = x_last.device

    G_last, _ = _G_T(x_last, objective_models, constraint_models, w, rho, ref_z, beta_final, tokenizer, ws_for_invalid=False)

    start_idx = min(last_step, time_grid.numel() - 2) if time_grid.numel() >= 2 else 0

    x_Ts = short_rollout_batch(model, x_last.squeeze(0), time_grid, start_idx, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_final_rollouts)
    G, _, = _G_T(x_Ts, objective_models, constraint_models, w, rho, ref_z, beta_final, tokenizer, ws_for_invalid=False)

    idx = torch.isfinite(G).nonzero(as_tuple=True)[0].tolist()
    
    if len(idx) == 0 or torch.max(G).item() < G_last.item():
        return x_last
    else:
        best_idx = torch.argmax(G).item()
        best_seq = x_Ts[best_idx].unsqueeze(0)
        return best_seq

def cope_strict(
    model,
    x0: torch.Tensor,
    *,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: Optional[torch.Tensor],
    objective_models: List[Callable[[torch.Tensor], Tuple[str, Any]]],
    constraint_models: List[Callable[[torch.Tensor], torch.Tensor]],
    w: torch.Tensor,
    rho: float,
    ref_z: torch.Tensor,
    beta_start: float = 1.0,
    beta_end: float = 3.0,
    num_steps: int = 32,
    num_candidates: int = 8,
    num_rollouts: int = 4,
    max_len_cap: Optional[int] = None,
    device: Optional[torch.device] = None,
    num_final_rollouts: int = 16,
    cfg, tokenizer
) -> torch.Tensor:
    if device is None:
        device = x0.device
    x = x0.clone().to(device)
    time_grid = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
    last_timestep = 0

    with torch.no_grad():
        for step in tqdm(range(num_steps - 1)):
            t = time_grid[step].view(1)
            frac = step / max(1, (num_steps - 1))
            beta_t = beta_start + (beta_end - beta_start) * frac

            # model forward
            mask = (x != pad_id)
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type = model(x_t=x, mask=mask, t=t)
            # pdb.set_trace()
            # candidates from model
            candidates = [x.squeeze(0)] # compute the scores of current sequence with the candidates
            base_rates = []
            for _ in range(num_candidates):
                cand_seq, base_rate = _sample_single_edit_from_outputs(
                    x,
                    lam_ins, logits_ins,
                    lam_del, lam_sub, logits_sub,
                    pad_id, bos_id, eos_id,
                    allowed_tokens,
                    max_len_cap=max_len_cap,
                )
                candidates.append(cand_seq)
                base_rates.append(base_rate)

            # pdb.set_trace()
            # We only want the survived candidates to improve the objective weights
            batch_candidates = torch.nn.utils.rnn.pad_sequence(candidates, batch_first=True, padding_value=pad_id)
            start = time.time()
            cand_G, cand_ws = _G_T(batch_candidates, objective_models, constraint_models, w, rho, ref_z, beta_t, tokenizer, ws_for_invalid=True)      
            print("Candidate Time: ", time.time() - start)

            curr_G = cand_G[0]
            curr_ws = cand_ws[0]
            cand_G = cand_G[1:]
            cand_ws = cand_ws[1:]
            batch_candidates = batch_candidates[1:]
            
            improve_idx = (cand_ws > curr_ws).nonzero(as_tuple=True)[0]
            survived_candidates = batch_candidates[improve_idx, :]
            base_rates = [base_rates[i] for i in improve_idx] # (num_survived_candidates,)
            # print([len(seq.replace(' ' ,'')) for seq in tokenizer.batch_decode(survived_candidates, skip_special_tokens=True)])
            print("Num Candidates Survived: ", len(improve_idx))
            if len(improve_idx) == 0:
                continue
            
            # Keep all the rollout terminal sequences in one batch
            start = time.time()
            x_Ts = short_rollout_batch(model, survived_candidates, time_grid, step, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_rollouts)
            print("Rollout Time: ", time.time() - start)

            # pdb.set_trace()
            # Constraints are taken into account for the terminal sequences
            start = time.time()
            G, _, = _G_T(x_Ts, objective_models, constraint_models, w, rho, ref_z, beta_t, tokenizer, ws_for_invalid=False)      
            print("Terminal Time: ", time.time() - start)
            G = G.reshape(survived_candidates.shape[0], num_rollouts)
            h_hat = torch.mean(G, dim=1)   # (num_survived_candidates,)
            idx = (torch.max(G, dim=1).values > curr_G).nonzero(as_tuple=True)[0]
            final_survived_candidates = survived_candidates[idx, :]
            if len(final_survived_candidates) == 0:
                continue

            # print([len(seq.replace(' ' ,'')) for seq in tokenizer.batch_decode(survived_candidates, skip_special_tokens=True)])

            # Doob-like transform
            h_hat = h_hat[idx]
            base_rates = torch.tensor([base_rates[i] for i in idx]).to(device)
            weights_t = (base_rates ** 0.5) * h_hat
            probs = weights_t / (weights_t.sum() + 1e-12)
            if torch.isnan(probs).any():
                pdb.set_trace()
            selected_idx = torch.multinomial(probs, 1).item()
            x = final_survived_candidates[selected_idx].unsqueeze(0)
            seq = tokenizer.batch_decode(x, skip_special_tokens=True)[0].replace(' ', '')
            print(seq)
            print("Current Length: ", len(seq))
            compute_scores_print([seq], objective_models, constraint_models, device)
            last_timestep = step

        # finalize
        x_final = _finalize_from_last(
            model,
            x,
            time_grid,
            last_timestep,
            pad_id,
            bos_id,
            eos_id,
            allowed_tokens,
            objective_models,
            constraint_models,
            w,
            rho,
            ref_z,
            beta_end,
            max_len_cap=max_len_cap,
            num_final_rollouts=num_final_rollouts,
            cfg=cfg, tokenizer=tokenizer
        )
    return x_final

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/scratch/pranamlab/tong/cope/editflows/gfp/FPredX")
    parser.add_argument("--config", type=str, default="./configs/config_test.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--max_len_cap", type=int, default=None)
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--beta_start", type=float, default=1.0)
    parser.add_argument("--beta_end", type=float, default=3.0)
    parser.add_argument("--alpha_start", type=float, default=0.8)
    parser.add_argument("--alpha_end", type=float, default=0.1)
    parser.add_argument("--num_final_rollouts", type=int, default=16)
    parser.add_argument("--objective_weights", type=float, nargs='+')
    parser.add_argument("--ref_z", type=float, nargs='+')
    parser.add_argument("--rho", type=float, default=1)
    parser.add_argument("--laser", type=float, default=488, help="Keep excitation maximum near the laser you actually have")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    editflow, source_dist, tokenizer, pad_id, bos_id, eos_id, eps_id = build_model_and_stuff(cfg, device)

    ckpt = torch.load(args.ckpt, map_location=device)
    editflow.load_state_dict(ckpt["state_dict"], strict=False)
    model = editflow.model.to(device)
    model.eval()

    x0 = tokenize_input_str(args.input, tokenizer, bos_id, eos_id, device)

    allowed_tokens = torch.tensor(
        [tok for tok in source_dist._allowed_tokens if tok not in (eps_id,) and tok not in range(24,33)],
        device=device,
        dtype=torch.long,
    )

    length = GFPLength(args.input)
    excitation = GFPExcitationPred(root_dir=args.root_dir, laser=args.laser)
    brightness = GFPBrightPred(root_dir=args.root_dir)

    objective_models = [excitation, brightness]
    num_objectives = len(objective_models)
    if not args.objective_weights:
        objective_weights = torch.tensor([1.0 / num_objectives] * num_objectives).to(device)
    else:
        objective_weights = torch.tensor(args.objective_weights).to(device)

    if not args.ref_z:
        ref_z = torch.zeros(num_objectives).to(device)
    else:
        ref_z = torch.tensor(args.ref_z).to(device)

    gfp_hard_constraint = GFP(device)
    emission_soft_constraint = GFPEmissionPred(root_dir=args.root_dir)
    length_soft_constraint = Length(args.input)
    constraint_models = [length_soft_constraint, gfp_hard_constraint, emission_soft_constraint]
    # pdb.set_trace()
    x_T = cope_strict(
        model=model,
        x0=x0,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        allowed_tokens=allowed_tokens,
        objective_models=objective_models,
        constraint_models=constraint_models,
        w=objective_weights,
        rho=0.5,
        ref_z=ref_z,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_steps=args.num_steps,
        num_candidates=args.num_candidates,
        num_rollouts=args.num_rollouts,
        max_len_cap=args.max_len_cap,
        num_final_rollouts=args.num_final_rollouts,
        cfg=cfg, tokenizer=tokenizer
    )
    
    out_str = tokenizer.batch_decode(x_T, skip_special_tokens=True)[0].replace(' ', '')[5:-5]
    print("----------------------------")
    print(f"Initial Sequence: {args.input}\n")
    print(f"Initial Scores:")
    compute_scores_print([args.input], objective_models, constraint_models, device)

    print(f"\nDesigned Sequence: {out_str}\n")
    print("Final scores:")
    compute_scores_print([out_str], objective_models, constraint_models, device)


if __name__ == "__main__":
    main()
