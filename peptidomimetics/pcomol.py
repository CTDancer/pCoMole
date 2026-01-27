import argparse
from typing import List, Callable, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm
import time
import math

from generate import build_model_and_stuff, tokenize_input_str, detokenize_output
from objectives import *
from constraints import *

import sys
sys.path.append('/scratch/pranamlab/tong/pCoMol/')
from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
import pdb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging
logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def mix_score(ratio, base_scores, admet_scores):
    return [r * b + (1.0 - r) * a for r, b, a in zip(ratio, base_scores, admet_scores)]

def extract_objective_vector(smiles_tokens, smiles_seqs, objective_models, device):
    sm_scores = None
    sm_ratio = None
    values = []

    for obj in objective_models:
        name, scores = obj(smiles_tokens, smiles_seqs) # list of length B
        if name == "small_molecule":
            sm_scores = scores
            sm_ratio = sm_scores["ratio"]
            continue

        if name == "motif" and isinstance(scores, tuple):
            values.append(torch.tensor(scores[0], device=device, dtype=torch.float32))
            values.append(torch.tensor(scores[1], device=device, dtype=torch.float32))
            continue

        if name in ["non_toxicity", "solubility", "permeability", "halflife", "affinity"] and sm_scores is not None:
            scores = mix_score(sm_ratio, scores, sm_scores[name])

        values.append(torch.tensor(scores, device=device, dtype=torch.float32))

    return torch.stack(values, dim=1)  # (B,m)


def compute_scores_print(smiles_tokens, smiles_seqs, objective_models, constraint_models, device, return_scores=False):
    objective_scores = extract_objective_vector(smiles_tokens, smiles_seqs, objective_models, device) # (B,m)
    constraint_scores = []
    for constraint in constraint_models:
        constraint_score = constraint(smiles_tokens, smiles_seqs)   # list of length B
        constraint_scores.append(torch.tensor(constraint_score, device=device, dtype=torch.float32))
    constraint_scores = torch.stack(constraint_scores, dim=1) # (B, n)
    # pdb.set_trace()
    scores = torch.concat([objective_scores, constraint_scores], dim=1) # (B, m+n)
    print(scores)

    ratio = analyze_peptide_likeness(smiles_seqs[0])['peptide_likeness']
    print("Peptide Likeliness: ", ratio)

    if return_scores:
        return scores


# ---------------------------------------------------------------------------
# edit utilities
# ---------------------------------------------------------------------------
@torch.no_grad()
def _sample_single_edit_batch(
    x: torch.Tensor,                 # (B, Lmax) padded
    lam_ins: torch.Tensor,           # (B, Lmax)
    logits_ins: torch.Tensor,        # (B, Lmax, V)
    lam_del: torch.Tensor,           # (B, Lmax)
    lam_sub: torch.Tensor,           # (B, Lmax)
    logits_sub: torch.Tensor,        # (B, Lmax, V)
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: Optional[torch.Tensor] = None,  # 1D LongTensor of vocab ids
    delta: float = 1.0,
    max_len_cap: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Single-edit proposal (batched):
      - Mask invalid positions/ops (same as multi-edit version)
      - Choose ONE position i (per sequence) proportional to p_fire_i = 1 - exp(-delta * λ_i)
        (fallback to λ_i if all p_fire are ~0 but λ has mass)
      - At that position: choose op ~ proportional to (λ_ins, λ_del, λ_sub)
      - If ins/sub: sample token from softmax(logits_{ins/sub}[i]) (with allowed_tokens mask)
      - Apply exactly that one edit.

    Returns:
      x_out: (B, Lout) padded
      base_rate: (B,) relative proposal weight (consistent with your multi-edit log_ratio per fired event)
                base_rate[b] = expm1(delta * λ_i) * P(op|i) * P(tok|op,i)   (tok prob = 1 for delete)
    """
    assert x.dim() == 2, f"x must be (B,Lmax), got {tuple(x.shape)}"
    device = x.device
    B, Lmax = x.shape
    V = logits_ins.shape[-1]
    eps = 1e-30

    if allowed_tokens is not None:
        if not torch.is_tensor(allowed_tokens):
            allowed_tokens = torch.tensor(allowed_tokens, device=device, dtype=torch.long)
        else:
            allowed_tokens = allowed_tokens.to(device=device, dtype=torch.long)

    # masks
    nonpad = (x != pad_id)
    lengths = nonpad.sum(dim=1)  # (B,)
    is_bos = (x == bos_id)
    is_eos = (x == eos_id)

    # mask rates on invalid positions
    ins_rate = lam_ins.clone()
    ins_rate = ins_rate.masked_fill(~nonpad, 0.0)
    ins_rate = ins_rate.masked_fill(is_eos, 0.0)              # no insertion at eos

    del_rate = lam_del.clone()
    del_rate = del_rate.masked_fill(~nonpad, 0.0)
    del_rate = del_rate.masked_fill(is_bos | is_eos, 0.0)      # no delete bos/eos

    sub_rate = lam_sub.clone()
    sub_rate = sub_rate.masked_fill(~nonpad, 0.0)
    sub_rate = sub_rate.masked_fill(is_bos | is_eos, 0.0)      # no sub bos/eos

    # if at cap, disallow insertions
    if max_len_cap is not None:
        at_cap = lengths >= max_len_cap
        if at_cap.any():
            ins_rate = ins_rate.masked_fill(at_cap.unsqueeze(1), 0.0)

    lam_pos = ins_rate + del_rate + sub_rate  # (B, Lmax)

    # p_fire per position (masked)
    a = (delta * lam_pos).clamp_min(0.0)
    p_fire = (-torch.expm1(-a)).masked_fill(~nonpad, 0.0)  # (B, Lmax)

    # choose one position per sequence
    pos_mass = p_fire.clone()
    sum_mass = pos_mass.sum(dim=1, keepdim=True)  # (B,1)

    # fallback if all p_fire are ~0 but lam_pos has mass (common when delta small)
    fallback = (sum_mass.squeeze(1) <= 1e-12) & (lam_pos.sum(dim=1) > 1e-12)
    if fallback.any():
        pos_mass[fallback] = lam_pos[fallback]
        sum_mass[fallback] = pos_mass[fallback].sum(dim=1, keepdim=True)

    # sequences with literally no valid edits
    no_edit = (sum_mass.squeeze(1) <= 1e-12)
    # make multinomial happy (won't be used because we'll treat no_edit separately)
    if no_edit.any():
        pos_mass[no_edit, 0] = 1.0
        sum_mass[no_edit] = pos_mass[no_edit].sum(dim=1, keepdim=True)

    pos_probs = pos_mass / sum_mass.clamp_min(1e-12)
    pos_idx = torch.multinomial(pos_probs, 1).squeeze(1)  # (B,)

    # gather rates at chosen positions
    b_arange = torch.arange(B, device=device)
    ins_at = ins_rate[b_arange, pos_idx]  # (B,)
    del_at = del_rate[b_arange, pos_idx]
    sub_at = sub_rate[b_arange, pos_idx]
    lam_at = lam_pos[b_arange, pos_idx].clamp_min(1e-12)  # (B,)

    rates3_at = torch.stack([ins_at, del_at, sub_at], dim=1)  # (B,3)
    op_probs_at = rates3_at / rates3_at.sum(dim=1, keepdim=True).clamp_min(1e-12)

    # sample op for edit rows only
    op_choice = torch.zeros((B,), device=device, dtype=torch.long)  # 0=ins,1=del,2=sub
    edit_mask = ~no_edit
    if edit_mask.any():
        op_choice[edit_mask] = torch.multinomial(op_probs_at[edit_mask], 1).squeeze(1)

    ins_mask = edit_mask & (op_choice == 0)
    del_mask = edit_mask & (op_choice == 1)
    sub_mask = edit_mask & (op_choice == 2)

    def _mask_logits_2d(logits_2d: torch.Tensor) -> torch.Tensor:
        # logits_2d: (K, V)
        if allowed_tokens is None:
            return logits_2d
        add = torch.full_like(logits_2d, -1e9)
        add[:, allowed_tokens] = 0.0
        return logits_2d + add

    # sample tokens for ins/sub
    ins_tok = torch.full((B,), pad_id, device=device, dtype=torch.long)
    sub_tok = torch.full((B,), pad_id, device=device, dtype=torch.long)

    log_tok = torch.zeros((B,), device=device, dtype=torch.float32)

    if ins_mask.any():
        idx = ins_mask.nonzero(as_tuple=True)[0]
        logits_sel = logits_ins[idx, pos_idx[idx], :]  # (K,V)
        logits_sel = _mask_logits_2d(logits_sel)
        q = F.softmax(logits_sel, dim=-1)
        samp = torch.multinomial(q, 1).squeeze(1)
        ins_tok[idx] = samp
        logq = F.log_softmax(logits_sel, dim=-1)
        log_tok[idx] = logq.gather(1, samp.view(-1, 1)).squeeze(1)

    if sub_mask.any():
        idx = sub_mask.nonzero(as_tuple=True)[0]
        logits_sel = logits_sub[idx, pos_idx[idx], :]  # (K,V)
        logits_sel = _mask_logits_2d(logits_sel)
        q = F.softmax(logits_sel, dim=-1)
        samp = torch.multinomial(q, 1).squeeze(1)
        sub_tok[idx] = samp
        logq = F.log_softmax(logits_sel, dim=-1)
        log_tok[idx] = logq.gather(1, samp.view(-1, 1)).squeeze(1)

    # base_rate: expm1(delta * λ_i) * P(op|i) * P(tok|op,i)
    base_log = torch.zeros((B,), device=device, dtype=torch.float32)
    if edit_mask.any():
        a_sel = a[b_arange, pos_idx].to(torch.float32)  # (B,)
        log_expm1 = torch.log(torch.expm1(a_sel).clamp_min(eps))  # (B,)

        op_prob_sel = op_probs_at.gather(1, op_choice.view(-1, 1)).squeeze(1).clamp_min(eps)
        log_op = torch.log(op_prob_sel)

        # delete has token_prob = 1, so log_tok already 0 there
        base_log = log_expm1 + log_op + log_tok

        # for no_edit rows, set base_rate=1 (won't be used if cand==x)
        base_log = torch.where(no_edit, torch.zeros_like(base_log), base_log)

    base_rate = torch.exp(base_log).clamp_min(0.0)  # (B,)

    # apply single edit
    new_seqs = []
    new_lens = []
    for b in range(B):
        seq = x[b]
        tokens = seq[seq != pad_id].tolist()
        if len(tokens) == 0 or no_edit[b].item():
            out_tokens = tokens if len(tokens) > 0 else [eos_id]
        else:
            i = int(pos_idx[b].item())
            i = min(i, len(tokens) - 1)

            if del_mask[b].item():
                # delete token at i (already masked so not BOS/EOS)
                out_tokens = tokens[:i] + tokens[i+1:]
            elif sub_mask[b].item():
                out_tokens = tokens[:]
                out_tokens[i] = int(sub_tok[b].item())
            else:  # insertion
                out_tokens = tokens[:]
                out_tokens.insert(i + 1, int(ins_tok[b].item()))

            if len(out_tokens) == 0 or out_tokens[-1] != eos_id:
                out_tokens.append(eos_id)

        if max_len_cap is not None and len(out_tokens) > max_len_cap:
            out_tokens = out_tokens[:max_len_cap]
            if out_tokens[-1] != eos_id:
                out_tokens[-1] = eos_id

        t = torch.tensor(out_tokens, device=device, dtype=torch.long)
        new_seqs.append(t)
        new_lens.append(t.numel())

    Lout = max(1, max(new_lens))
    x_out = torch.full((B, Lout), pad_id, device=device, dtype=x.dtype)
    for b, s in enumerate(new_seqs):
        x_out[b, :s.numel()] = s

    return x_out, base_rate


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
    selfies_tokens: torch.Tensor,
    objective_models: List[Callable[[torch.Tensor], Tuple[str, Any]]],
    constraint_models: List[Callable[[torch.Tensor], torch.Tensor]],
    w: torch.Tensor,
    rho: float,
    z: torch.Tensor,
    beta: float,
    selfies_tokenizer, 
    smiles_tokenizer,
    ws_for_invalid=False
):
    device = selfies_tokens.device
    
    smiles_seqs = selfies_tokenizer.batch_decode(selfies_tokens.tolist(), return_smiles=True)
    smiles_tokens = smiles_tokenizer(smiles_seqs, return_tensors='pt', padding=True)['input_ids'].to(device)

    constraint_results = []
    for constraint in constraint_models:
        res = constraint(smiles_tokens, smiles_seqs)    # list of length B
        constraint_results.append(res)

    constraint_results = torch.tensor(constraint_results, device=device)
    survived_seq_indices = (constraint_results == 1).all(dim=0).nonzero(as_tuple=True)[0]
    survived_seqs = [smiles_seqs[idx] for idx in survived_seq_indices.tolist()] # (B')
    survived_tokens = smiles_tokens[survived_seq_indices.tolist()] 
    # pdb.set_trace()
    weighted_sum_full = torch.full((len(smiles_seqs),), float("-inf"), device=device)
    G_full = torch.full((len(smiles_seqs),), float("-inf"), device=device)

    # objectives
    if ws_for_invalid:
        f_vals = extract_objective_vector(smiles_tokens, smiles_seqs, objective_models, device)
        weighted_sum_full = torch.sum(w * f_vals, dim=1) 
        u_atc = _augmented_tchebycheff(f_vals, w, rho, z)
        G = beta * u_atc
        G_full[survived_seq_indices] = G[survived_seq_indices]
    else:
        if survived_seq_indices.numel() > 0:
            f_vals = extract_objective_vector(survived_tokens, survived_seqs, objective_models, device)    # (B', m)
            u_atc = _augmented_tchebycheff(f_vals, w, rho, z)   # (B',)
            G = beta * u_atc # (B',)
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
    num_steps: int =32
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

        x, _ = _sample_single_edit_batch(
            x,
            lam_ins, logits_ins,
            lam_del, lam_sub, logits_sub,
            pad_id, bos_id, eos_id,
            allowed_tokens,
            delta=float(1/(num_steps-1)),
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
    num_final_rollouts: int = 50,
    num_steps: int = 32,
    selfies_tokenizer=None,
    smiles_tokenizer=None
) -> torch.Tensor:

    logG_last, _ = _G_T(x_last, objective_models, constraint_models, w, rho, ref_z, beta_final, selfies_tokenizer, smiles_tokenizer, ws_for_invalid=False)

    # start_idx = min(last_step, time_grid.numel() - 2) if time_grid.numel() >= 2 else 0

    x_Ts = short_rollout_batch(model, x_last, time_grid, last_step, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_final_rollouts, num_steps)
    logG, _, = _G_T(x_Ts, objective_models, constraint_models, w, rho, ref_z, beta_final, selfies_tokenizer, smiles_tokenizer, ws_for_invalid=False)

    idx = torch.isfinite(logG).nonzero(as_tuple=True)[0].tolist()
    
    if len(idx) == 0 or torch.max(logG) < logG_last:
        return x_last, logG_last
    else:
        best_idx = torch.argmax(logG).item()
        best_seq = x_Ts[best_idx].unsqueeze(0)
        return best_seq, logG[best_idx]

def pCoMol(
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
    cfg, 
    selfies_tokenizer,
    smiles_tokenizer
) -> torch.Tensor:
    if device is None:
        device = x0.device
    x = x0.clone().to(device)
    time_grid = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
    last_timestep = 0

    best_terminal = None
    best_terminal_logG = float("-inf")

    with torch.no_grad():
        for step in tqdm(range(num_steps - 1)):
            t = time_grid[step].view(1)
            frac = step / max(1, (num_steps - 1))
            beta_t = beta_start + (beta_end - beta_start) * frac

            # model forward
            mask = (x != pad_id)
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type = model(x_t=x, mask=mask, t=t)

            candidates = [x.squeeze(0)] # compute the scores of current sequence with the candidates
            base_rates = []
            for _ in range(num_candidates):
                cand_seq, base_rate = _sample_single_edit_batch(
                    x,
                    lam_ins, logits_ins,
                    lam_del, lam_sub, logits_sub,
                    pad_id, bos_id, eos_id,
                    allowed_tokens,
                    delta=float(1/(num_steps-1)),
                    max_len_cap=max_len_cap,
                )
                if not torch.equal(cand_seq, x):
                    candidates.append(cand_seq.squeeze(0))
                    base_rates.append(base_rate)
            candidates = list(set(candidates))
            batch_candidates = torch.nn.utils.rnn.pad_sequence(candidates, batch_first=True, padding_value=pad_id)
            # print("Initial Candidates: ", len(candidates) - 1)
            # pdb.set_trace()
            # We only want the survived candidates to improve the objective weights
            start = time.time()
            cand_logG, cand_ws = _G_T(batch_candidates, objective_models, constraint_models, w, rho, ref_z, beta_t, selfies_tokenizer, smiles_tokenizer, ws_for_invalid=True)      
            # print("Candidate Time: ", time.time() - start)

            curr_logG = cand_logG[0]
            curr_ws = cand_ws[0]
            cand_logG = cand_logG[1:]
            cand_ws = cand_ws[1:]
            batch_candidates = batch_candidates[1:, :]

            if len(batch_candidates) == 0:
                continue

            improve_idx = (cand_ws > curr_ws).nonzero(as_tuple=True)[0]
            survived_candidates = batch_candidates[improve_idx, :]
            base_rates = [base_rates[i] for i in improve_idx] # (num_survived_candidates,)
            # print([len(seq.replace(' ' ,'')) for seq in tokenizer.batch_decode(survived_candidates, skip_special_tokens=True)])
            # print("Num Candidates Survived: ", len(improve_idx))
            if len(improve_idx) == 0:
                continue
            
            # Keep all the rollout terminal sequences in one batch
            start = time.time()
            x_Ts = short_rollout_batch(model, survived_candidates, time_grid, step, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_rollouts, num_steps)
            # print("Rollout Time: ", time.time() - start)

            # pdb.set_trace()
            # Constraints are taken into account for the terminal sequences
            start = time.time()
            logG, _, = _G_T(x_Ts, objective_models, constraint_models, w, rho, ref_z, beta_t, selfies_tokenizer, smiles_tokenizer, ws_for_invalid=False)      
            # pdb.set_trace()

            # Save the best teminal sequence
            curr_best_terminal_logG = torch.max(logG)
            if curr_best_terminal_logG != float('-inf') and best_terminal_logG <= curr_best_terminal_logG:
                best_terminal_idx = torch.argmax(logG)
                best_terminal = x_Ts[best_terminal_idx].unsqueeze(0)
                best_terminal_logG = curr_best_terminal_logG

                best_terminal_smiles_seq = selfies_tokenizer.batch_decode(best_terminal.tolist(), return_smiles=True)
                best_terminal_smiles_token = smiles_tokenizer(best_terminal_smiles_seq, return_tensors='pt')['input_ids'].to(device)
                print("\nSaved Best Terminal: ", best_terminal_smiles_seq)
                print("Saved Best Terminal Length: ", best_terminal_smiles_token.shape[1])
                print("Saved Best Terminal logG: ", best_terminal_logG)

            # print("Terminal Time: ", time.time() - start)
            logG = logG.reshape(survived_candidates.shape[0], num_rollouts)
            log_h_hat = torch.logsumexp(logG, dim=1) - math.log(num_rollouts)   # (num_survived_candidates,)
            idx = (logG.max(dim=1).values > curr_logG).nonzero(as_tuple=True)[0]
            final_survived_candidates = survived_candidates[idx, :]
            if len(final_survived_candidates) == 0:
                continue

            # print([len(seq.replace(' ' ,'')) for seq in tokenizer.batch_decode(survived_candidates, skip_special_tokens=True)])

            # Doob-like transform
            log_h_hat = log_h_hat[idx]
            base_rates_t = torch.tensor([base_rates[i] for i in idx.tolist()], device=device, dtype=torch.float32)
            log_base = 0.5 * torch.log(base_rates_t.clamp_min(1e-30))
            log_weights = log_base + log_h_hat
            probs = torch.softmax(log_weights, dim=0) 
            # if torch.isnan(probs).any():
            #     pdb.set_trace()
            selected_idx = torch.multinomial(probs, 1).item()
            x = final_survived_candidates[selected_idx].unsqueeze(0)
            smiles_seq = selfies_tokenizer.batch_decode(x.tolist(), return_smiles=True)
            smiles_token = smiles_tokenizer(smiles_seq, return_tensors='pt')['input_ids'].to(device)
            print(smiles_seq[0])
            print("Current Length: ", smiles_token.shape[1])
            compute_scores_print(smiles_token, smiles_seq, objective_models, constraint_models, device)
            last_timestep = step

        # finalize
        x_final_rollout, logG_final_rollout = _finalize_from_last(
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
            num_steps=num_steps,
            selfies_tokenizer=selfies_tokenizer,
            smiles_tokenizer=smiles_tokenizer,
        )

        if logG_final_rollout >= best_terminal_logG:
            best_terminal = x_final_rollout

    return best_terminal

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcomol_config", type=str, required=True)
    parser.add_argument("--bindevaluator_config", type=str, required=False)
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
    parser.add_argument("--num_final_rollouts", type=int, default=50)
    parser.add_argument("--objective_weights", type=float, nargs='+')
    parser.add_argument("--ref_z", type=float, nargs='+')
    parser.add_argument("--rho", type=float, default=1)
    parser.add_argument("--target", type=str, default=None)    
    parser.add_argument("--motifs", type=str, default=None)
    parser.add_argument("--specificity", action='store_true')
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.pcomol_config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    if args.bindevaluator_config:
        with open(args.bindevaluator_config, "r") as f:
            bindevaluator_config = edict(yaml.safe_load(f))

    editflow, source_dist, selfies_tokenizer, pad_id, bos_id, eos_id, eps_id = build_model_and_stuff(cfg, device)
    smiles_tokenizer = SMILES_SPE_Tokenizer(
        "/scratch/pranamlab/tong/pCoMol/smiles_tokenizer/new_vocab.txt",
        "/scratch/pranamlab/tong/pCoMol/smiles_tokenizer/new_splits.txt",
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    editflow.load_state_dict(ckpt["state_dict"], strict=False)
    model = editflow.model.to(device)
    model.eval()

    x0 = tokenize_input_str(args.input, selfies_tokenizer, bos_id, eos_id, device)

    allowed_tokens = torch.tensor(
        [tok for tok in source_dist._allowed_tokens if tok not in (eps_id,)],
        device=device,
        dtype=torch.long,
    )

    toxicity = Toxicity(device)
    solubility = Solubility(device)
    permeability = Permeability(device)
    halflife = Halflife(device, apply_log1p=True)
    if args.target:
        affinity = BindingAffinity(args.target, device)
    if args.motifs:
        motif = MotifModel(bindevaluator_config, args.target, args.motifs, device, args.specificity)
    
    sm = SmallMolecule(args.target, device)
    objective_models = [sm, toxicity, solubility, permeability, halflife, affinity, motif]

    num_objectives = len(objective_models) if args.specificity else len(objective_models) - 1
    if not args.objective_weights:
        objective_weights = torch.tensor([1.0 / num_objectives] * num_objectives).to(device)
    else:
        objective_weights = torch.tensor(args.objective_weights).to(device)

    if not args.ref_z:
        ref_z = torch.zeros(num_objectives).to(device)
    else:
        ref_z = torch.tensor(args.ref_z).to(device)

    peptidomimetic_hard_constraint = Peptidomimetic()
    length_soft_constraint = Length(args.input, cfg, smiles_tokenizer)
    no_constraint = Pass()
    constraint_models = [peptidomimetic_hard_constraint, length_soft_constraint]

    input_smiles_token = smiles_tokenizer(args.input, return_tensors='pt')['input_ids'].to(device)
    print("Initial Scores:")
    compute_scores_print(input_smiles_token, [args.input], objective_models, constraint_models, device, return_scores=False)
    
    valid = 0
    target_valid = 100  # how many successful designs you want

    attempt = 0
    max_attempts = 200  # safety cap so you don't infinite-loop if it keeps OOM'ing

    start = time.time()
    while valid < target_valid and attempt < max_attempts:
        attempt += 1
        try:
            x_T = pCoMol(
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
                cfg=cfg, 
                selfies_tokenizer=selfies_tokenizer,
                smiles_tokenizer=smiles_tokenizer
            )

            out_str = selfies_tokenizer.batch_decode(x_T.tolist())[0]
            smiles_token = smiles_tokenizer(out_str, return_tensors='pt')['input_ids'].to(device)

            print("----------------------------")
            print(f"\nDesigned Sequence: {out_str}\n")
            print("Final scores:")
            scores = compute_scores_print(
                smiles_token, [out_str],
                objective_models, constraint_models,
                device, return_scores=True
            ).squeeze(0)

            # Only count + save on success
            valid += 1
            with open(args.output_file, 'a') as f:
                f.write(f"{smiles_token.shape[1]}")
                for score in scores:
                    f.write(f",{score.item()}")
                f.write(f",{out_str}\n")

        except torch.cuda.OutOfMemoryError:
            # Don't count this round; skip to next attempt
            print("[WARN] CUDA OOM during pCoMol (or downstream). Skipping this round.")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # optional, can help in some cases
            continue

        # for _ in range(100):
        #     x_T = pCoMol(
        #         model=model,
        #         x0=x0,
        #         pad_id=pad_id,
        #         bos_id=bos_id,
        #         eos_id=eos_id,
        #         allowed_tokens=allowed_tokens,
        #         objective_models=objective_models,
        #         constraint_models=constraint_models,
        #         w=objective_weights,
        #         rho=0.5,
        #         ref_z=ref_z,
        #         beta_start=args.beta_start,
        #         beta_end=args.beta_end,
        #         num_steps=args.num_steps,
        #         num_candidates=args.num_candidates,
        #         num_rollouts=args.num_rollouts,
        #         max_len_cap=args.max_len_cap,
        #         num_final_rollouts=args.num_final_rollouts,
        #         cfg=cfg, 
        #         selfies_tokenizer=selfies_tokenizer,
        #         smiles_tokenizer=smiles_tokenizer
        #     )
            
        #     out_str = selfies_tokenizer.batch_decode(x_T.tolist())[0]
        #     smiles_token = smiles_tokenizer(out_str, return_tensors='pt')['input_ids'].to(device)
        #     print("----------------------------")
        #     # print(f"Initial Sequence: {args.input}\n")
        #     # print(f"Initial Scores:")
        #     # compute_scores_print([args.input], objective_models, constraint_models, device)

        #     print(f"\nDesigned Sequence: {out_str}\n")
        #     print("Final scores:")
        #     scores = compute_scores_print(smiles_token, [out_str], objective_models, constraint_models, device, return_scores=True).squeeze(0)

        #     with open(args.output_file, 'a') as f:
        #         f.write(f"{smiles_token.shape[1]}")
        #         for score in scores:
        #             f.write(f",{score.item()}")
        #         f.write(f",{out_str}\n")

    end = time.time() - start
    print("Total Time: ", end)
    print("Average Time Per Sequence: ", end / attempt)

if __name__ == "__main__":
    main()
