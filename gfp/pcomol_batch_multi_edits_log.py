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
@torch.no_grad()
def _sample_multiple_edits_batch(
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
    Multi-edit small-step proposal:
      - per position i: total rate λ_i = λ_ins + λ_del + λ_sub (after masking invalid ops)
      - fire with p_i = 1 - exp(-delta * λ_i)  (independently per position)
      - if fired: pick op ~ proportional to (λ_ins, λ_del, λ_sub)
      - if op is ins/sub: draw token from softmax(logits_{ins/sub}[i]) (with allowed_tokens masking)
      - apply all fired edits "simultaneously" using a left-to-right scan on the original tokens:
          del: skip token
          sub: replace token
          ins: insert *after* the token

    Returns:
      x_out: (B, Lout) padded
      base_rate: (B,)   relative proposal weight (safe vs underflow): exp(sum_fired log_ratio)
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

    # mask rates on invalid positions (match your single-edit masking rules)
    lam_ins[:] = 0.0
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

    lam_total = ins_rate + del_rate + sub_rate  # (B, Lmax)
    del_rate *= 500

    # pdb.set_trace()
    # fire prob: p = 1 - exp(-delta*lam_total)  (use expm1 for stability)
    a = (delta * lam_total).clamp_min(0.0)
    p_fire = (-torch.expm1(-a)).masked_fill(~nonpad, 0.0)  # (B, Lmax)
    fired = (torch.rand_like(p_fire) < p_fire) & (lam_total > 1e-12) & nonpad

    # op probs per fired position: proportional to rates
    rates3 = torch.stack([ins_rate, del_rate, sub_rate], dim=-1)           # (B,Lmax,3)
    denom = lam_total.unsqueeze(-1).clamp_min(1e-12)
    op_probs = rates3 / denom                                              # (B,Lmax,3)

    # sample op only where fired
    fired_flat = fired.view(-1)
    idx_fired = fired_flat.nonzero(as_tuple=True)[0]                       # (K,)
    op_idx_flat = torch.zeros((B * Lmax,), device=device, dtype=torch.long)  # default 0

    if idx_fired.numel() > 0:
        op_p = op_probs.view(-1, 3)[idx_fired]                             # (K,3)
        op_p = op_p / op_p.sum(dim=1, keepdim=True).clamp_min(1e-12)
        op_idx_flat[idx_fired] = torch.multinomial(op_p, 1).squeeze(1)     # (K,)

    op_idx = op_idx_flat.view(B, Lmax)  # 0=ins,1=del,2=sub

    ins_mask = fired & (op_idx == 0)
    del_mask = fired & (op_idx == 1)
    sub_mask = fired & (op_idx == 2)

    # helper: mask logits to allowed_tokens
    def _mask_logits_full(logits_2d: torch.Tensor) -> torch.Tensor:
        # logits_2d: (K, V)
        if allowed_tokens is None:
            return logits_2d
        add = torch.full_like(logits_2d, -1e9)
        add[:, allowed_tokens] = 0.0
        return logits_2d + add

    # sample tokens for ins/sub at masked positions
    ins_tok = torch.full((B, Lmax), pad_id, device=device, dtype=torch.long)
    sub_tok = torch.full((B, Lmax), pad_id, device=device, dtype=torch.long)

    if ins_mask.any():
        idx_ins = ins_mask.view(-1).nonzero(as_tuple=True)[0]
        logits_sel = logits_ins.view(-1, V)[idx_ins]
        logits_sel = _mask_logits_full(logits_sel)
        q = F.softmax(logits_sel, dim=-1)
        samp = torch.multinomial(q, 1).squeeze(1)
        ins_tok.view(-1)[idx_ins] = samp

    if sub_mask.any():
        idx_sub = sub_mask.view(-1).nonzero(as_tuple=True)[0]
        logits_sel = logits_sub.view(-1, V)[idx_sub]
        logits_sel = _mask_logits_full(logits_sel)
        q = F.softmax(logits_sel, dim=-1)
        samp = torch.multinomial(q, 1).squeeze(1)
        sub_tok.view(-1)[idx_sub] = samp

    # -------------------------
    # base_rate: (B,) relative weight to avoid underflow
    # For each fired position:
    #   ratio = ((1-exp(-a)) / exp(-a)) * P(op | fired) * P(token | op)
    #         = (exp(a)-1) * (rate/total) * token_prob
    # log_ratio = log(expm1(a)) + log(op_prob) + log(token_prob)
    # -------------------------
    base_log = torch.zeros((B,), device=device, dtype=torch.float32)

    if idx_fired.numel() > 0:
        b_idx = (idx_fired // Lmax).to(torch.long)                # (K,)
        op_choice = op_idx_flat[idx_fired].to(torch.long)         # (K,)

        a_sel = a.view(-1)[idx_fired].to(torch.float32)           # (K,)
        log_expm1 = torch.log(torch.expm1(a_sel).clamp_min(eps))   # (K,)

        op_p_sel = op_probs.view(-1, 3)[idx_fired].to(torch.float32)
        op_p_sel = op_p_sel / op_p_sel.sum(dim=1, keepdim=True).clamp_min(1e-12)
        op_prob_sel = op_p_sel.gather(1, op_choice.view(-1, 1)).squeeze(1).clamp_min(eps)
        log_op = torch.log(op_prob_sel)

        log_tok = torch.zeros_like(log_op)

        # token prob for ins
        ins_k = (op_choice == 0)
        if ins_k.any():
            idx_ins_k = idx_fired[ins_k]
            tok_sel = ins_tok.view(-1)[idx_ins_k]
            logits_sel = logits_ins.view(-1, V)[idx_ins_k]
            logits_sel = _mask_logits_full(logits_sel)
            logq = F.log_softmax(logits_sel, dim=-1)
            log_tok[ins_k] = logq.gather(1, tok_sel.view(-1, 1)).squeeze(1)

        # token prob for sub
        sub_k = (op_choice == 2)
        if sub_k.any():
            idx_sub_k = idx_fired[sub_k]
            tok_sel = sub_tok.view(-1)[idx_sub_k]
            logits_sel = logits_sub.view(-1, V)[idx_sub_k]
            logits_sel = _mask_logits_full(logits_sel)
            logq = F.log_softmax(logits_sel, dim=-1)
            log_tok[sub_k] = logq.gather(1, tok_sel.view(-1, 1)).squeeze(1)

        log_ratio = log_expm1 + log_op + log_tok
        base_log.scatter_add_(0, b_idx, log_ratio)

    base_rate = torch.exp(base_log).clamp_min(0.0)  # (B,)

    # -------------------------
    # apply edits to build new padded batch
    # -------------------------
    new_seqs = []
    new_lens = []

    for b in range(B):
        seq = x[b]
        valid = (seq != pad_id)
        tokens = seq[valid].tolist()
        Lb = len(tokens)

        if Lb == 0:
            out_tokens = [eos_id]
        else:
            out_tokens = []
            for i in range(Lb):
                t_i = tokens[i]

                if i < Lmax and bool(del_mask[b, i].item()):
                    continue

                if i < Lmax and bool(sub_mask[b, i].item()):
                    out_tokens.append(int(sub_tok[b, i].item()))
                else:
                    out_tokens.append(int(t_i))

                if i < Lmax and bool(ins_mask[b, i].item()):
                    out_tokens.append(int(ins_tok[b, i].item()))

            if len(out_tokens) == 0 or out_tokens[-1] != eos_id:
                out_tokens.append(eos_id)

        if max_len_cap is not None and len(out_tokens) > max_len_cap:
            out_tokens = out_tokens[:max_len_cap]
            if out_tokens[-1] != eos_id:
                out_tokens[-1] = eos_id

        new_seqs.append(torch.tensor(out_tokens, device=device, dtype=torch.long))
        new_lens.append(len(out_tokens))

    Lout = max(1, max(new_lens) if new_lens else 1)
    x_out = torch.full((B, Lout), pad_id, device=device, dtype=x.dtype)
    for b, s in enumerate(new_seqs):
        x_out[b, : s.numel()] = s

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
    G_full = torch.full((len(seqs),), float("-inf"), device=device)

    # objectives
    if ws_for_invalid:
        f_vals = extract_objective_vector(seqs, objective_models, x.device)
        weighted_sum_full = torch.sum(w * f_vals, dim=1) 
        u_atc = _augmented_tchebycheff(f_vals, w, rho, z)
        G = beta * u_atc
        G_full[survived_seq_indices] = G[survived_seq_indices]
    else:
        if survived_seq_indices.numel() > 0:
            f_vals = extract_objective_vector(survived_seqs, objective_models, x.device)    # (B', m)
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

        x, _ = _sample_multiple_edits_batch(
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
    num_final_rollouts: int = 16,
    num_steps: int = 32,
    cfg=None, tokenizer=None
) -> torch.Tensor:
    device = x_last.device

    G_last, _ = _G_T(x_last, objective_models, constraint_models, w, rho, ref_z, beta_final, tokenizer, ws_for_invalid=False)

    start_idx = min(last_step, time_grid.numel() - 2) if time_grid.numel() >= 2 else 0

    x_Ts = short_rollout_batch(model, x_last, time_grid, start_idx, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_final_rollouts, num_steps)
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
            # curr_G, curr_ws = _G_T(x, objective_models, constraint_models, w, rho, ref_z, beta_t, tokenizer, ws_for_invalid=True)      

            # pdb.set_trace()
            # candidates from model
            # batch_candidates, base_rates = _sample_multiple_edits_batch(
            #     x.repeat(num_candidates, 1),
            #     lam_ins.repeat(num_candidates, 1), 
            #     logits_ins.repeat(num_candidates, 1, 1),
            #     lam_del.repeat(num_candidates, 1), 
            #     lam_sub.repeat(num_candidates, 1), 
            #     logits_sub.repeat(num_candidates, 1, 1),
            #     pad_id, bos_id, eos_id,
            #     allowed_tokens,
            #     delta=float(1 / num_steps),
            #     max_len_cap=max_len_cap,
            # )

            candidates = [x.squeeze(0)] # compute the scores of current sequence with the candidates
            base_rates = []
            for _ in range(num_candidates):
                cand_seq, base_rate = _sample_multiple_edits_batch(
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
            print("Initial Candidates: ", len(candidates))
            # pdb.set_trace()
            # We only want the survived candidates to improve the objective weights
            start = time.time()
            cand_G, cand_ws = _G_T(batch_candidates, objective_models, constraint_models, w, rho, ref_z, beta_t, tokenizer, ws_for_invalid=True)      
            print("Candidate Time: ", time.time() - start)

            curr_G = cand_G[0]
            curr_ws = cand_ws[0]
            cand_G = cand_G[1:]
            cand_ws = cand_ws[1:]
            batch_candidates = batch_candidates[1:, :]

            if len(batch_candidates) == 0:
                continue

            improve_idx = (cand_ws > curr_ws).nonzero(as_tuple=True)[0]
            survived_candidates = batch_candidates[improve_idx, :]
            base_rates = [base_rates[i] for i in improve_idx] # (num_survived_candidates,)
            # print([len(seq.replace(' ' ,'')) for seq in tokenizer.batch_decode(survived_candidates, skip_special_tokens=True)])
            print("Num Candidates Survived: ", len(improve_idx))
            if len(improve_idx) == 0:
                continue
            
            # Keep all the rollout terminal sequences in one batch
            start = time.time()
            x_Ts = short_rollout_batch(model, survived_candidates, time_grid, step, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_rollouts, num_steps)
            print("Rollout Time: ", time.time() - start)

            # pdb.set_trace()
            # Constraints are taken into account for the terminal sequences
            start = time.time()
            logG, _, = _G_T(x_Ts, objective_models, constraint_models, w, rho, ref_z, beta_t, tokenizer, ws_for_invalid=False)      
            print("Terminal Time: ", time.time() - start)
            logG = logG.reshape(survived_candidates.shape[0], num_rollouts)
            log_h_hat = torch.logsumexp(logG, dim=1) - math.log(num_rollouts)   # (num_survived_candidates,)
            idx = (logG.max(dim=1).values > curr_G).nonzero(as_tuple=True)[0]
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
            num_steps=num_steps,
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
    
    out_str = tokenizer.batch_decode(x_T, skip_special_tokens=True)[0].replace(' ', '')
    print("----------------------------")
    print(f"Initial Sequence: {args.input}\n")
    print(f"Initial Scores:")
    compute_scores_print([args.input], objective_models, constraint_models, device)

    print(f"\nDesigned Sequence: {out_str}\n")
    print("Final scores:")
    compute_scores_print([out_str], objective_models, constraint_models, device)


if __name__ == "__main__":
    main()
