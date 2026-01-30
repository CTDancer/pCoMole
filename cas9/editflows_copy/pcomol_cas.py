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
from cope_models.objectives import Cas9Classification, DeletionCount, PAMMatching
from cope_models.constraints import Cas9DomainCompleteness, ProteinLength, TargetLength, MaxTargetLength, PAMMatchingConstraint, Cas9ScoreThreshold

# PAM/PI-domain detector + mask builder
from utils.pam_domain_detector.cas9_pam_detector_hmm import HMMSCAN, Cas9PIMasker

import pdb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging
logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def extract_objective_vector(protein_seqs, objective_models, device, return_names=False):
    """
    Extract objective vector from protein sequences.
    
    Args:
        protein_seqs: List of protein sequence strings
        objective_models: List of objective model callables
        device: torch device
        return_names: If True, also return list of objective names
    
    Returns:
        torch.Tensor of shape (B, m) where m is number of objectives
        If return_names=True, also returns list of objective names
    """
    values = []
    names = []

    for obj in objective_models:
        name, scores = obj(protein_tokens=None, protein_seqs=protein_seqs)  # list of length B
        values.append(torch.tensor(scores, device=device, dtype=torch.float32))
        if return_names:
            names.append(name)

    result = torch.stack(values, dim=1)  # (B,m)
    if return_names:
        return result, names
    return result


def compute_scores_print(protein_seqs, objective_models, constraint_models, device, return_scores=False):
    """
    Compute and print scores for protein sequences.
    
    Args:
        protein_seqs: List of protein sequence strings
        objective_models: List of objective model callables
        constraint_models: List of constraint model callables
        device: torch device
        return_scores: Whether to return scores
    
    Returns:
        scores if return_scores=True, else None
    """
    objective_scores = extract_objective_vector(protein_seqs, objective_models, device)  # (B,m)
    constraint_scores = []
    for constraint in constraint_models:
        # Handle constraints that take single sequences vs batches
        if hasattr(constraint, 'predict_batch'):
            # Use batch prediction if available (more efficient)
            constraint_score = constraint.predict_batch(protein_seqs)
            constraint_score = [int(c) for c in constraint_score]  # Convert bool to int
        else:
            # Call for each sequence individually
            constraint_score = [constraint(None, seq) for seq in protein_seqs]
        constraint_scores.append(torch.tensor(constraint_score, device=device, dtype=torch.float32))
    constraint_scores = torch.stack(constraint_scores, dim=1)  # (B, n)
    scores = torch.concat([objective_scores, constraint_scores], dim=1)  # (B, m+n)
    print(scores)
    
    # Print predicted PAM and PAM probability score if PAM matching objective is present
    for obj in objective_models:
        if hasattr(obj, 'predict_pam'):
            predicted_pams = obj.predict_pam(protein_seqs)
            # Get PAM probability scores for each predicted PAM (not the target PAM)
            # Use raw probabilities (no temperature scaling) for more interpretable results
            pam_prob_scores = []
            for i, pam in enumerate(predicted_pams):
                # Compute score for this specific predicted PAM using raw probabilities
                score = obj.get_score_for_pam([protein_seqs[i]], pam, use_temperature_scaling=False)[0]
                pam_prob_scores.append(score)
            for i, (pam, prob_score) in enumerate(zip(predicted_pams, pam_prob_scores)):
                print(f"Predicted PAM: {pam} (target: {obj.target_pam}) | Predicted PAM probability (raw): {prob_score:.8f}")
            break  # Only print once if multiple PAM objectives exist

    if return_scores:
        return scores


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
    protected_mask: Optional[torch.Tensor] = None,  # (B, Lmax) bool, True=no edits allowed (PAM-protected)
    pam_scale_edits: bool = False,  # If True, scale up edit rates in PAM region instead of masking
    pam_edit_scale_factor: float = 10.0,  # Scaling factor for ins/sub rates in PAM region
    deletion_rate_scale: float = 1500.0,  # Scaling factor for deletion rate
    debug_context: Optional[str] = None,  # Context label for debug output (e.g., "CANDIDATE", "ROLLOUT")
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict]:
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
      protected_mask_out: (B, Lout) bool or None, updated mask with same edits applied
      edit_stats: dict with keys 'num_ins', 'num_del', 'num_sub', 'total_edits' (per batch item)
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

    # Debug: print raw model outputs before any masking or modification
    context_prefix = f"[{debug_context}] " if debug_context else ""
    # Skip rollout debugging, skip extra info for candidates (only show edit stats)
    is_rollout = debug_context is not None and "ROLLOUT" in debug_context
    is_candidate = debug_context is not None and "CANDIDATE" in debug_context
    # Only print raw model outputs for non-rollout, non-candidate contexts (e.g., SELECTED)
    # if not is_rollout and not is_candidate and nonpad.any():
    #     avg_lam_del_raw = lam_del[nonpad].mean().item()
    #     avg_lam_ins_raw = lam_ins[nonpad].mean().item()
    #     avg_lam_sub_raw = lam_sub[nonpad].mean().item()
    #     if avg_lam_ins_raw > 0:
    #         ratio_raw = (avg_lam_del_raw / avg_lam_ins_raw) - 1.0  # Positive = lam_del larger, negative = lam_ins larger
    #         print(f"{context_prefix}[DEBUG] Raw model outputs: avg_lam_del={avg_lam_del_raw:.6f}, avg_lam_ins={avg_lam_ins_raw:.6f}, avg_lam_sub={avg_lam_sub_raw:.6f}, ratio={ratio_raw:.4f} (lam_del is {ratio_raw*100:.1f}% {'larger' if ratio_raw > 0 else 'smaller'})")
    #     else:
    #         print(f"{context_prefix}[DEBUG] Raw model outputs: avg_lam_del={avg_lam_del_raw:.6f}, avg_lam_ins={avg_lam_ins_raw:.6f}, avg_lam_sub={avg_lam_sub_raw:.6f}")

    # mask rates on invalid positions (match your single-edit masking rules)
    # lam_ins[:] = 0.0
    ins_rate = lam_ins.clone()
    ins_rate = ins_rate.masked_fill(~nonpad, 0.0)
    ins_rate = ins_rate.masked_fill(is_eos, 0.0)              # no insertion at eos

    del_rate = lam_del.clone()
    del_rate = del_rate.masked_fill(~nonpad, 0.0)
    del_rate = del_rate.masked_fill(is_bos | is_eos, 0.0)      # no delete bos/eos

    sub_rate = lam_sub.clone()
    sub_rate = sub_rate.masked_fill(~nonpad, 0.0)
    sub_rate = sub_rate.masked_fill(is_bos | is_eos, 0.0)      # no sub bos/eos
    
    # Only print shapes/averages during candidate generation, not during rollouts
    # if not is_rollout:
    #     print(f"Shapes -- ins_rate: {ins_rate.shape}, del_rate: {del_rate.shape}, sub_rate: {sub_rate.shape}")
    #     print(f"Averages -- ins_rate: {ins_rate.mean(dim=-1)}, del_rate: {del_rate.mean(dim=-1)}, sub_rate: {sub_rate.mean(dim=-1)}")

    # Apply protected mask: either block edits OR scale up ins/sub rates in PAM region
    if protected_mask is not None:
        if pam_scale_edits:
            # Scale up insertion and substitution rates in PAM region (but not deletion)
            ins_rate = torch.where(protected_mask, ins_rate * pam_edit_scale_factor, ins_rate)
            sub_rate = torch.where(protected_mask, sub_rate * pam_edit_scale_factor, sub_rate)
            # Deletion rate remains unchanged (not scaled)
        else:
            # Original behavior: block ALL edits (ins/del/sub) in PAM-protected regions
            ins_rate = ins_rate.masked_fill(protected_mask, 0.0)
            del_rate = del_rate.masked_fill(protected_mask, 0.0)
            sub_rate = sub_rate.masked_fill(protected_mask, 0.0)

    # if at cap, disallow insertions
    if max_len_cap is not None:
        at_cap = lengths >= max_len_cap
        if at_cap.any():
            ins_rate = ins_rate.masked_fill(at_cap.unsqueeze(1), 0.0)

    lam_total = ins_rate + del_rate + sub_rate  # (B, Lmax)

    # Debug: print average rates before amplification
    valid_mask = nonpad  # (B, Lmax)
    # Skip rollout debugging, skip extra info for candidates (only show edit stats)
    # Only print before amplification for non-rollout, non-candidate contexts (e.g., SELECTED)
    # if not is_rollout and not is_candidate and valid_mask.any():
    #     avg_lam_del_before = del_rate[valid_mask].mean().item()
    #     avg_lam_ins_before = ins_rate[valid_mask].mean().item()
    #     avg_lam_sub_before = sub_rate[valid_mask].mean().item()
    #     if avg_lam_ins_before > 0:
    #         ratio = (avg_lam_del_before / avg_lam_ins_before) - 1.0  # Positive = lam_del larger, negative = lam_ins larger
    #         print(f"{context_prefix}[DEBUG] Before amplification: avg_lam_del={avg_lam_del_before:.6f}, avg_lam_ins={avg_lam_ins_before:.6f}, avg_lam_sub={avg_lam_sub_before:.6f}, ratio={ratio:.4f}")
    #     else:
    #         print(f"{context_prefix}[DEBUG] Before amplification: avg_lam_del={avg_lam_del_before:.6f}, avg_lam_ins={avg_lam_ins_before:.6f}, avg_lam_sub={avg_lam_sub_before:.6f}")

    # note: you had this amplification; kept unchanged
    del_rate *= deletion_rate_scale
    
    # Debug: print average deletion rate after amplification
    # Skip rollout debugging, skip extra info for candidates (only show edit stats)
    # Only print after amplification for non-rollout, non-candidate contexts (e.g., SELECTED)
    # if not is_rollout and not is_candidate and valid_mask.any():
    #     avg_lam_del_after = del_rate[valid_mask].mean().item()
    #     avg_lam_ins_after = ins_rate[valid_mask].mean().item()
    #     avg_lam_sub_after = sub_rate[valid_mask].mean().item()
    #     print(f"{context_prefix}[DEBUG] After amplification (1000x): avg_lam_del={avg_lam_del_after:.6f}, avg_lam_ins={avg_lam_ins_after:.6f}, avg_lam_sub={avg_lam_sub_after:.6f}")

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

    # Count edit types for statistics
    num_ins = ins_mask.sum().item()
    num_del = del_mask.sum().item()
    num_sub = sub_mask.sum().item()
    total_edits = num_ins + num_del + num_sub

    # Compute average rates post-amplification for each batch item
    avg_rates_per_batch = []
    for b in range(B):
        valid_positions = nonpad[b]  # (Lmax,) bool
        if valid_positions.any():
            avg_ins = ins_rate[b][valid_positions].mean().item()
            avg_del = del_rate[b][valid_positions].mean().item()
            avg_sub = sub_rate[b][valid_positions].mean().item()
            avg_rates_per_batch.append((avg_ins, avg_del, avg_sub))
        else:
            avg_rates_per_batch.append((0.0, 0.0, 0.0))

    # Skip rollout debugging, but keep candidate debugging
    if not is_rollout and total_edits > 0:
        pct_ins = 100.0 * num_ins / total_edits
        pct_del = 100.0 * num_del / total_edits
        pct_sub = 100.0 * num_sub / total_edits
        # For candidates, we typically have B=1, so use first batch item
        avg_ins_rate, avg_del_rate, avg_sub_rate = avg_rates_per_batch[0] if avg_rates_per_batch else (0.0, 0.0, 0.0)
        print(f"{context_prefix}[EDIT STATS] ins={num_ins} ({pct_ins:.1f}%), del={num_del} ({pct_del:.1f}%), sub={num_sub} ({pct_sub:.1f}%) | avg_rates: ins={avg_ins_rate:.6f}, del={avg_del_rate:.6f}, sub={avg_sub_rate:.6f}")

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
    new_masks = []  # Track masks if provided

    for b in range(B):
        seq = x[b]
        valid = (seq != pad_id)
        tokens = seq[valid].tolist()
        Lb = len(tokens)
        
        # Extract mask for this batch item if provided
        mask_vals = None
        if protected_mask is not None:
            mask_vals = protected_mask[b, :Lb].tolist()  # (Lb,) bool list

        if Lb == 0:
            out_tokens = [eos_id]
            out_mask = [False] if mask_vals is not None else None
        else:
            out_tokens = []
            out_mask = [] if mask_vals is not None else None
            for i in range(Lb):
                t_i = tokens[i]
                m_i = mask_vals[i] if mask_vals is not None else None

                if i < Lmax and bool(del_mask[b, i].item()):
                    # Delete: skip token and mask value
                    continue

                if i < Lmax and bool(sub_mask[b, i].item()):
                    out_tokens.append(int(sub_tok[b, i].item()))
                else:
                    out_tokens.append(int(t_i))
                
                # Keep mask value for this position (substitution doesn't change position)
                if out_mask is not None:
                    out_mask.append(m_i)

                if i < Lmax and bool(ins_mask[b, i].item()):
                    out_tokens.append(int(ins_tok[b, i].item()))
                    # Insert: new position, not in PAM domain, so False
                    if out_mask is not None:
                        out_mask.append(False)

            if len(out_tokens) == 0 or out_tokens[-1] != eos_id:
                out_tokens.append(eos_id)
                if out_mask is not None:
                    out_mask.append(False)  # EOS can't be deleted anyway

        if max_len_cap is not None and len(out_tokens) > max_len_cap:
            out_tokens = out_tokens[:max_len_cap]
            if out_mask is not None:
                out_mask = out_mask[:max_len_cap]
            if out_tokens[-1] != eos_id:
                out_tokens[-1] = eos_id

        new_seqs.append(torch.tensor(out_tokens, device=device, dtype=torch.long))
        new_lens.append(len(out_tokens))
        if out_mask is not None:
            new_masks.append(torch.tensor(out_mask, device=device, dtype=torch.bool))

    Lout = max(1, max(new_lens) if new_lens else 1)
    x_out = torch.full((B, Lout), pad_id, device=device, dtype=x.dtype)
    for b, s in enumerate(new_seqs):
        x_out[b, : s.numel()] = s
    
    # Reconstruct mask tensor if masks were provided
    protected_mask_out = None
    if new_masks:
        protected_mask_out = torch.full((B, Lout), False, device=device, dtype=torch.bool)
        for b, m in enumerate(new_masks):
            protected_mask_out[b, : m.numel()] = m

    # Collect edit stats per batch item
    edit_stats = {
        'num_ins': [ins_mask[b].sum().item() for b in range(B)],
        'num_del': [del_mask[b].sum().item() for b in range(B)],
        'num_sub': [sub_mask[b].sum().item() for b in range(B)],
        'total_edits': [ins_mask[b].sum().item() + del_mask[b].sum().item() + sub_mask[b].sum().item() for b in range(B)],
        'avg_rates': avg_rates_per_batch  # Store average rates post-amplification
    }

    return x_out, base_rate, protected_mask_out, edit_stats


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
    protein_tokens: torch.Tensor,
    objective_models,
    constraint_models,
    w: torch.Tensor,
    rho: float,
    z: torch.Tensor,
    beta: float,
    tokenizer,
    ws_for_invalid: bool = False,
    debug_context=None,
):
    """
    Matches behavior of:
      - cope_batch_multi_edits_log_length (1).py
      - pcomol (1).py

    Key semantics:
      - Constraints are evaluated for all sequences.
      - If ws_for_invalid=True:
          * weighted_sum_full is computed for ALL sequences (valid or invalid)
          * G_full is ONLY assigned for constraint-valid sequences (invalid remain -inf)
      - If ws_for_invalid=False:
          * both weighted_sum_full and G_full are ONLY assigned for constraint-valid sequences
            (invalid remain -inf)
    """
    device = protein_tokens.device

    # Decode sequences
    protein_seqs = [
        seq.replace(" ", "")
        for seq in tokenizer.batch_decode(protein_tokens, skip_special_tokens=True)
    ]

    # -------------------------
    # constraints (evaluate on ALL)
    # -------------------------
    constraint_results = []
    for constraint in constraint_models:
        if hasattr(constraint, "predict_batch"):
            res = constraint.predict_batch(protein_seqs)
            res = [int(r) for r in res]
        else:
            res = [
                constraint(protein_tokens[i] if protein_tokens is not None else None, seq)
                for i, seq in enumerate(protein_seqs)
            ]
        constraint_results.append(res)

    constraint_results = torch.tensor(constraint_results, device=device)
    survived_seq_indices = (constraint_results == 1).all(dim=0).nonzero(as_tuple=True)[0]
    survived_seqs = [protein_seqs[idx] for idx in survived_seq_indices.tolist()]

    # outputs
    B = len(protein_seqs)
    weighted_sum_full = torch.full((B,), float("-inf"), device=device)
    G_full = torch.full((B,), float("-inf"), device=device)

    # -------------------------
    # objectives
    # -------------------------
    if ws_for_invalid:
        # Compute objective vector for ALL sequences
        f_vals = extract_objective_vector(protein_seqs, objective_models, device)  # (B, m)

        # Compute weighted sum for ALL sequences (matching pcomol.py)
        weighted_sum_full = torch.sum(w * f_vals, dim=1)  # (B,)
        
        # Compute G for ALL sequences (matching pcomol.py behavior)
        u_atc = _augmented_tchebycheff(f_vals, w, rho, z)  # (B,)
        G = beta * u_atc  # (B,)
        # Only assign valid sequences to G_full (invalid remain -inf)
        G_full[survived_seq_indices] = G[survived_seq_indices]

    else:
        # Terminal scoring mode: both ws and G only for constraint-valid sequences
        if survived_seq_indices.numel() > 0:
            f_vals = extract_objective_vector(survived_seqs, objective_models, device)  # (B', m)
            u_atc = _augmented_tchebycheff(f_vals, w, rho, z)  # (B',)
            G = beta * u_atc  # (B',)
            weighted_sum = torch.sum(w * f_vals, dim=1)  # (B',)
            G_full[survived_seq_indices] = G
            weighted_sum_full[survived_seq_indices] = weighted_sum

    return G_full, weighted_sum_full
# def _G_T(
#     protein_tokens: torch.Tensor,
#     objective_models: List[Callable[[torch.Tensor], Tuple[str, Any]]],
#     constraint_models: List[Callable[[torch.Tensor], torch.Tensor]],
#     w: torch.Tensor,
#     rho: float,
#     z: torch.Tensor,
#     beta: float,
#     tokenizer,
#     ws_for_invalid=False,
#     debug_context: Optional[str] = None
# ):
#     device = protein_tokens.device
    
#     # Decode protein sequences from tokens
#     protein_seqs = [seq.replace(' ', '') for seq in tokenizer.batch_decode(protein_tokens, skip_special_tokens=True)]

#     constraint_results = []
#     for constraint in constraint_models:
#         # Handle constraints that take single sequences vs batches
#         if hasattr(constraint, 'predict_batch'):
#             # Use batch prediction if available (more efficient)
#             res = constraint.predict_batch(protein_seqs)
#             res = [int(r) for r in res]  # Convert bool to int
#         else:
#             # Call for each sequence individually
#             res = [constraint(protein_tokens[i] if protein_tokens is not None else None, seq) 
#                    for i, seq in enumerate(protein_seqs)]
#         constraint_results.append(res)

#     constraint_results = torch.tensor(constraint_results, device=device)
#     survived_seq_indices = (constraint_results == 1).all(dim=0).nonzero(as_tuple=True)[0]
#     survived_seqs = [protein_seqs[idx] for idx in survived_seq_indices.tolist()]  # (B')

#     weighted_sum_full = torch.full((len(protein_seqs),), float("-inf"), device=device)
#     G_full = torch.full((len(protein_seqs),), float("-inf"), device=device)

#     # Get objective names and find DeletionCount objective for absolute count
#     obj_names = []
#     deletion_obj = None
#     deletion_obj_idx = None
#     if protein_seqs:
#         # Get names by calling with first sequence
#         for obj_idx, obj in enumerate(objective_models):
#             name, _ = obj(protein_tokens=None, protein_seqs=[protein_seqs[0]])
#             obj_names.append(name)
#             # Check if this is DeletionCount objective
#             if hasattr(obj, 'original_length') and hasattr(obj, 'max_deletion'):
#                 deletion_obj = obj
#                 deletion_obj_idx = obj_idx
#     else:
#         # Fallback: use generic names
#         obj_names = [f"obj_{i}" for i in range(len(objective_models))]

#     # Helper function to format objective score with absolute deletion count if applicable
#     def format_obj_score(obj_name, raw_score, obj_idx, seq):
#         if obj_name == 'deletion_count' and deletion_obj is not None:
#             current_length = len(seq.replace(' ', ''))
#             abs_deletion = deletion_obj.original_length - current_length
#             return f"{obj_name}: {raw_score:.4f} (abs: {abs_deletion})"
#         return f"{obj_name}: {raw_score:.4f}"

#     # objectives
#     if ws_for_invalid:
#         # Compute objective scores for all sequences (including invalid ones)
#         f_vals = extract_objective_vector(protein_seqs, objective_models, device)
        
#         # Compute weighted sum for all sequences
#         weighted_scores = w.unsqueeze(0) * f_vals  # (B, m) - element-wise multiplication
#         weighted_sum_all = torch.sum(weighted_scores, dim=1)  # (B,)
        
#         # Only set weighted_sum_full for valid sequences (invalid ones stay -inf)
#         weighted_sum_full[survived_seq_indices] = weighted_sum_all[survived_seq_indices]
#         # Compute G only for valid sequences
#         if survived_seq_indices.numel() > 0:
#             f_vals_valid = f_vals[survived_seq_indices]
#             u_atc = _augmented_tchebycheff(f_vals_valid, w, rho, z)
#             G = beta * u_atc
#             G_full[survived_seq_indices] = G
            
#             # DEBUG: Print logG calculation (ATC-based)
#             if debug_context is not None:
#                 seq_idx = 0  # Show first sequence only for brevity
#                 if seq_idx < len(f_vals_valid):
#                     f_seq = f_vals_valid[seq_idx]  # (m,)
#                     diff = f_seq - z  # (m,) - distance from reference point
#                     w_diff = w * diff  # (m,) - weighted differences
#                     term1 = torch.min(w_diff).item()  # min(w * diff)
#                     term2 = (rho * torch.sum(w_diff)).item()  # rho * sum(w * diff)
#                     u_atc_val = u_atc[seq_idx].item()
#                     logG_val = G[seq_idx].item()
                    
#                     # Format output
#                     diff_parts = []
#                     w_diff_parts = []
#                     for obj_idx, obj_name in enumerate(obj_names):
#                         raw_score = f_seq[obj_idx].item()
#                         diff_val = diff[obj_idx].item()
#                         w_diff_val = w_diff[obj_idx].item()
#                         weight = w[obj_idx].item()
#                         ref_val = z[obj_idx].item()
                        
#                         # Add absolute deletion count if applicable
#                         if obj_name == 'deletion_count' and deletion_obj is not None:
#                             # Get the actual sequence index in the original protein_seqs
#                             actual_seq_idx = survived_seq_indices[seq_idx].item()
#                             seq_str = protein_seqs[actual_seq_idx]
#                             abs_deletion = deletion_obj.original_length - len(seq_str.replace(' ', ''))
#                             diff_parts.append(f"{obj_name}: {raw_score:.4f} (abs: {abs_deletion}) - {ref_val:.4f} = {diff_val:.4f}")
#                         else:
#                             diff_parts.append(f"{obj_name}: {raw_score:.4f} - {ref_val:.4f} = {diff_val:.4f}")
#                         w_diff_parts.append(f"{obj_name}: {diff_val:.4f} × {weight:.3f} = {w_diff_val:.4f}")
                    
#                     print(f"[{debug_context}] logG calc: diffs=[{', '.join(diff_parts)}] | "
#                           f"w×diffs=[{', '.join(w_diff_parts)}] | "
#                           f"min={term1:.4f}, rho×sum={term2:.4f} (rho={rho:.3f}) | "
#                           f"u_atc={u_atc_val:.4f}, logG={logG_val:.4f} (β={beta:.3f})")
#     else:
#         if survived_seq_indices.numel() > 0:
#             f_vals = extract_objective_vector(survived_seqs, objective_models, device)  # (B', m)
            
#             # Compute weighted scores
#             weighted_scores = w.unsqueeze(0) * f_vals  # (B', m)
#             weighted_sum = torch.sum(weighted_scores, dim=1)  # (B',)
            
#             u_atc = _augmented_tchebycheff(f_vals, w, rho, z)  # (B',)
#             G = beta * u_atc  # (B',)
#             G_full[survived_seq_indices] = G
#             weighted_sum_full[survived_seq_indices] = weighted_sum
            
#             # DEBUG: Print logG calculation (ATC-based)
#             if debug_context is not None:
#                 seq_idx = 0  # Show first sequence only for brevity
#                 if seq_idx < len(f_vals):
#                     f_seq = f_vals[seq_idx]  # (m,)
#                     diff = f_seq - z  # (m,) - distance from reference point
#                     w_diff = w * diff  # (m,) - weighted differences
#                     term1 = torch.min(w_diff).item()  # min(w * diff)
#                     term2 = (rho * torch.sum(w_diff)).item()  # rho * sum(w * diff)
#                     u_atc_val = u_atc[seq_idx].item()
#                     logG_val = G[seq_idx].item()
                    
#                     # Format output
#                     diff_parts = []
#                     w_diff_parts = []
#                     for obj_idx, obj_name in enumerate(obj_names):
#                         raw_score = f_seq[obj_idx].item()
#                         diff_val = diff[obj_idx].item()
#                         w_diff_val = w_diff[obj_idx].item()
#                         weight = w[obj_idx].item()
#                         ref_val = z[obj_idx].item()
                        
#                         # Add absolute deletion count if applicable
#                         if obj_name == 'deletion_count' and deletion_obj is not None:
#                             seq_str = survived_seqs[seq_idx]
#                             abs_deletion = deletion_obj.original_length - len(seq_str.replace(' ', ''))
#                             diff_parts.append(f"{obj_name}: {raw_score:.4f} (abs: {abs_deletion}) - {ref_val:.4f} = {diff_val:.4f}")
#                         else:
#                             diff_parts.append(f"{obj_name}: {raw_score:.4f} - {ref_val:.4f} = {diff_val:.4f}")
#                         w_diff_parts.append(f"{obj_name}: {diff_val:.4f} × {weight:.3f} = {w_diff_val:.4f}")
                    
#                     print(f"[{debug_context}] logG calc: diffs=[{', '.join(diff_parts)}] | "
#                           f"w×diffs=[{', '.join(w_diff_parts)}] | "
#                           f"min={term1:.4f}, rho×sum={term2:.4f} (rho={rho:.3f}) | "
#                           f"u_atc={u_atc_val:.4f}, logG={logG_val:.4f} (β={beta:.3f})")

#     # return full-size tensors (B,)
#     return G_full, weighted_sum_full



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
    num_steps: int =32,
    protected_mask: Optional[torch.Tensor] = None,  # (B, Lmax) bool, True=no edits allowed
    deletion_rate_scale: float = 1500.0,  # Scaling factor for deletion rate
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

    # repeat protected_mask if provided
    if protected_mask is not None:
        protected_mask_repeated = protected_mask.repeat_interleave(num_rollouts, dim=0)  # (B*num_rollouts, Lmax)
    else:
        protected_mask_repeated = None

    # rollout in batch
    for j in range(start_idx + 1, time_grid.numel()):
        t_j = time_grid[j].view(1).to(device)

        mask = (x != pad_id) 

        lam_ins, logits_ins, lam_del, lam_sub, logits_sub, *_ = model(x_t=x, mask=mask, t=t_j)

        x, _, protected_mask_repeated, _ = _sample_multiple_edits_batch(
            x,
            lam_ins, logits_ins,
            lam_del, lam_sub, logits_sub,
            pad_id, bos_id, eos_id,
            allowed_tokens,
            delta=float(1/(num_steps-1)),
            max_len_cap=max_len_cap,
            protected_mask=protected_mask_repeated,
            pam_scale_edits=False,  # Rollouts use masking mode (scale_edits only applies to candidate generation)
            pam_edit_scale_factor=10.0,  # Not used in rollouts
            deletion_rate_scale=deletion_rate_scale,
            debug_context=f"ROLLOUT t={j}/{time_grid.numel()-1}",
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
    tokenizer=None,
    # NEW: recompute PI mask for finalization step
    pam_masker: Optional[Cas9PIMasker] = None,
    deletion_rate_scale: float = 1500.0,  # Scaling factor for deletion rate
) -> torch.Tensor:

    logG_last, _ = _G_T(x_last, objective_models, constraint_models, w, rho, ref_z, beta_final, tokenizer, ws_for_invalid=False, debug_context="FINALIZATION_INITIAL")

    # start_idx = min(last_step, time_grid.numel() - 2) if time_grid.numel() >= 2 else 0

    # Recompute PI mask for finalization step (protects against all edits)
    protected_mask = None
    if pam_masker is not None:
        seq_last = tokenizer.batch_decode(x_last, skip_special_tokens=True)[0].replace(" ", "")
        protected_mask = pam_masker.build_no_del_mask(x_last, seq_last, pad_id=pad_id, bos_at_index0=True)

    x_Ts = short_rollout_batch(model, x_last, time_grid, last_step, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_final_rollouts, num_steps, protected_mask=protected_mask, deletion_rate_scale=deletion_rate_scale)
    logG, _ = _G_T(x_Ts, objective_models, constraint_models, w, rho, ref_z, beta_final, tokenizer, ws_for_invalid=False, debug_context="FINALIZATION_ROLLOUTS")

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
    tokenizer,
    # PAM masking parameters
    pam_masker: Optional[Cas9PIMasker] = None,
    pam_mask_refresh_every: int = 1,
    pam_debug: bool = False,
    pam_scale_edits: bool = False,  # If True, scale up edit rates in PAM region instead of masking
    pam_edit_scale_factor: float = 10.0,  # Scaling factor for ins/sub rates in PAM region
    deletion_rate_scale: float = 1500.0,  # Scaling factor for deletion rate
) -> torch.Tensor:
    if device is None:
        device = x0.device
    x = x0.clone().to(device)
    time_grid = torch.linspace(0.0, 1.0, steps=num_steps, device=device)
    last_timestep = 0

    best_terminal = None
    best_terminal_logG = float("-inf")
    
    # Track cumulative edit statistics for selected steps only
    total_ins = 0
    total_del = 0
    total_sub = 0
    
    protected_mask = None  # (1, Lmax) bool, True=no edits allowed in PAM-protected regions

    def _refresh_protected_mask(curr_x: torch.Tensor, step_num: int) -> Optional[torch.Tensor]:
        if pam_masker is None:
            return None
        seq = tokenizer.batch_decode(curr_x, skip_special_tokens=True)[0].replace(" ", "")
        m = pam_masker.build_no_del_mask(curr_x, seq, pad_id=pad_id, bos_at_index0=True)
        if pam_debug:
            # Get masked interval for debug output
            interval = pam_masker.pi_core_interval(seq)
            masked_count = int(m.sum().item())
            seq_len = len(seq)
            if interval is not None:
                s, t = interval  # 1-based AA positions
                mode_str = f"edit rates scaled by {pam_edit_scale_factor}x" if pam_scale_edits else "all edits blocked"
                print(f"[PAM mask] Step {step_num}: protected_range=[{s}-{t}] (1-based AA), protected_positions={masked_count}, seq_len={seq_len} ({mode_str})")
            else:
                print(f"[PAM mask] Step {step_num}: no PI hit found, protected_positions={masked_count}, seq_len={seq_len}")
        return m

    with torch.no_grad():
        for step in tqdm(range(num_steps - 1)):
            t = time_grid[step].view(1)
            frac = step / max(1, (num_steps - 1))
            beta_t = beta_start + (beta_end - beta_start) * frac

            # DEBUG: Print current sequence logG calculation at start of step (compact)
            if step == 0 or step % max(1, num_steps // 5) == 0:  # Print at start and every ~20% of steps
                curr_seq_str = tokenizer.batch_decode(x, skip_special_tokens=True)[0].replace(" ", "")
                curr_obj_scores, obj_names = extract_objective_vector([curr_seq_str], objective_models, device, return_names=True)
                curr_obj_scores = curr_obj_scores.squeeze(0)
                
                # Find DeletionCount objective for absolute count
                deletion_obj = None
                for obj in objective_models:
                    if hasattr(obj, 'original_length') and hasattr(obj, 'max_deletion'):
                        deletion_obj = obj
                        break
                
                # Compute logG components
                diff = curr_obj_scores - ref_z
                w_diff = w * diff
                term1 = torch.min(w_diff).item()
                term2 = (rho * torch.sum(w_diff)).item()
                u_atc_val = term1 + term2
                logG_val = beta_t * u_atc_val
                
                # Format output
                diff_parts = []
                w_diff_parts = []
                for obj_idx, obj_name in enumerate(obj_names):
                    raw_score = curr_obj_scores[obj_idx].item()
                    diff_val = diff[obj_idx].item()
                    w_diff_val = w_diff[obj_idx].item()
                    weight = w[obj_idx].item()
                    ref_val = ref_z[obj_idx].item()
                    
                    if obj_name == 'deletion_count' and deletion_obj is not None:
                        abs_deletion = deletion_obj.original_length - len(curr_seq_str)
                        diff_parts.append(f"{obj_name}: {raw_score:.4f} (abs: {abs_deletion}) - {ref_val:.4f} = {diff_val:.4f}")
                    else:
                        diff_parts.append(f"{obj_name}: {raw_score:.4f} - {ref_val:.4f} = {diff_val:.4f}")
                    w_diff_parts.append(f"{obj_name}: {diff_val:.4f} × {weight:.3f} = {w_diff_val:.4f}")
                
                print(f"[STEP {step} START] len={len(curr_seq_str)} | diffs=[{', '.join(diff_parts)}] | "
                      f"w×diffs=[{', '.join(w_diff_parts)}] | min={term1:.4f}, rho×sum={term2:.4f} | "
                      f"u_atc={u_atc_val:.4f}, logG={logG_val:.4f} (β={beta_t:.3f})")

            # Refresh PI protection mask for current accepted sequence (blocks all edits)
            if pam_masker is not None and (step % max(1, pam_mask_refresh_every) == 0):
                protected_mask = _refresh_protected_mask(x, step)

            # model forward
            mask = (x != pad_id)
            # ReparameterizedProteinEditFlowModel returns 8 values, ProteinEditFlowModel returns 5
            model_output = model(x_t=x, mask=mask, t=t)
            if len(model_output) == 8:
                # ReparameterizedProteinEditFlowModel: (lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, logits_type, pi_type)
                lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, logits_type, pi_type = model_output
            elif len(model_output) == 5:
                # ProteinEditFlowModel: (lam_ins, logits_ins, lam_del, lam_sub, logits_sub)
                lam_ins, logits_ins, lam_del, lam_sub, logits_sub = model_output
                lam_total = lam_ins + lam_del + lam_sub
                pi_type = torch.stack([lam_ins, lam_del, lam_sub], dim=-1) / lam_total.clamp_min(1e-12)
            else:
                raise ValueError(f"Unexpected model output length: {len(model_output)}")

            candidates = [x.squeeze(0)] # compute the scores of current sequence with the candidates
            base_rates = []
            candidate_edit_stats = []  # Store edit stats for each candidate
            candidate_to_stats = {}  # Map candidate tensor to edit stats (for deduplication)
            for cand_idx in range(num_candidates):
                cand_seq, base_rate, _, edit_stats = _sample_multiple_edits_batch(
                    x,
                    lam_ins, logits_ins,
                    lam_del, lam_sub, logits_sub,
                    pad_id, bos_id, eos_id,
                    allowed_tokens,
                    delta=float(1/(num_steps-1)),
                    max_len_cap=max_len_cap,
                    protected_mask=protected_mask,
                    pam_scale_edits=pam_scale_edits,
                    pam_edit_scale_factor=pam_edit_scale_factor,
                    deletion_rate_scale=deletion_rate_scale,
                    debug_context=f"CANDIDATE step={step} cand={cand_idx}",
                )
                if not torch.equal(cand_seq, x):
                    cand_seq_squeezed = cand_seq.squeeze(0)
                    # Use a hash of the tensor as key (simple approach)
                    cand_key = tuple(cand_seq_squeezed.cpu().tolist())
                    if cand_key not in candidate_to_stats:
                        candidates.append(cand_seq_squeezed)
                        base_rates.append(base_rate)
                        candidate_edit_stats.append(edit_stats)
                        candidate_to_stats[cand_key] = len(candidate_edit_stats) - 1
                    else:
                        # Duplicate candidate, keep the stats from first occurrence
                        pass
            batch_candidates = torch.nn.utils.rnn.pad_sequence(candidates, batch_first=True, padding_value=pad_id)
            num_generated_candidates = len(candidates) - 1  # Exclude the current sequence
            # print("Initial Candidates: ", len(candidates) - 1)
            # pdb.set_trace()
            # We only want the survived candidates to improve the objective weights
            start = time.time()
            cand_logG, cand_ws = _G_T(batch_candidates, objective_models, constraint_models, w, rho, ref_z, beta_t, tokenizer, ws_for_invalid=True, debug_context=f"CANDIDATE_EVAL step={step}")      
            # print("Candidate Time: ", time.time() - start)

            curr_logG = cand_logG[0]
            curr_ws = cand_ws[0]
            cand_logG = cand_logG[1:]
            cand_ws = cand_ws[1:]
            batch_candidates = batch_candidates[1:, :]
            
            # DEBUG: Print final scores used for candidate selection (compact)
            if len(cand_ws) > 0:
                # valid_mask = torch.isfinite(cand_ws)
                valid_mask = torch.isfinite(cand_logG)   # valid = passed constraints
                if valid_mask.any():
                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    print(f"[CANDIDATE_SELECTION step={step}] Current: logG={curr_logG.item():.4f}, WS={curr_ws.item():.4f} | "
                          f"Valid: {valid_mask.sum().item()}/{len(cand_ws)} | "
                          f"Top 3: {', '.join([f'logG={cand_logG[valid_indices[i]].item():.4f},WS={cand_ws[valid_indices[i]].item():.4f}' for i in range(min(3, len(valid_indices)))])}")
                else:
                    print(f"[CANDIDATE_SELECTION step={step}] WARNING: No valid candidates (all failed constraints)")
                    
                    # Debug: Print which constraints each candidate failed
                    if len(batch_candidates) > 0:
                        candidate_seqs = tokenizer.batch_decode(batch_candidates, skip_special_tokens=True)
                        candidate_seqs_clean = [seq.replace(" ", "").replace("\n", "") for seq in candidate_seqs]
                        print(f"[CONSTRAINT_FAILURE_DEBUG step={step}] Analyzing constraint failures for {len(batch_candidates)} candidates:")
                        for cand_idx in range(len(batch_candidates)):
                            failed_constraints = []
                            seq_len = len(candidate_seqs_clean[cand_idx])
                            for constraint in constraint_models:
                                constraint_name = constraint.__class__.__name__
                                if hasattr(constraint, "predict_batch"):
                                    result = constraint.predict_batch([candidate_seqs[cand_idx]])[0]
                                else:
                                    result = constraint(None, candidate_seqs[cand_idx])
                                if not result:
                                    # Get failure reason
                                    if constraint_name == "MaxTargetLength":
                                        failed_constraints.append(f"{constraint_name}(length {seq_len} >= {constraint.max_target_length})")
                                    elif constraint_name == "TargetLength":
                                        failed_constraints.append(f"{constraint_name}(length {seq_len} != {constraint.target_length})")
                                    elif constraint_name == "ProteinLength":
                                        failed_constraints.append(f"{constraint_name}(length {seq_len} not in [{constraint.L0 // 2}, {constraint.L0}))")
                                    elif constraint_name == "Cas9DomainCompleteness":
                                        failed_constraints.append(f"{constraint_name}(domain incomplete)")
                                    elif constraint_name == "PAMMatchingConstraint":
                                        failed_constraints.append(f"{constraint_name}(predicted PAM does not match target)")
                                    elif constraint_name == "Cas9ScoreThreshold":
                                        failed_constraints.append(f"{constraint_name}(Cas9 score <= {constraint.threshold})")
                                    else:
                                        failed_constraints.append(f"{constraint_name}")
                            if failed_constraints:
                                print(f"  Candidate {cand_idx}: length={seq_len}, failed: {', '.join(failed_constraints)}")
                            else:
                                print(f"  Candidate {cand_idx}: length={seq_len}, passed all constraints (unexpected!)")

            if len(batch_candidates) == 0:
                if pam_masker is not None:
                    print(f"[PAM DEBUG] Step {step}: No candidates generated (all identical to current sequence). "
                          f"This may indicate PAM mask is blocking all edits.")
                continue

            improve_idx = (cand_ws > curr_ws).nonzero(as_tuple=True)[0]
            survived_candidates = batch_candidates[improve_idx, :]
            base_rates = [base_rates[i] for i in improve_idx] # (num_survived_candidates,)
            survived_edit_stats = [candidate_edit_stats[i] for i in improve_idx]  # Store edit stats for survived candidates
            # print([len(seq.replace(' ' ,'')) for seq in tokenizer.batch_decode(survived_candidates, skip_special_tokens=True)])
            # print("Num Candidates Survived: ", len(improve_idx))
            if len(improve_idx) == 0:
                # Debug: Print why no candidates improved
                if num_generated_candidates > 0:
                    print(f"[PAM DEBUG] Step {step}: {num_generated_candidates} candidates generated, but none improved weighted sum.")
                    print(f"  Current weighted sum: {curr_ws.item():.6f}")
                    if len(cand_ws) > 0:
                        valid_cand_mask = torch.isfinite(cand_ws)
                        num_valid = valid_cand_mask.sum().item()
                        print(f"  Valid candidates: {num_valid}/{len(cand_ws)}")
                        if num_valid > 0:
                            print(f"  Valid candidate weighted sums: min={cand_ws[valid_cand_mask].min().item():.6f}, max={cand_ws[valid_cand_mask].max().item():.6f}, mean={cand_ws[valid_cand_mask].mean().item():.6f}")
                        else:
                            print(f"  WARNING: All candidates are invalid (don't pass constraints)!")
                        # Decode and show objective scores for current and a few candidates
                        curr_seq_str = tokenizer.batch_decode(x, skip_special_tokens=True)[0].replace(" ", "")
                        curr_obj_scores = extract_objective_vector([curr_seq_str], objective_models, device)
                        print(f"  Current sequence: ws={curr_ws.item():.6f}, obj_scores={curr_obj_scores.squeeze().tolist()}, weights={w.tolist()}")
                        # Show a few valid candidates if any
                        valid_indices = valid_cand_mask.nonzero(as_tuple=True)[0][:3]
                        for idx in valid_indices:
                            cand_seq_str = tokenizer.batch_decode(batch_candidates[idx:idx+1], skip_special_tokens=True)[0].replace(" ", "")
                            cand_obj_scores = extract_objective_vector([cand_seq_str], objective_models, device)
                            print(f"  Valid candidate {idx.item()}: ws={cand_ws[idx].item():.6f}, obj_scores={cand_obj_scores.squeeze().tolist()}")
                else:
                    print(f"[PAM DEBUG] Step {step}: No candidates generated (all identical to current sequence).")
                    if pam_masker is not None:
                        curr_seq_str = tokenizer.batch_decode(x, skip_special_tokens=True)[0].replace(" ", "")
                        if protected_mask is not None:
                            protected_count = protected_mask.sum().item()
                            print(f"  Protected positions: {protected_count}/{len(curr_seq_str)} (all edits blocked in these positions)")
                continue
            
            # Expand mask to survived batch if provided
            if protected_mask is not None:
                B_mask, L_mask = protected_mask.shape
                B_survived, L_survived = survived_candidates.shape
                
                if L_survived != L_mask:
                    # Length mismatch: mask was computed for a different sequence length
                    # Recompute mask for each candidate sequence to ensure correct positions
                    if pam_masker is not None:
                        print(f"[PAM mask] Warning: Length mismatch detected (mask_len={L_mask}, candidate_len={L_survived}). "
                              f"Recomputing mask for each candidate sequence.")
                        protected_masks = []
                        for i in range(B_survived):
                            cand_seq = survived_candidates[i:i+1]  # Keep batch dim
                            seq_str = tokenizer.batch_decode(cand_seq, skip_special_tokens=True)[0].replace(" ", "")
                            cand_mask = pam_masker.build_no_del_mask(cand_seq, seq_str, pad_id=pad_id, bos_at_index0=True)
                            protected_masks.append(cand_mask)
                        protected_mask_expanded = torch.cat(protected_masks, dim=0)  # (B_survived, L_survived)
                    else:
                        # No masker available, can't recompute - skip mask
                        print(f"[PAM mask] Warning: Length mismatch (mask_len={L_mask}, candidate_len={L_survived}) "
                              f"but no pam_masker available. Skipping mask for rollouts.")
                        protected_mask_expanded = None
                else:
                    # Lengths match, safe to expand
                    protected_mask_expanded = protected_mask.expand(B_survived, -1)
            else:
                protected_mask_expanded = None
            
            # Keep all the rollout terminal sequences in one batch
            # Note: Rollouts use masking mode (not scaling) to preserve stability
            start = time.time()
            x_Ts = short_rollout_batch(model, survived_candidates, time_grid, step, pad_id, bos_id, eos_id, allowed_tokens, max_len_cap, num_rollouts, num_steps, protected_mask=protected_mask_expanded)
            # print("Rollout Time: ", time.time() - start)

            # Debug: Print minimum terminal sequence length per candidate
            if len(survived_candidates) > 0:
                terminal_seqs = tokenizer.batch_decode(x_Ts, skip_special_tokens=True)
                terminal_lengths = [len(seq.replace(" ", "").replace("\n", "")) for seq in terminal_seqs]
                # Reshape: (num_candidates, num_rollouts) - each candidate has num_rollouts terminal sequences
                num_survived = len(survived_candidates)
                terminal_lengths_reshaped = [terminal_lengths[i*num_rollouts:(i+1)*num_rollouts] for i in range(num_survived)]
                min_lengths_per_candidate = [min(lengths) for lengths in terminal_lengths_reshaped]
                print(f"[TERMINAL_LENGTHS step={step}] Min length per candidate (across {num_rollouts} rollouts): {min_lengths_per_candidate}")
                
                # Debug: Print objective values, Cas9 scores, lengths, and predicted PAMs for all terminal sequences per candidate
                terminal_seqs_clean = [seq.replace(" ", "").replace("\n", "") for seq in terminal_seqs]
                
                # Get objective values for all terminal sequences
                obj_vals, obj_names = extract_objective_vector(terminal_seqs_clean, objective_models, device, return_names=True)
                obj_vals = obj_vals.cpu().tolist()  # list of lists: (num_total_terminals, num_objectives)
                
                # Find Cas9 classifier and PAM matching objects
                cas9_classifier_obj = None
                pam_matching_obj = None
                for obj in objective_models:
                    if isinstance(obj, Cas9Classification):
                        cas9_classifier_obj = obj
                    if isinstance(obj, PAMMatching):
                        pam_matching_obj = obj
                
                # Get Cas9 scores for all terminal sequences
                cas9_scores = None
                if cas9_classifier_obj is not None:
                    cas9_scores = cas9_classifier_obj.get_scores(terminal_seqs_clean)  # list of length num_total_terminals
                
                # Get predicted PAMs for all terminal sequences
                predicted_pams_all = None
                if pam_matching_obj is not None:
                    predicted_pams_all = pam_matching_obj.predict_pam(terminal_seqs_clean)  # list of length num_total_terminals
                
                # Print per candidate
                print(f"[TERMINAL_DEBUG step={step}] Per-candidate terminal sequence details:")
                for cand_idx in range(num_survived):
                    print(f"  Candidate {cand_idx}:")
                    start_idx = cand_idx * num_rollouts
                    end_idx = start_idx + num_rollouts
                    
                    for rollout_idx in range(num_rollouts):
                        term_idx = start_idx + rollout_idx
                        seq_clean = terminal_seqs_clean[term_idx]
                        seq_len = terminal_lengths[term_idx]
                        
                        # Objective values
                        obj_str = ", ".join([f"{obj_names[i]}={obj_vals[term_idx][i]:.4f}" for i in range(len(obj_names))])
                        
                        # Cas9 score
                        cas9_str = f"cas9_score={cas9_scores[term_idx]:.4f}" if cas9_scores is not None else "cas9_score=N/A"
                        
                        # Predicted PAM
                        pam_str = f"predicted_pam={predicted_pams_all[term_idx]}" if predicted_pams_all is not None else "predicted_pam=N/A"
                        
                        print(f"    Rollout {rollout_idx}: len={seq_len}, {obj_str}, {cas9_str}, {pam_str}")

            # pdb.set_trace()
            # Constraints are taken into account for the terminal sequences
            start = time.time()
            logG, _, = _G_T(x_Ts, objective_models, constraint_models, w, rho, ref_z, beta_t, tokenizer, ws_for_invalid=False, debug_context=f"ROLLOUT_TERMINAL step={step}")      
            # pdb.set_trace()

            # Save the best teminal sequence
            curr_best_terminal_logG = torch.max(logG)
            if best_terminal_logG <= curr_best_terminal_logG:
                best_terminal_idx = torch.argmax(logG)
                best_terminal = x_Ts[best_terminal_idx].unsqueeze(0)
                best_terminal_logG = curr_best_terminal_logG

                best_terminal_seq = tokenizer.batch_decode(best_terminal.tolist(), skip_special_tokens=True)[0]
                print("\nSaved Best Terminal: ", best_terminal_seq)
                print("Saved Best Terminal Length: ", len(best_terminal_seq.replace(' ', '')))
                print("Saved Best Terminal logG: ", best_terminal_logG)
                
                # If logG is -inf, check which constraints failed
                # Handle both tensor and float values
                logG_value = best_terminal_logG.item() if isinstance(best_terminal_logG, torch.Tensor) else best_terminal_logG
                if math.isinf(logG_value) and logG_value < 0:
                    print("Saved Best Terminal FAILED constraints. Checking which constraints failed:")
                    failed_constraints = []
                    seq_clean = best_terminal_seq.replace(' ', '').replace('\n', '')
                    seq_len = len(seq_clean)
                    
                    # Debug: Print sequence being evaluated
                    print(f"  [DEBUG] Sequence being evaluated (length={seq_len}):")
                    print(f"    First 100 chars: {seq_clean[:100]}")
                    print(f"    Last 100 chars: {seq_clean[-100:]}")
                    print(f"    Sequence contains spaces: {' ' in seq_clean}")
                    newline_char = '\n'
                    print(f"    Sequence contains newlines: {newline_char in seq_clean}")
                    
                    for constraint in constraint_models:
                        constraint_name = constraint.__class__.__name__
                        if hasattr(constraint, "predict_batch"):
                            result = constraint.predict_batch([seq_clean])[0]
                        else:
                            result = constraint(None, seq_clean)
                        
                        if not result:
                            # Get failure reason
                            if constraint_name == "MaxTargetLength":
                                failed_constraints.append(f"{constraint_name}(length {seq_len} >= {constraint.max_target_length})")
                            elif constraint_name == "TargetLength":
                                failed_constraints.append(f"{constraint_name}(length {seq_len} != {constraint.target_length})")
                            elif constraint_name == "ProteinLength":
                                failed_constraints.append(f"{constraint_name}(length {seq_len} not in [{constraint.L0 // 2}, {constraint.L0}))")
                            elif constraint_name == "Cas9DomainCompleteness":
                                failed_constraints.append(f"{constraint_name}(domain incomplete)")
                            elif constraint_name == "PAMMatchingConstraint":
                                # Get predicted PAM to show what it was vs target
                                predicted_pams = constraint.pam_matching_obj.predict_pam([seq_clean])
                                predicted_pam = predicted_pams[0] if predicted_pams else "N/A"
                                failed_constraints.append(f"{constraint_name}(predicted PAM '{predicted_pam}' does not match target '{constraint.target_pam}')")
                            elif constraint_name == "Cas9ScoreThreshold":
                                # Get actual Cas9 score to show what it was vs threshold
                                scores = constraint.cas9_classifier_obj.get_scores([seq_clean])
                                cas9_score = scores[0] if scores else 0.0
                                failed_constraints.append(f"{constraint_name}(Cas9 score {cas9_score:.4f} <= {constraint.threshold})")
                            else:
                                failed_constraints.append(f"{constraint_name}")
                    
                    if failed_constraints:
                        print(f"  Failed constraints: {', '.join(failed_constraints)}")
                    else:
                        print(f"  WARNING: logG is -inf but no constraints failed (unexpected!)")

            # print("Terminal Time: ", time.time() - start)
            logG = logG.reshape(survived_candidates.shape[0], num_rollouts)
            log_h_hat = torch.logsumexp(logG, dim=1) - math.log(num_rollouts)   # (num_survived_candidates,)
            idx = (logG.max(dim=1).values > curr_logG).nonzero(as_tuple=True)[0]
            final_survived_candidates = survived_candidates[idx, :]
            if len(final_survived_candidates) == 0:
                continue

            # DEBUG: Print final selection scores (compact)
            if len(idx) > 0:
                top_logG_vals = [logG.max(dim=1).values[cand_idx].item() for cand_idx in idx[:3]]
                print(f"[FINAL_SELECTION step={step}] Improved: {len(idx)}/{len(survived_candidates)} | "
                      f"Top logG: {', '.join([f'{v:.4f}' for v in top_logG_vals])}")

            # Doob-like transform
            log_h_hat = log_h_hat[idx]
            base_rates_t = torch.tensor([base_rates[i] for i in idx.tolist()], device=device, dtype=torch.float32)
            log_base = 0.5 * torch.log(base_rates_t.clamp_min(1e-30))
            log_weights = log_base + log_h_hat
            probs = torch.softmax(log_weights, dim=0) 
            
            # if torch.isnan(probs).any():
            #     pdb.set_trace()
            selected_idx = torch.multinomial(probs, 1).item()
            print(f"[FINAL_SELECTION step={step}] Selected: cand {idx[selected_idx].item()} (prob={probs[selected_idx].item():.4f})")
            x = final_survived_candidates[selected_idx].unsqueeze(0)
            
            # Reprint debug info for the selected candidate
            selected_edit_stats = survived_edit_stats[idx[selected_idx]]
            num_ins = selected_edit_stats['num_ins'][0]
            num_del = selected_edit_stats['num_del'][0]
            num_sub = selected_edit_stats['num_sub'][0]
            total_edits = selected_edit_stats['total_edits'][0]
            avg_ins_rate, avg_del_rate, avg_sub_rate = selected_edit_stats['avg_rates'][0] if 'avg_rates' in selected_edit_stats and selected_edit_stats['avg_rates'] else (0.0, 0.0, 0.0)
            
            # Accumulate edit statistics for selected steps
            total_ins += num_ins
            total_del += num_del
            total_sub += num_sub
            
            # DEBUG: Print final logG calculation for selected candidate (compact)
            selected_seq_str = tokenizer.batch_decode(x, skip_special_tokens=True)[0].replace(" ", "")
            selected_obj_scores, obj_names = extract_objective_vector([selected_seq_str], objective_models, device, return_names=True)
            selected_obj_scores = selected_obj_scores.squeeze(0)
            selected_logG = logG.max(dim=1).values[idx[selected_idx]].item()
            
            # Find DeletionCount objective for absolute count
            deletion_obj = None
            for obj in objective_models:
                if hasattr(obj, 'original_length') and hasattr(obj, 'max_deletion'):
                    deletion_obj = obj
                    break
            
            # Compute logG components
            diff = selected_obj_scores - ref_z
            w_diff = w * diff
            term1 = torch.min(w_diff).item()
            term2 = (rho * torch.sum(w_diff)).item()
            u_atc_val = term1 + term2
            logG_calc = beta_t * u_atc_val
            
            # Format output
            diff_parts = []
            w_diff_parts = []
            for obj_idx, obj_name in enumerate(obj_names):
                raw_score = selected_obj_scores[obj_idx].item()
                diff_val = diff[obj_idx].item()
                w_diff_val = w_diff[obj_idx].item()
                weight = w[obj_idx].item()
                ref_val = ref_z[obj_idx].item()
                
                if obj_name == 'deletion_count' and deletion_obj is not None:
                    abs_deletion = deletion_obj.original_length - len(selected_seq_str)
                    diff_parts.append(f"{obj_name}: {raw_score:.4f} (abs: {abs_deletion}) - {ref_val:.4f} = {diff_val:.4f}")
                else:
                    diff_parts.append(f"{obj_name}: {raw_score:.4f} - {ref_val:.4f} = {diff_val:.4f}")
                w_diff_parts.append(f"{obj_name}: {diff_val:.4f} × {weight:.3f} = {w_diff_val:.4f}")
            
            # Always print selected step info, even if no edits
            edit_info = ""
            if total_edits > 0:
                pct_ins = 100.0 * num_ins / total_edits
                pct_del = 100.0 * num_del / total_edits
                pct_sub = 100.0 * num_sub / total_edits
                edit_info = f" | Edits: ins={num_ins}({pct_ins:.0f}%), del={num_del}({pct_del:.0f}%), sub={num_sub}({pct_sub:.0f}%)"
            else:
                edit_info = " | No edits"
            
            print(f"[SELECTED step={step}] len={len(selected_seq_str)} | diffs=[{', '.join(diff_parts)}] | "
                  f"w×diffs=[{', '.join(w_diff_parts)}] | min={term1:.4f}, rho×sum={term2:.4f} | "
                  f"u_atc={u_atc_val:.4f}, logG={selected_logG:.4f} (calc: {logG_calc:.4f}){edit_info}")
            
            protein_seq = tokenizer.batch_decode(x.tolist(), skip_special_tokens=True)[0]
            # print(protein_seq)  # Commented out: don't print full sequence at every step
            print("Current Length: ", len(protein_seq.replace(' ', '')))
            compute_scores_print([protein_seq], objective_models, constraint_models, device)
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
            tokenizer=tokenizer,
            pam_masker=pam_masker,
            deletion_rate_scale=deletion_rate_scale,
        )

        if logG_final_rollout >= best_terminal_logG:
            best_terminal = x_final_rollout

    # Print cumulative edit statistics for all selected steps
    total_all_edits = total_ins + total_del + total_sub
    
    # Compute actual net deletion count from final sequence
    final_seq_str = tokenizer.batch_decode(best_terminal, skip_special_tokens=True)[0].replace(" ", "")
    final_seq_len = len(final_seq_str)
    
    # Find DeletionCount objective to get original length
    deletion_obj = None
    for obj in objective_models:
        if hasattr(obj, 'original_length') and hasattr(obj, 'max_deletion'):
            deletion_obj = obj
            break
    
    net_deletion_count = None
    if deletion_obj is not None:
        net_deletion_count = deletion_obj.original_length - final_seq_len
    
    print(f"\n{'='*80}")
    print(f"[FINAL SUMMARY] Cumulative edit statistics for all selected steps:")
    print(f"  NOTE: These statistics only count edits in selected candidate steps during generation,")
    print(f"        not including any edits made during finalization rollouts.")
    if total_all_edits > 0:
        pct_ins_total = 100.0 * total_ins / total_all_edits
        pct_del_total = 100.0 * total_del / total_all_edits
        pct_sub_total = 100.0 * total_sub / total_all_edits
        print(f"  Total insertions: {total_ins} ({pct_ins_total:.1f}%)")
        print(f"  Total deletions: {total_del} ({pct_del_total:.1f}%)")
        print(f"  Total substitutions: {total_sub} ({pct_sub_total:.1f}%)")
        print(f"  Total edits: {total_all_edits}")
    else:
        print(f"  No edits were made across all selected steps")
    
    if net_deletion_count is not None:
        print(f"\n  Net deletion count (from final sequence length): {net_deletion_count}")
        print(f"    Original length: {deletion_obj.original_length}")
        print(f"    Final length: {final_seq_len}")
        print(f"    Difference: {net_deletion_count} (may differ from edit statistics due to")
        print(f"                 insertions reducing net deletions and finalization edits)")
    
    print(f"{'='*80}\n")

    return best_terminal

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcomol_config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--max_len_cap", type=int, default=None)
    parser.add_argument("--num_candidates", type=int, default=10)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--beta_start", type=float, default=1.0)
    parser.add_argument("--beta_end", type=float, default=3.0)
    parser.add_argument("--num_final_rollouts", type=int, default=50)
    parser.add_argument("--deletion_rate_scale", type=float, default=1500.0,
                        help="Scaling factor for deletion rate during candidate generation and rollouts (default: 1500.0)")
    parser.add_argument("--num_sequences", type=int, default=100,
                        help="Number of sequences to generate overall (default: 100)")
    parser.add_argument("--objective_weights", type=float, nargs='+')
    parser.add_argument("--ref_z", type=float, nargs='+')
    parser.add_argument("--rho", type=float, default=1)
    parser.add_argument("--output_file", type=str, default=None)
    
    # Cas9 domain completeness constraint arguments
    parser.add_argument("--cas9_hmm_db", type=str, default=None,
                        help="Path to HMM database for Cas9 domain detection (e.g., cas9_bootstrap_pfam.hmm). If not provided, domain detection constraint will be disabled.")
    parser.add_argument("--cas9_evalue", type=float, default=1e-2,
                        help="E-value cutoff for domain detection")
    parser.add_argument("--cas9_minlen", type=int, default=35,
                        help="Minimum domain length for detection")
    
    # Cas9 classifier arguments
    parser.add_argument("--cas9_classifier_ckpt", type=str, default=None,
                        help="Path to Cas9 classifier checkpoint (default: uses default path)")
    parser.add_argument("--cas9_classifier_config", type=str, default=None,
                        help="Path to Cas9 classifier config YAML (optional)")
    parser.add_argument("--cas9_score_threshold", type=float, default=None,
                        help="Minimum Cas9 classifier score threshold (0-1). If provided, adds a terminal constraint requiring sequences to have Cas9 score > threshold. Default: None (no threshold constraint)")
    
    # PAM matching objective arguments
    parser.add_argument("--target_pam", type=str, default=None,
                        help="Target PAM sequence (10 nucleotides, e.g., 'NGGNNNNNNN'). If set to 'matching', will predict the PAM from the input sequence and use that as the target. If provided, enables PAM matching objective.")
    parser.add_argument("--pam_model_name", type=str, default="Profluent-Bio/protein2pam-cas9_full",
                        help="HuggingFace model name for PAM prediction (default: Profluent-Bio/protein2pam-cas9_full)")
    parser.add_argument("--pam_no_entropy", action="store_false", dest="pam_use_entropy", default=True,
                        help="Disable entropy-based scoring for N positions. Score only considers log likelihood of target PAM at non-N positions. By default, entropy scoring is enabled for N positions.")
    parser.add_argument("--use_ce_loss", action="store_true",
                        help="Use cross-entropy loss for PAM matching objective instead of log probability approach. Score = exp(-mean_ce_loss) to convert to [0, 1] range.")
    parser.add_argument("--pam_min_confidence", type=float, default=0.55,
                        help="Minimum probability threshold for PAM prediction. If max probability < pam_min_confidence, predict 'N' instead of specific nucleotide (default: 0.55)")
    parser.add_argument("--pam_prediction_temperature", type=float, default=1.0,
                        help="Temperature for PAM prediction. Values < 1.0 make distributions sharper (more confident), > 1.0 make them softer. Default: 1.0 (no temperature scaling). Note: This only affects prediction, not scoring.")
    
    # Deletion count objective arguments
    parser.add_argument("--max_deletion_percentage", type=float, default=None,
                        help="Maximum deletion percentage (0-1) for normalization. The deletion count objective will return a value between 0 and 1, representing the percentage of (original_length * max_deletion_percentage) that has been deleted. If not provided, defaults to 1.0 (allowing 100%% deletion).")
    
    # Target length constraint arguments
    parser.add_argument("--target_length", type=int, default=None,
                        help="Target sequence length in amino acids. If provided, adds a terminal constraint requiring the final sequence length to exactly match this value.")
    parser.add_argument("--max_target_length", type=float, default=None,
                        help="Maximum target sequence length in amino acids (exclusive). If provided as a decimal (0-1), interpreted as a percentage of input sequence length. If provided as an integer (>=1), used as absolute value. If provided, adds a terminal constraint requiring the final sequence length to be strictly less than this value.")
    
    # PAM/PI-domain masking arguments
    parser.add_argument("--pam_hmm_db", type=str, default=None,
                        help="Path to cas9_pi.hmm (mini Pfam DB with Cas9_PI models). Required if --pam_mask or --pam_scale_edits is set.")
    parser.add_argument("--pam_mask", action="store_true",
                        help="Enable PAM/PI domain masking (blocks all edits in PI domain region). Requires --pam_hmm_db.")
    parser.add_argument("--pam_mask_max_len", type=int, default=200,
                        help="Max number of AA positions to hard-mask (Option 1 core window cap).")
    parser.add_argument("--pam_evalue", type=float, default=1e-5,
                        help="Per-domain i-evalue cutoff for PI hits.")
    parser.add_argument("--pam_refresh_every", type=int, default=1,
                        help="Recompute PI deletion mask every N accepted steps (>=1).")
    parser.add_argument("--hmmscan_bin", type=str, default="hmmscan",
                        help="hmmscan executable (default: hmmscan).")
    parser.add_argument("--hmmscan_cpu", type=int, default=1,
                        help="CPUs to give hmmscan.")
    parser.add_argument("--pam_debug", action="store_true",
                        help="Print PI masking debug info during generation.")
    parser.add_argument("--pam_scale_edits", action="store_true",
                        help="If set, scale up insertion/substitution rates in PAM region instead of masking edits. Requires --pam_hmm_db.")
    parser.add_argument("--pam_edit_scale_factor", type=float, default=10.0,
                        help="Scaling factor for insertion/substitution rates in PAM region when --pam_scale_edits is enabled (default: 10.0).")

    args = parser.parse_args()

    # Validate that pam_hmm_db is provided if pam_mask or pam_scale_edits is enabled
    if args.pam_mask and args.pam_hmm_db is None:
        raise ValueError("--pam_mask requires --pam_hmm_db to be specified")
    if args.pam_scale_edits and args.pam_hmm_db is None:
        raise ValueError("--pam_scale_edits requires --pam_hmm_db to be specified")
    if args.pam_mask and args.pam_scale_edits:
        raise ValueError("--pam_mask and --pam_scale_edits cannot be used together. Choose one.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.pcomol_config, "r") as f:
        cfg = edict(yaml.safe_load(f))

    editflow, source_dist, tokenizer, pad_id, bos_id, eos_id, eps_id = build_model_and_stuff(cfg, device)

    ckpt = torch.load(args.ckpt, map_location=device)
    editflow.load_state_dict(ckpt["state_dict"], strict=False)
    model = editflow.model.to(device)
    model.eval()

    x0 = tokenize_input_str(args.input, cfg, tokenizer, bos_id, eos_id, pad_id, device)

    allowed_tokens = torch.tensor(
        [tok for tok in source_dist._allowed_tokens if tok not in (eps_id,)],
        device=device,
        dtype=torch.long,
    )

    # Initialize Cas9 classifier objective with shared ESM model
    # This avoids loading ESM twice, saving GPU memory
    cas9_classifier = Cas9Classification(
        device=device,
        checkpoint_path=args.cas9_classifier_ckpt,
        config_path=args.cas9_classifier_config,
        shared_esm_model=model.esm_emb  # Share ESM model from editflow
    )
    
    # Initialize deletion count objective (maximizes number of deletions, normalized to [0, 1])
    deletion_count_obj = DeletionCount(original_seq=args.input, max_deletion_percentage=args.max_deletion_percentage)
    
    objective_models = [cas9_classifier, deletion_count_obj]
    
    # Initialize PAM matching objective if target PAM is provided
    pam_matching_obj_for_constraint = None  # Will be set if PAM matching is enabled
    if args.target_pam is not None:
        # Handle 'matching' mode: predict PAM from input sequence
        if args.target_pam.lower() == 'matching':
            print(f"Target PAM set to 'matching' - will predict PAM from input sequence")
            # Create temporary PAM matching object to predict the input sequence's PAM
            temp_pam_obj = PAMMatching(
                device=device,
                target_pam="NNNNNNNNNN",  # Dummy target, we'll replace it
                model_name=args.pam_model_name,
                use_entropy_for_n_positions=args.pam_use_entropy,
                use_ce_loss=args.use_ce_loss,
                pam_min_confidence=args.pam_min_confidence,
                pam_prediction_temperature=args.pam_prediction_temperature
            )
            # Predict PAM for input sequence
            predicted_pams = temp_pam_obj.predict_pam([args.input])
            if predicted_pams:
                actual_target_pam = predicted_pams[0]
                print(f"  Predicted PAM for input sequence: {actual_target_pam}")
                print(f"  Using this as target PAM for optimization")
            else:
                raise ValueError("Failed to predict PAM for input sequence")
            
            # Now create the actual PAM matching objective with the predicted PAM
            entropy_mode = "enabled" if args.pam_use_entropy else "disabled"
            ce_loss_mode = "CE loss" if args.use_ce_loss else "log probability"
            print(f"  Entropy scoring for N positions: {entropy_mode}")
            print(f"  Scoring method: {ce_loss_mode}")
            pam_matching_obj = PAMMatching(
                device=device,
                target_pam=actual_target_pam,
                model_name=args.pam_model_name,
                use_entropy_for_n_positions=args.pam_use_entropy,
                use_ce_loss=args.use_ce_loss,
                pam_min_confidence=args.pam_min_confidence,
                pam_prediction_temperature=args.pam_prediction_temperature
            )
            objective_models.append(pam_matching_obj)
            print(f"PAM matching objective initialized successfully with target PAM: {actual_target_pam}")
            
            # Store reference to PAM matching object for constraint creation later
            pam_matching_obj_for_constraint = pam_matching_obj
        else:
            # Regular mode: use provided PAM string
            print(f"Initializing PAM matching objective with target PAM: {args.target_pam}")
            entropy_mode = "enabled" if args.pam_use_entropy else "disabled"
            ce_loss_mode = "CE loss" if args.use_ce_loss else "log probability"
            print(f"  Entropy scoring for N positions: {entropy_mode}")
            print(f"  Scoring method: {ce_loss_mode}")
            pam_matching_obj = PAMMatching(
                device=device,
                target_pam=args.target_pam,
                model_name=args.pam_model_name,
                use_entropy_for_n_positions=args.pam_use_entropy,
                use_ce_loss=args.use_ce_loss,
                pam_min_confidence=args.pam_min_confidence,
                pam_prediction_temperature=args.pam_prediction_temperature
            )
            objective_models.append(pam_matching_obj)
            print(f"PAM matching objective initialized successfully.")
            
            # Store reference to PAM matching object for constraint creation later
            pam_matching_obj_for_constraint = pam_matching_obj

    num_objectives = len(objective_models)
    if not args.objective_weights:
        objective_weights = torch.tensor([1.0 / num_objectives] * num_objectives).to(device)
    else:
        objective_weights = torch.tensor(args.objective_weights).to(device)

    if not args.ref_z:
        ref_z = torch.zeros(num_objectives).to(device)
    else:
        ref_z = torch.tensor(args.ref_z).to(device)

    # Initialize protein length constraint
    protein_length_constraint = ProteinLength(
        orig_protein_seq=args.input,
        cfg=cfg,
        tokenizer=tokenizer,
    )
    constraint_models = [protein_length_constraint]
    
    # Initialize Cas9 domain completeness constraint (optional)
    if args.cas9_hmm_db is not None:
        cas9_domain_constraint = Cas9DomainCompleteness(
            hmm_path=args.cas9_hmm_db,
            ev1=args.cas9_evalue,
            minlen1=args.cas9_minlen,
            cpu=1,  # Can be made configurable if needed
        )
        constraint_models.append(cas9_domain_constraint)
        print(f"Cas9 domain completeness constraint enabled with HMM database: {args.cas9_hmm_db}")
    else:
        print("Cas9 domain completeness constraint disabled (--cas9_hmm_db not provided)")
    
    # Initialize target length constraint if specified
    if args.target_length is not None:
        target_length_constraint = TargetLength(target_length=args.target_length)
        constraint_models.append(target_length_constraint)
        print(f"Target length constraint enabled: sequences must have exactly {args.target_length} amino acids")
    
    # Initialize max target length constraint if specified
    if args.max_target_length is not None:
        # If max_target_length is a decimal (0 < value < 1), interpret as percentage of input length
        # If it's >= 1, use as absolute value
        input_length = len(args.input.replace(' ', ''))
        if 0 < args.max_target_length < 1:
            # Decimal: interpret as percentage
            computed_max_target_length = int(args.max_target_length * input_length)
            print(f"Max target length constraint: {args.max_target_length} (decimal) interpreted as {args.max_target_length*100:.1f}% of input length ({input_length}) = {computed_max_target_length}")
        else:
            # Integer: use as absolute value
            computed_max_target_length = int(args.max_target_length)
            print(f"Max target length constraint: {args.max_target_length} (integer) used as absolute value = {computed_max_target_length}")
        
        max_target_length_constraint = MaxTargetLength(max_target_length=computed_max_target_length)
        constraint_models.append(max_target_length_constraint)
        print(f"Max target length constraint enabled: sequences must have length < {computed_max_target_length} amino acids")

    # Initialize PAM matching constraint if target_pam is specified
    if args.target_pam is not None and pam_matching_obj_for_constraint is not None:
        pam_matching_constraint = PAMMatchingConstraint(pam_matching_obj_for_constraint)
        constraint_models.append(pam_matching_constraint)
        print(f"PAM matching constraint enabled: sequences must have predicted PAM matching target PAM")

    # Initialize Cas9 score threshold constraint if threshold is specified
    if args.cas9_score_threshold is not None:
        cas9_score_threshold_constraint = Cas9ScoreThreshold(cas9_classifier, args.cas9_score_threshold)
        constraint_models.append(cas9_score_threshold_constraint)
        print(f"Cas9 score threshold constraint enabled: sequences must have Cas9 score > {args.cas9_score_threshold}")

    # Initialize PAM/PI masker if --pam_mask flag is set
    pam_masker = None
    if args.pam_mask:
        if args.pam_hmm_db is None:
            raise ValueError("--pam_mask requires --pam_hmm_db to be specified")
        hmmscan_runner = HMMSCAN(
            hmm_db_path=args.pam_hmm_db,
            hmmscan_bin=args.hmmscan_bin,
            cpu=args.hmmscan_cpu,
        )
        pam_masker = Cas9PIMasker(
            hmmscan=hmmscan_runner,
            use_env_coords=False,         # ali coords (smaller)
            max_mask_len=args.pam_mask_max_len,
            evalue_cutoff=args.pam_evalue,
            min_ali_len=30,
            fallback_last_n=None,         # avoid masking arbitrary tail when no hit
            cache_size=2048,
        )
        print(f"[PI mask] enabled: db={args.pam_hmm_db}, max_len={args.pam_mask_max_len}, evalue<={args.pam_evalue} (blocks all edits: ins/del/sub)")
    elif args.pam_scale_edits:
        if args.pam_hmm_db is None:
            raise ValueError("--pam_scale_edits requires --pam_hmm_db to be specified")
        # For scaling mode, we still need the masker to identify the PAM region
        hmmscan_runner = HMMSCAN(
            hmm_db_path=args.pam_hmm_db,
            hmmscan_bin=args.hmmscan_bin,
            cpu=args.hmmscan_cpu,
        )
        pam_masker = Cas9PIMasker(
            hmmscan=hmmscan_runner,
            use_env_coords=False,         # ali coords (smaller)
            max_mask_len=args.pam_mask_max_len,
            evalue_cutoff=args.pam_evalue,
            min_ali_len=30,
            fallback_last_n=None,         # avoid masking arbitrary tail when no hit
            cache_size=2048,
        )
        print(f"[PI scaling] enabled: db={args.pam_hmm_db}, max_len={args.pam_mask_max_len}, evalue<={args.pam_evalue} (scales ins/sub rates by {args.pam_edit_scale_factor}x)")

    print("Initial Scores:")
    input_scores = compute_scores_print([args.input], objective_models, constraint_models, device, return_scores=True).squeeze(0)

    # Write CSV header and input sequence row if file doesn't exist
    if args.output_file is not None:
        import os
        if not os.path.exists(args.output_file):
            # Get objective names for header
            obj_names = []
            pam_obj = None
            for obj in objective_models:
                name, _ = obj(protein_tokens=None, protein_seqs=[args.input])
                obj_names.append(name)
                # Check if this is the PAM matching objective
                if hasattr(obj, 'predict_pam'):
                    pam_obj = obj
            
            with open(args.output_file, 'w') as f:
                # Header: final_len, length_diff, obj1_score, obj2_score, ..., weight1, weight2, ..., predicted_pam, [pam_ce_loss_score], [pam_probability], deletion_rate_scale, pam_min_confidence, sequence
                header_parts = ["final_len", "length_diff"]
                header_parts.extend([f"{name}_score" for name in obj_names])
                header_parts.extend([f"{name}_weight" for name in obj_names])
                if pam_obj is not None:
                    header_parts.append("predicted_pam")
                    # Add CE loss score column if using CE loss
                    if hasattr(pam_obj, 'use_ce_loss') and pam_obj.use_ce_loss:
                        header_parts.append("pam_matching_ce_loss_score")
                    # Always add probability column
                    header_parts.append("pam_matching_probability")
                header_parts.append("deletion_rate_scale")
                if pam_obj is not None:
                    header_parts.append("pam_min_confidence")
                header_parts.append("generation_time")
                header_parts.append("sequence")
                f.write(",".join(header_parts) + "\n")
                
                # Write input sequence as first row
                input_len = len(args.input.replace(' ', ''))
                f.write(f"{input_len},0")  # length_diff is 0 for input sequence
                # Write objective scores only (input_scores includes both objectives and constraints)
                num_objectives = len(objective_models)
                input_objective_scores_only = input_scores[:num_objectives]
                for score in input_objective_scores_only:
                    f.write(f",{score.item()}")
                # Write objective weights
                for weight in objective_weights:
                    f.write(f",{weight.item()}")
                # Write predicted PAM and additional PAM scores if available
                if pam_obj is not None:
                    predicted_pams = pam_obj.predict_pam([args.input])
                    predicted_pam = predicted_pams[0] if predicted_pams else ""
                    f.write(f",{predicted_pam}")
                    
                    # Write CE loss score if using CE loss (the score already in input_objective_scores_only is the CE loss)
                    if hasattr(pam_obj, 'use_ce_loss') and pam_obj.use_ce_loss:
                        # The PAM matching score in input_objective_scores_only is the CE loss score
                        pam_score_idx = None
                        for i, obj in enumerate(objective_models):
                            if obj == pam_obj:
                                pam_score_idx = i
                                break
                        if pam_score_idx is not None:
                            ce_loss_score = input_objective_scores_only[pam_score_idx].item()
                            f.write(f",{ce_loss_score}")
                    
                    # Always compute and write the direct probability of the target PAM
                    target_pam = pam_obj.target_pam
                    prob_score = pam_obj.get_score_for_pam([args.input], target_pam, use_temperature_scaling=False)[0]
                    f.write(f",{prob_score}")
                f.write(f",{args.deletion_rate_scale}")
                if pam_obj is not None:
                    f.write(f",{args.pam_min_confidence}")
                f.write(f",0")  # generation_time is 0 for input sequence (not generated)
                f.write(f",{args.input}\n")

    valid = 0
    target_valid = args.num_sequences  # how many successful designs you want

    attempt = 0
    max_attempts = 500  # safety cap so you don't infinite-loop if it keeps OOM'ing

    while valid < target_valid and attempt < max_attempts:
        attempt += 1
        try:
            # Start timing for this generation
            generation_start_time = time.time()
            
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
                tokenizer=tokenizer,
                pam_masker=pam_masker,
                pam_mask_refresh_every=max(1, args.pam_refresh_every),
                pam_debug=args.pam_debug,
                pam_scale_edits=args.pam_scale_edits,
                pam_edit_scale_factor=args.pam_edit_scale_factor,
                deletion_rate_scale=args.deletion_rate_scale,
            )

            # End timing for this generation
            generation_time = time.time() - generation_start_time

            out_str = tokenizer.batch_decode(x_T.tolist(), skip_special_tokens=True)[0]

            print("----------------------------")
            print(f"\nDesigned Sequence: {out_str}\n")
            print("Final scores:")
            scores = compute_scores_print(
                [out_str],
                objective_models, constraint_models,
                device, return_scores=True
            ).squeeze(0)

            # Only count + save on success
            valid += 1
            orig_len = len(args.input.replace(' ', ''))
            final_len = len(out_str.replace(' ', ''))
            length_diff = orig_len - final_len
            if args.output_file is not None:
                # Find PAM matching objective to get predicted PAM
                pam_obj = None
                for obj in objective_models:
                    if hasattr(obj, 'predict_pam'):
                        pam_obj = obj
                        break
                
                with open(args.output_file, 'a') as f:
                    f.write(f"{final_len},{length_diff}")
                    # Write objective scores only (scores includes both objectives and constraints)
                    num_objectives = len(objective_models)
                    objective_scores_only = scores[:num_objectives]
                    for score in objective_scores_only:
                        f.write(f",{score.item()}")
                    # Write objective weights
                    for weight in objective_weights:
                        f.write(f",{weight.item()}")
                    # Write predicted PAM and additional PAM scores if available
                    if pam_obj is not None:
                        predicted_pams = pam_obj.predict_pam([out_str])
                        predicted_pam = predicted_pams[0] if predicted_pams else ""
                        f.write(f",{predicted_pam}")
                        
                        # Write CE loss score if using CE loss (the score already in objective_scores_only is the CE loss)
                        if hasattr(pam_obj, 'use_ce_loss') and pam_obj.use_ce_loss:
                            # The PAM matching score in objective_scores_only is the CE loss score
                            pam_score_idx = None
                            for i, obj in enumerate(objective_models):
                                if obj == pam_obj:
                                    pam_score_idx = i
                                    break
                            if pam_score_idx is not None:
                                ce_loss_score = objective_scores_only[pam_score_idx].item()
                                f.write(f",{ce_loss_score}")
                        
                        # Always compute and write the direct probability of the target PAM
                        target_pam = pam_obj.target_pam
                        prob_score = pam_obj.get_score_for_pam([out_str], target_pam, use_temperature_scaling=False)[0]
                        f.write(f",{prob_score}")
                    f.write(f",{args.deletion_rate_scale}")
                    if pam_obj is not None:
                        f.write(f",{args.pam_min_confidence}")
                    f.write(f",{generation_time:.4f}")  # generation_time in seconds
                    f.write(f",{out_str}\n")

        except torch.cuda.OutOfMemoryError:
            # OOM error occurred - discard this generation and restart a new one
            print(f"[WARN] CUDA OOM during pCoMol generation (attempt {attempt}/{max_attempts}). Discarding this generation and restarting.")
            print(f"  Current progress: {valid}/{target_valid} valid sequences generated")
            
            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # optional, can help in some cases
            
            # Continue to next iteration - this will start a new generation attempt
            # Note: attempt counter already incremented, so this doesn't count as a valid sequence
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


if __name__ == "__main__":
    main()
