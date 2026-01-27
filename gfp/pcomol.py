import argparse
from typing import List, Callable, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from generate import build_model_and_stuff, tokenize_input_str, detokenize_output
from objectives import GFPExcitationPred, GFPBrightPred
from constraints import GFP, Length, GFPEmissionPred

import pdb

# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def extract_objective_vector(seq, objective_models, device):
    values = []

    for obj in objective_models:
        score = obj(seq)
        values.append(torch.tensor(float(score), device=device, dtype=torch.float32))

    if len(values) == 0:
        return torch.empty(0, device=device)

    return torch.stack(values, dim=0)  # (m,)


def compute_scores_print(seq, objective_models, constraint_models, device):
    objective_scores = extract_objective_vector(seq, objective_models, device)
    scores = [score.item() for score in objective_scores]
    for constraint in constraint_models:
        scores.append(constraint(seq))
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
    term1 = torch.min(w * diff)
    term2 = rho * torch.sum(w * diff)
    return term1 + term2


def _G_T(
    x: torch.Tensor,
    objective_models: List[Callable[[torch.Tensor], Tuple[str, Any]]],
    constraint_models: List[Callable[[torch.Tensor], torch.Tensor]],
    w: torch.Tensor,
    rho: float,
    z: torch.Tensor,
    beta: float,
    cfg, tokenizer
):
    
    seq = tokenizer.decode(x[0].tolist()[1:-1]).replace(' ', '')

    # constraints
    for constraint in constraint_models:
        # pdb.set_trace()
        if constraint(seq) < 1:
            return torch.tensor(0.0, device=x.device), 0.0, None

    # objectives
    # pdb.set_trace()
    f_vals = extract_objective_vector(seq, objective_models, x.device)
    u_atc = _augmented_tchebycheff(f_vals, w, rho, z)
    return torch.exp(beta * u_atc), float(torch.sum(w * f_vals).item()), seq


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
def _short_rollout(
    model,
    seq: torch.Tensor,
    time_grid: torch.Tensor,
    start_idx: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    allowed_tokens: Optional[torch.Tensor],
    max_len_cap: Optional[int],
) -> torch.Tensor:
    device = seq.device
    x = seq.clone()
    for j in range(start_idx + 1, time_grid.numel()):
        t_j = time_grid[j].view(1).to(device)
        x_batch = x.unsqueeze(0)
        mask = (x_batch != pad_id)
        lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type = model(x_t=x_batch, mask=mask, t=t_j)
        x_next, _ = _sample_single_edit_from_outputs(
            x,
            lam_ins, logits_ins,
            lam_del, lam_sub, logits_sub,
            pad_id, bos_id, eos_id,
            allowed_tokens,
            max_len_cap=max_len_cap,
        )
        x = x_next
    return x.unsqueeze(0)


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

    G_last, _, seq = _G_T(
        x_last,
        objective_models,
        constraint_models,
        w,
        rho,
        ref_z,
        beta_final,
        cfg, tokenizer
    )

    if G_last.item() > 0.0:
        f_vals_last = extract_objective_vector(seq, objective_models, device)
        best_val = _augmented_tchebycheff(f_vals_last, w, rho, ref_z).item()
        best_seq = x_last
    else:
        best_val = None
        best_seq = None

    start_idx = min(last_step, time_grid.numel() - 2) if time_grid.numel() >= 2 else 0

    for _ in range(num_final_rollouts):
        x_T = _short_rollout(
            model,
            x_last.squeeze(0),
            time_grid,
            start_idx,
            pad_id,
            bos_id,
            eos_id,
            allowed_tokens,
            max_len_cap,
        )
        G, _, seq = _G_T(
            x_T,
            objective_models,
            constraint_models,
            w,
            rho,
            ref_z,
            beta_final,
            cfg, tokenizer
        )
        if G.item() <= 0.0:
            continue
    
        f_vals = extract_objective_vector(seq, objective_models, device)
        atc_val = _augmented_tchebycheff(f_vals, w, rho, ref_z).item()

        if (best_val is None) or (atc_val > best_val):
            best_val = atc_val
            best_seq = x_T

    # 3) return whichever feasible we decided is best
    if best_seq is not None:
        return best_seq

    # 4) fallback: no feasible rollout found (should be rare) → return x_last as-is
    return x_last

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
        def _ws(x: torch.Tensor) -> float:
            seq = tokenizer.decode(x[0].tolist()[1:-1]).replace(' ', '')
            constraint = constraint_models[1]
            if not constraint(seq):
                return 0
            f_vals = extract_objective_vector(seq, objective_models, device)
            return float(torch.sum(w * f_vals).item())

        for step in tqdm(range(num_steps - 1)):
            t = time_grid[step].view(1)
            frac = step / max(1, (num_steps - 1))
            beta_t = beta_start + (beta_end - beta_start) * frac
            # ws_curr = _ws(x)
            curr_G, _, _ = _G_T(x, objective_models, constraint_models, w, rho, ref_z, beta_t, cfg, tokenizer)

            # model forward
            mask = (x != pad_id)
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type = model(x_t=x, mask=mask, t=t)

            # candidates from model
            candidates = []
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

            survivors = []
            pdb.set_trace()
            for k, cand in enumerate(candidates):
                cand_G, _, _ = _G_T(cand.unsqueeze(0), objective_models, constraint_models, w, rho, ref_z, beta_t, cfg, tokenizer)
                if cand_G <= curr_G:
                    continue

                # rollout check
                scores = []
                feasible_found = False
                # max_ws = 0
                max_G = 0
                for _ in range(num_rollouts):
                    x_T = _short_rollout(
                        model,
                        cand,
                        time_grid,
                        step,
                        pad_id,
                        bos_id,
                        eos_id,
                        allowed_tokens,
                        max_len_cap,
                    )
                    G, ws, _ = _G_T(
                        x_T,
                        objective_models,
                        constraint_models,
                        w,
                        rho,
                        ref_z,
                        beta_t,
                        cfg, tokenizer
                    )
                    # max_ws = max(ws, max_ws)
                    max_G = max(G, max_G)
                    val = G.item()
                    scores.append(val)
                    if val > 0.0:
                        feasible_found = True
                # if max_ws <= ws_curr:
                    # continue
                if max_G <= curr_G:
                    continue
                if not feasible_found:
                    continue

                h_hat = sum(scores) / len(scores)

                # local improvement gate
                # ws_cand = _ws(cand.unsqueeze(0))
                # if ws_cand <= ws_curr:
                #     continue

                survivors.append((cand, base_rates[k], h_hat))

            if len(survivors) == 0:
                # no good moves; keep current
                continue

            # Doob-like sampling inside survivors: weight ∝ tempered base rate * h_hat
            weights = []
            for cand, br, h_hat in survivors:
                weights.append((br ** 0.5) * h_hat)
            weights_t = torch.tensor(weights, device=device, dtype=torch.float32)
            probs = weights_t / (weights_t.sum() + 1e-12)
            idx = torch.multinomial(probs, 1).item()
            x = survivors[idx][0].unsqueeze(0)
            # pdb.set_trace()
            seq = tokenizer.decode(x[0].tolist()[1:-1]).replace(' ', '')
            print(seq)
            print("Current Length: ", len(seq))
            compute_scores_print(seq, objective_models, constraint_models, device)
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
    
    out_str = tokenizer.decode(x_T[0].tolist()[1:-1]).replace(' ', '')
    print("----------------------------")
    print(f"Initial Sequence: {args.input}\n")
    print(f"Initial Scores:")
    compute_scores_print(args.input, objective_models, constraint_models, device)

    print(f"\nDesigned Sequence: {out_str}\n")
    print("Final scores:")
    compute_scores_print(out_str, objective_models, constraint_models, device)


if __name__ == "__main__":
    main()
