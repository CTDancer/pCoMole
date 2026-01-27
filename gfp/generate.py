import argparse
import torch
import yaml
from easydict import EasyDict as edict


from constraints import GFP

import sys
sys.path.append('/scratch/pCoMol')
# from model.base_models import EditFlow, ProteinEditFlowModel, SMILESEditFlowModel
from model.reparam_models import EditFlow, ProteinEditFlowModel
from model.utils import generate_from_x0, generate_from_x0_multi_edit
from logic import flow

# tokenizers used in train.py
from transformers import EsmTokenizer
import pdb

def build_model_and_stuff(cfg, device):
    """
    Rebuild exactly what train.py builds, but we won't set up lightning Trainer.
    Returns:
      editflow_module  (LightningModule)
      source_dist
      (pad_id, bos_id, eos_id)
    """
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    vocab_size = 24
    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution,
        vocab_size=vocab_size,
        special_token_ids=[0, 1, 2, 3],
    )
    pad_id = 1
    bos_id = 0
    eos_id = 2
    model = ProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)

    eps_id = getattr(cfg.flow, "eps_id", -1)
    path = flow.get_path(
        scheduler_type=cfg.flow.scheduler_type,
        exponent=cfg.flow.exponent,
        eps_id=eps_id,
    )
    loss_fn = flow.get_loss_function(
        loss_function=cfg.flow.loss_function,
        path=path,
    )

    editflow = EditFlow(
        model,
        loss_fn,
        path,
        source_distribution,
        pad_id,
        bos_id,
        eos_id,
        cfg,
    ).to(device)

    return editflow, source_distribution, tokenizer, pad_id, bos_id, eos_id, eps_id


def tokenize_input_str(input_str, tokenizer, bos_id, eos_id, device):
    toks = tokenizer(input_str, return_tensors='pt')
    ids = toks["input_ids"][0].to(device)
    if ids[0].item() != bos_id:
        ids = torch.cat([torch.tensor([bos_id], device=device), ids], dim=0)
    if ids[-1].item() != eos_id:
        ids = torch.cat([ids, torch.tensor([eos_id], device=device)], dim=0)
    x0 = ids.unsqueeze(0)  # (1, L)

    return x0


def detokenize_output(x, tokenizer, bos_id, eos_id, pad_id):
    """
    Convert a single generated sequence (1, L) back to string.
    """
    seq = x[0].tolist()
    # strip padding
    seq = [tok for tok in seq if tok != pad_id]
    # strip BOS/EOS
    if len(seq) > 0 and seq[0] == bos_id:
        seq = seq[1:]
    if len(seq) > 0 and seq[-1] == eos_id:
        seq = seq[:-1]

    return tokenizer.batch_decode([seq], skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config_test.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="path to lightning checkpoint (.ckpt)")
    parser.add_argument("--input", type=str, required=True, help="input x_0 as raw string (smiles/protein/selfies)")
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--max_len_cap", type=int, default=None)
    parser.add_argument("--op_temperature", type=float, default=1)
    parser.add_argument("--token_temperature", type=float, default=1)

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

    x_gen = generate_from_x0_multi_edit(
        model,
        x0,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        allowed_tokens=allowed_tokens,
        num_steps=args.num_steps,
        max_len_cap=args.max_len_cap,
        op_temperature=args.op_temperature,      # soften op choice
        token_temperature=args.token_temperature,   # soften token choice
    )

    out_str = detokenize_output(x_gen, tokenizer, bos_id, eos_id, pad_id)
    out_str = out_str.replace(' ', '')
    print(len(out_str))
    print('----------------------------')
    print(f"Input Sequence: {args.input}\n")
    print(f"Designed Sequence: {out_str}\n")

    gfp_classifier = GFP(device)
    gfp_probs = gfp_classifier.get_scores(out_str, return_probs=True)
    print(gfp_probs)

if __name__ == "__main__":
    main()
