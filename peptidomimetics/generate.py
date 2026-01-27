import argparse
import torch
import yaml
from easydict import EasyDict as edict
import random
from tqdm import tqdm
from rdkit import Chem
import pandas as pd

from is_peptidomimetic import is_peptidomimetic_not_natural

import sys
sys.path.append('/scratch/pCoMol')
# from model.base_models import EditFlow, SMILESEditFlowModel
from model.reparam_models import EditFlow, SMILESEditFlowModel
from model.utils import generate_from_x0, generate_from_x0_multi_edit
from smiles_tokenizer.selfies_tokenizers import SelfiesTokenizer
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
    tokenizer = SelfiesTokenizer.load("/scratch/data/selfies/28k_mimetics/tokenizer/vocab.json")
    vocab_size = 44
    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution,
        vocab_size=vocab_size,
        special_token_ids=[0, 1, 2, 3],
    )
    pad_id = 0
    bos_id = 1
    eos_id = 2
    model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)

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
    toks = tokenizer.encode(input_str, already_selfies=False, add_bos_eos=True)
    ids = torch.tensor(toks["input_ids"]).to(device)
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

    return tokenizer.decode(seq, return_smiles=True, stop_at_eos=False, sanitize_smiles=False)

def detokenize_output_selfies(x, tokenizer, bos_id, eos_id, pad_id):
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

    return tokenizer.decode(seq, return_smiles=False, stop_at_eos=False, sanitize_smiles=False)

def randomize_smiles_from_pool(smiles_pool, n=1000):
    out = []
    while len(out) < n:
        s = random.choice(smiles_pool)
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        out.append(Chem.MolToSmiles(m, doRandom=True))  # valid randomized SMILES
    return out

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
    
    smiles_df = pd.read_csv('/scratch/data/smiles/smiles.csv')
    smiles_candidates = smiles_df['SMILES'].sample(10000, random_state=42).tolist()

    # x0 = tokenize_input_str(args.input, tokenizer, bos_id, eos_id, device)
    success = 0
    count = 0
    samples = []
    for smiles in tqdm(smiles_candidates, total=len(smiles_candidates)):
        # L = random.randint(0, 1000)
        # x0 = torch.randint(low=3, high=44, size=(L,))
        # x0[0] = 1
        # x0[-1] = 2
        # x0 = x0.unsqueeze(0).to(device)
        try:
            x0 = tokenize_input_str(smiles, tokenizer, bos_id, eos_id, device)
        except:
            continue
        # pdb.set_trace()
        # print(x0)
        allowed_tokens = torch.tensor(
            [tok for tok in source_dist._allowed_tokens if tok not in (eps_id,)],
            device=device,
            dtype=torch.long,
        )

        x_gen = generate_from_x0(
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
        # print(len(out_str))
        # print('----------------------------')
        input_str = detokenize_output(x0, tokenizer, bos_id, eos_id, pad_id)
        # print(f"Input Sequence: {input_str}\n")
        # print(f"Designed Sequence: {out_str}\n")
        samples.append(input_str)

        flag, audit = is_peptidomimetic_not_natural(out_str)
        # print(flag)
        # print(audit)

        if flag:
            success += 1

        count += 1
        if count == 1000:
            break

    print(success)

    with open('./samples.csv', 'w') as f:
        for seq in samples:
            f.write(seq + '\n')

if __name__ == "__main__":
    main()
