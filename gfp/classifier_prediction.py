#!/usr/bin/env python3
import os
import json
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import EsmModel, EsmTokenizer


class GFPClassifier(nn.Module):
    def __init__(
        self,
        esm_name: str = "facebook/esm2_t33_650M_UR50D",
        mlp_hidden: int = 512,
        mlp_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(esm_name)
        self.esm = EsmModel.from_pretrained(esm_name)

        # Freeze ESM
        for p in self.esm.parameters():
            p.requires_grad = False

        emb_dim = self.esm.config.hidden_size

        layers: List[nn.Module] = []
        in_dim = emb_dim
        for _ in range(mlp_layers):
            layers.append(nn.Linear(in_dim, mlp_hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_dim = mlp_hidden
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def _esm_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, L, H)

    def _mean_pool(self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.bool()

        special = torch.zeros_like(mask)
        if self.pad_id is not None:
            special |= (input_ids == self.pad_id)
        if self.cls_id is not None:
            special |= (input_ids == self.cls_id)
        if self.eos_id is not None:
            special |= (input_ids == self.eos_id)

        mask = mask & (~special)

        lengths = mask.sum(dim=1)  # (B,)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1)  # (B, H)
        pooled = pooled / lengths.clamp(min=1).unsqueeze(-1)

        if self.cls_id is not None:
            empty = (lengths == 0)
            if empty.any():
                pooled = torch.where(empty.unsqueeze(-1), hidden[:, 0, :], pooled)

        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self._esm_forward(input_ids, attention_mask)
        pooled = self._mean_pool(hidden, input_ids, attention_mask)
        logit = self.mlp(pooled).squeeze(-1)
        return logit


def read_fasta(path: str) -> List[Tuple[str, str]]:
    records = []
    header = None
    seq_chunks = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


@torch.no_grad()
def predict_sequences(
    model: GFPClassifier,
    sequences: List[str],
    device: torch.device,
    max_length: int = 1024,
    batch_size: int = 8,
) -> List[float]:
    model.eval()
    probs_all: List[float] = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        tok = model.tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits).detach().cpu().tolist()
        probs_all.extend(probs)

    return probs_all


def load_model_from_ckpt(ckpt_path: str, device: torch.device) -> GFPClassifier:
    # We saved training args to config.json in out_dir; use it if available.
    out_dir = os.path.dirname(ckpt_path)
    cfg_path = os.path.join(out_dir, "config.json")

    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        model = GFPClassifier(
            esm_name=cfg.get("esm_name", "facebook/esm2_t33_650M_UR50D"),
            mlp_hidden=int(cfg.get("mlp_hidden", 512)),
            mlp_layers=int(cfg.get("mlp_layers", 2)),
            dropout=float(cfg.get("dropout", 0.2)),
        )
        max_length = int(cfg.get("max_length", 1024))
    else:
        # fallback to defaults
        model = GFPClassifier()
        max_length = 1024

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    return model, max_length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default='/scratch/pranamlab/tong/cope/editflows/gfp/classifier_ckpt/best.pt', help="Path to best.pt")
    ap.add_argument("--seq", type=str, default=None, help="Single protein sequence string")
    ap.add_argument("--fasta", type=str, default=None, help="FASTA file path (multiple sequences)")
    ap.add_argument("--txt", type=str, default=None, help="Text file: one sequence per line")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for label")
    args = ap.parse_args()

    if (args.seq is None) and (args.fasta is None) and (args.txt is None):
        raise ValueError("Provide one of: --seq, --fasta, --txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, max_length = load_model_from_ckpt(args.ckpt, device)

    names: List[str] = []
    seqs: List[str] = []

    if args.seq is not None:
        names = ["query"]
        seqs = [args.seq.strip()]
    elif args.fasta is not None:
        recs = read_fasta(args.fasta)
        names = [h for h, _ in recs]
        seqs = [s for _, s in recs]
    elif args.txt is not None:
        with open(args.txt, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        names = [f"seq_{i}" for i in range(len(lines))]
        seqs = lines

    probs = predict_sequences(
        model=model,
        sequences=seqs,
        device=device,
        max_length=max_length,
        batch_size=args.batch_size,
    )

    # Print results
    for name, seq, p in zip(names, seqs, probs):
        pred = int(p >= args.threshold)
        print(f">{name}  prob_GFP={p:.6f}  pred={pred}  len={len(seq)}")


if __name__ == "__main__":
    main()
