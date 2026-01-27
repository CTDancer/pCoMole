#!/usr/bin/env python3
import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import EsmModel, EsmTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
except ImportError:
    raise ImportError("Please `pip install scikit-learn` in your environment.")


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(sec: float) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h}h{m}m{s}s"
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


# ----------------------------
# Dataset / Collator
# ----------------------------
class CSVGFPDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        assert "Sequence" in df.columns and "Label" in df.columns
        self.seqs = df["Sequence"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"sequence": self.seqs[idx], "label": self.labels[idx]}


@dataclass
class ESMCollator:
    tokenizer: EsmTokenizer
    max_length: int = 1024  # includes special tokens; ESM2 typical max is 1022 tokens + specials

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seqs = [b["sequence"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)

        # ESM tokenizer expects protein sequences as strings
        tok = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        tok["labels"] = labels
        return tok


# ----------------------------
# Model
# ----------------------------
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

        # cache special ids for pooling mask
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def _esm_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Return last_hidden_state (B, L, H)
        out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def _mean_pool(self, hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pool over non-special, non-pad tokens.
        hidden: (B, L, H)
        """
        # base mask from attention_mask
        mask = attention_mask.bool()

        # exclude special tokens
        special = torch.zeros_like(mask)
        if self.pad_id is not None:
            special |= (input_ids == self.pad_id)
        if self.cls_id is not None:
            special |= (input_ids == self.cls_id)
        if self.eos_id is not None:
            special |= (input_ids == self.eos_id)

        mask = mask & (~special)

        # If a sequence becomes empty after masking, fall back to CLS token embedding
        lengths = mask.sum(dim=1)  # (B,)
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1)  # (B, H)
        denom = lengths.clamp(min=1).unsqueeze(-1)
        pooled = pooled / denom

        # fallback for empty: replace pooled with CLS embedding where lengths==0
        if self.cls_id is not None:
            empty = (lengths == 0)
            if empty.any():
                cls_emb = hidden[:, 0, :]  # CLS at position 0
                pooled = torch.where(empty.unsqueeze(-1), cls_emb, pooled)

        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # compute ESM embeddings (frozen)
        hidden = self._esm_forward(input_ids, attention_mask)  # (B, L, H)
        pooled = self._mean_pool(hidden, input_ids, attention_mask)  # (B, H)
        logit = self.mlp(pooled).squeeze(-1)  # (B,)
        return logit


# ----------------------------
# LR Scheduler: 0.1*lr -> warmup to lr (10%) -> cosine back to 0.1*lr
# ----------------------------
def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_ratio: float = 0.10,
    min_lr_ratio: float = 0.10,
) -> LambdaLR:
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        # factor to multiply base lr
        if step < warmup_steps:
            # linear warmup from min_lr_ratio -> 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * (step / float(warmup_steps))
        # cosine decay from 1.0 -> min_lr_ratio
        progress = (step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_p = []
    all_logit = []
    total_loss = 0.0
    n = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")

        probs = torch.sigmoid(logits)

        total_loss += float(loss) * labels.size(0)
        n += labels.size(0)

        all_y.append(labels.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())
        all_logit.append(logits.detach().cpu().numpy())

    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    # logit = np.concatenate(all_logit)

    # handle edge cases where a split contains only one class
    metrics = {"loss": total_loss / max(1, n)}
    if len(np.unique(y)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y, p))
        metrics["auprc"] = float(average_precision_score(y, p))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    pred = (p >= 0.5).astype(np.int32)
    metrics["acc"] = float(accuracy_score(y, pred))
    metrics["f1"] = float(f1_score(y, pred, zero_division=0))
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
    pos_weight: torch.Tensor,
    grad_clip: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()

    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    total_loss = 0.0
    n = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight, reduction="mean"
            )

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += float(loss) * labels.size(0)
        n += labels.size(0)

    return {"train_loss": total_loss / max(1, n)}


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str, default="/scratch/pranamlab/tong/data/gfp/classifier_total_data.csv")
    ap.add_argument("--out_dir", type=str, default="/scratch/pranamlab/tong/cope/editflows/gfp/classifier_ckpt")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--esm_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.10)
    ap.add_argument("--min_lr_ratio", type=float, default=0.10)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--mlp_hidden", type=int, default=512)
    ap.add_argument("--mlp_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)

    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)

    ap.add_argument("--amp", action="store_true", help="Use CUDA AMP (fp16) mixed precision.")
    ap.add_argument("--metric_for_best", type=str, default="auroc", choices=["auroc", "auprc", "f1", "acc"])

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device={device}")

    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["Sequence", "Label"]).reset_index(drop=True)

    # Stratified split: train / val / test
    y = df["Label"].astype(int).values
    df_train, df_tmp = train_test_split(
        df, test_size=(args.val_ratio + args.test_ratio), random_state=args.seed, stratify=y
    )
    y_tmp = df_tmp["Label"].astype(int).values
    val_size = args.val_ratio / (args.val_ratio + args.test_ratio)
    df_val, df_test = train_test_split(
        df_tmp, test_size=(1.0 - val_size), random_state=args.seed, stratify=y_tmp
    )

    print(f"[Split] train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    print(f"[Split] train pos={df_train['Label'].sum()} neg={len(df_train)-df_train['Label'].sum()}")

    # pos_weight = n_neg / n_pos for BCEWithLogitsLoss
    n_pos = int(df_train["Label"].sum())
    n_neg = int(len(df_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    print(f"[Info] pos_weight={pos_weight.item():.4f}")

    # Model
    model = GFPClassifier(
        esm_name=args.esm_name,
        mlp_hidden=args.mlp_hidden,
        mlp_layers=args.mlp_layers,
        dropout=args.dropout,
    ).to(device)

    # DataLoaders
    collator = ESMCollator(tokenizer=model.tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        CSVGFPDataset(df_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        CSVGFPDataset(df_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        CSVGFPDataset(df_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )

    # Optim / Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    num_training_steps = args.epochs * len(train_loader)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        num_training_steps=num_training_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
    )

    # Save config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best_metric = -1e9
    best_path = os.path.join(args.out_dir, "best.pt")

    t0 = time.time()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pos_weight=pos_weight,
            grad_clip=args.grad_clip,
            use_amp=args.amp,
        )
        global_step += len(train_loader)

        val_stats = evaluate(model, val_loader, device)

        metric = val_stats.get(args.metric_for_best, float("nan"))
        if not (isinstance(metric, float) and math.isnan(metric)) and metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "metric_name": args.metric_for_best,
                },
                best_path,
            )

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"lr={lr_now:.3e} "
            f"train_loss={train_stats['train_loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_auroc={val_stats['auroc']:.4f} "
            f"val_auprc={val_stats['auprc']:.4f} "
            f"val_f1={val_stats['f1']:.4f} "
            f"val_acc={val_stats['acc']:.4f} "
            f"best_{args.metric_for_best}={best_metric:.4f} "
            f"time={format_seconds(time.time()-ep_t0)}"
        )

    print(f"[Done] total_time={format_seconds(time.time()-t0)} best_ckpt={best_path}")

    # Final test with best checkpoint
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_stats = evaluate(model, test_loader, device)
        print(
            f"[Test @ best] loss={test_stats['loss']:.4f} "
            f"auroc={test_stats['auroc']:.4f} "
            f"auprc={test_stats['auprc']:.4f} "
            f"f1={test_stats['f1']:.4f} "
            f"acc={test_stats['acc']:.4f}"
        )
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_stats, f, indent=2)


if __name__ == "__main__":
    main()
