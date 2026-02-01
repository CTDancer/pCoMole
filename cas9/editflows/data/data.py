# Copyright (c) Meta Platforms, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple, Optional

import os

from datasets import Dataset as HFDataset, DatasetDict, load_dataset
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
try:
    from transformers import EsmTokenizer
    _HAVE_ESM = True
except Exception:
    _HAVE_ESM = False

from data.tokenizer import wt_detokenizer
from data.utils import cycle_loader, StatefulDistributedSampler


# -------------------------
# Tokenizer helper
# -------------------------

def _get_tokenizer(name: Optional[str]):
    """
    Returns a HF tokenizer by name. Supports ESM when available.
    """
    tok_name = name or "gpt2"
    if _HAVE_ESM and tok_name.startswith("facebook/esm"):
        return EsmTokenizer.from_pretrained(tok_name)
    return AutoTokenizer.from_pretrained(tok_name)


# -------------------------
# HuggingFace datasets (ragged)
# -------------------------

def _get_hf_dataset_ragged(
    name: str,
    mode: str,
    cache_dir: Optional[str],
    num_proc: int,
    tokenizer_name: Optional[str],
) -> HFDataset:
    """
    Loads an HF dataset split and tokenizes into variable-length sequences.
    No chunking to fixed block_size; each example is one tokenized text (+ EOS if available).
    Returns a HuggingFace Dataset with a single column "input_ids" (list[int]) per row.
    """
    detokenizer = None
    if name == "wikitext103":
        ds = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)[mode]
        detokenizer = wt_detokenizer
    elif name == "fineweb-edu":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", cache_dir=cache_dir)[mode]
    else:
        ds = load_dataset(name, cache_dir=cache_dir)[mode]

    tok = _get_tokenizer(tokenizer_name)
    eos_id = None
    if hasattr(tok, "eos_token") and tok.eos_token is not None:
        eos_id = tok.convert_tokens_to_ids(tok.eos_token)
        if isinstance(eos_id, list):
            eos_id = eos_id[0]

    def _apply_detok(batch: Dict):
        texts = batch["text"]
        if detokenizer is not None:
            texts = [detokenizer(t) for t in texts]
        return {"text": texts}

    if "text" in ds.column_names:
        ds = ds.map(_apply_detok, batched=True, num_proc=num_proc, load_from_cache_file=True)

    def _tokenize(batch: Dict):
        # Use fast tokenizer; do not return attention masks
        ids = tok(batch["text"], add_special_tokens=True, return_attention_mask=False)["input_ids"]
        # if eos_id is not None:
        #     for arr in ids:
        #         arr.append(eos_id)
        return {"input_ids": ids}

    # Tokenize
    tokenized = ds.map(
        _tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in ds.column_names if c != "text"],
        load_from_cache_file=True,
    )

    # Keep only input_ids
    if "text" in tokenized.column_names:
        tokenized = tokenized.remove_columns("text")

    # For fineweb there can be extra meta columns; keep only input_ids
    extra_cols = [c for c in tokenized.column_names if c != "input_ids"]
    if extra_cols:
        tokenized = tokenized.remove_columns(extra_cols)

    tokenized = tokenized.with_format("python")  # return plain python objects; collate will tensorize
    return tokenized


# -------------------------
# Local FASTA dataset (ragged)
# -------------------------

class FASTADataset(Dataset):
    """
    Loads a local FASTA file and tokenizes each sequence to variable-length ids.
    One HF-like item per sequence: {"input_ids": List[int]}.
    """

    def __init__(self, fasta_path: str, tokenizer_name: Optional[str]):
        assert os.path.isfile(fasta_path), f"FASTA file not found: {fasta_path}"
        self.fasta_path = fasta_path
        self.tok = _get_tokenizer(tokenizer_name)
        self.eos_id = None
        if hasattr(self.tok, "eos_token") and self.tok.eos_token is not None:
            eid = self.tok.convert_tokens_to_ids(self.tok.eos_token)
            self.eos_id = eid[0] if isinstance(eid, list) else eid

        # Parse FASTA (simple parser; no Biopython dependency)
        self._seqs: List[str] = []
        with open(fasta_path, "r") as f:
            cur = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        self._seqs.append("".join(cur))
                        cur = []
                else:
                    cur.append(line)
            if cur:
                self._seqs.append("".join(cur))

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        seq = self._seqs[idx]
        ids = self.tok(seq, add_special_tokens=True, return_attention_mask=False)["input_ids"]
        if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            # Some tokenizers may return nested lists for per-char tokenization; flatten
            ids = [t for sub in ids for t in (sub if isinstance(sub, list) else [sub])]
        # if self.eos_id is not None:
        #     ids.append(self.eos_id)
        return {"input_ids": ids}


# -------------------------
# Ragged collate
# -------------------------

def ragged_collate(batch: List[Dict[str, List[int]]]) -> Dict[str, List[torch.Tensor]]:
    """
    Collate a list of {"input_ids": List[int]} into {"input_ids": List[LongTensor]}.
    No padding; tensorization only.
    """
    out: List[torch.Tensor] = []
    for item in batch:
        ids = item["input_ids"]
        out.append(torch.tensor(ids, dtype=torch.long))
    return {"input_ids": out}


# -------------------------
# Public dataclasses
# -------------------------

@dataclass
class DatasetWrap:
    dataset: Dataset  # HF Dataset (python format) OR torch Dataset
    sampler: StatefulDistributedSampler


@dataclass
class DataState:
    train: DatasetWrap
    test: DatasetWrap


# -------------------------
# Builders
# -------------------------

def _get_dataset(
    name: str,
    mode: str,
    cache_dir: Optional[str],
    num_proc: int,
    batch_size: int,
    ngpus: int,
    tokenizer_name: Optional[str],
    fasta_path: Optional[str] = None,
) -> DatasetWrap:
    """
    Build either an HF dataset (ragged) or a local FASTA dataset.
    - name == "fasta" -> requires fasta_path
    - otherwise -> HF dataset by name
    """
    assert batch_size % ngpus == 0, f"{mode} batch size must be divisible by number of gpus."

    if name.lower() == "fasta":
        assert fasta_path is not None, "For name='fasta', you must set data.fasta_path in the config."
        dataset = FASTADataset(fasta_path=fasta_path, tokenizer_name=tokenizer_name)
    else:
        dataset = _get_hf_dataset_ragged(
            name=name,
            mode=mode,
            cache_dir=cache_dir,
            num_proc=num_proc,
            tokenizer_name=tokenizer_name,
        )

    sampler = StatefulDistributedSampler(dataset=dataset)
    return DatasetWrap(dataset=dataset, sampler=sampler)


def get_data_state(config: OmegaConf) -> DataState:
    """
    Expects (typical):
      config.data.train / config.data.valid : dataset name (e.g., "wikitext103", "fineweb-edu", "fasta", or any HF hub dataset id)
      config.data.cache_dir
      config.data.num_workers
      config.data.tokenizer (string, e.g., "gpt2" or "facebook/esm2_t33_650M_UR50S")
      config.data.fasta_path (optional, used when name == "fasta")
      config.training.batch_size
      config.eval.batch_size
      config.compute.ngpus
    """
    tok_name = getattr(config.data, "tokenizer", None)

    train = _get_dataset(
        name=config.data.train,
        mode="train",
        cache_dir=config.data.cache_dir,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        tokenizer_name=tok_name,
        fasta_path=getattr(config.data, "fasta_path", None),
    )

    valid = _get_dataset(
        name=config.data.valid,
        mode="validation",
        cache_dir=config.data.cache_dir,
        num_proc=config.data.num_workers,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
        tokenizer_name=tok_name,
        fasta_path=getattr(config.data, "fasta_path", None),
    )

    return DataState(train=train, test=valid)


def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    """
    Returns infinite iterators over train/valid using ragged_collate.
    Each batch item is a dict: {"input_ids": List[LongTensor]}.
    """
    per_gpu_train_bs = config.training.batch_size // config.compute.ngpus
    per_gpu_valid_bs = config.eval.batch_size // config.compute.ngpus

    train_loader = cycle_loader(
        DataLoader(
            data_state.train.dataset,
            batch_size=per_gpu_train_bs,
            sampler=data_state.train.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.train.sampler is None),
            persistent_workers=True,
            collate_fn=ragged_collate,
        )
    )

    valid_loader = cycle_loader(
        DataLoader(
            data_state.test.dataset,
            batch_size=per_gpu_valid_bs,
            sampler=data_state.test.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.test.sampler is None),
            collate_fn=ragged_collate,
        )
    )

    return iter(train_loader), iter(valid_loader)   
