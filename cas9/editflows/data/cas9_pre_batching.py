import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict

# import selfies as sf
# from rdkit import Chem
# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

import sys

from transformers import EsmTokenizer

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
# Set PROJECT_ROOT or edit paths below
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', 'path/to/project')
FASTA_PATH = os.path.join(PROJECT_ROOT, 'path/to/protein2pam_db_v1.0/cas9_full.fasta')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data/cas9/cas9_dataset_esm2_tokenized_under_1125')

MAX_TOKENS_PER_BATCH = 8192
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

# ------------------------------------------------------------
# 1) Load sequences from FASTA file
# ------------------------------------------------------------
from Bio import SeqIO

sequences = []
print("Loading sequences from FASTA and validating…")
for record in SeqIO.parse(FASTA_PATH, "fasta"):
    s = str(record.seq)
    # skip empty sequences
    if not s:
        continue
    sequences.append(s)

# remove all duplicates 
sequences = list(set(sequences))

## filter out any sequence that contains non-natural amino acids
print(f"len(sequences) before filtering: {len(sequences)}")
sequences = [
    seq for seq in sequences
    if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq) and len(seq) <= 1125
]
print(f"len(sequences) after filtering: {len(sequences)}")
print(f"Average length of sequences: {sum(len(seq) for seq in sequences) / len(sequences)}")

# ---- Initialize ESM tokenizer ----
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
print(f"Vocabulary Size: {tokenizer.vocab_size}")

# shuffle before splitting
random.shuffle(sequences)
n = len(sequences)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_sequences = sequences[:n_train]
val_sequences   = sequences[n_train:n_train + n_val]
test_sequences  = sequences[n_train + n_val:]

# ------------------------------------------------------------
# 2) helper: tokenize sequences using the ESM tokenizer
# ------------------------------------------------------------
def tokenize_sequences(sequences):
    enc = {"input_ids": [], "attention_mask": []}
    for s in sequences:
        res = tokenizer(s, add_special_tokens=True)
        enc["input_ids"].append(res["input_ids"])
        enc["attention_mask"].append(res["attention_mask"])
    return enc


# ------------------------------------------------------------
# 3) build batched dataset (token-based batching: groups sequences by length, packs up to max_tokens)
# ------------------------------------------------------------
from tqdm import tqdm

def build_batched_dataset(sequences, max_tokens=8192):
    # tokenize all first
    toks = tokenize_sequences(sequences)
    input_ids_list = toks["input_ids"]
    attn_mask_list = toks["attention_mask"]

    # add lengths
    items = []
    for ids, mask in tqdm(zip(input_ids_list, attn_mask_list), total=len(input_ids_list), desc="Collecting items with length"):
        items.append({
                "input_ids": ids,
                "attention_mask": mask,
                "length": len(ids),
        })

    # sort by length
    items.sort(key=lambda x: x["length"])

    batched_input_ids = []
    batched_attention_masks = []
    batched_lengths = []
    batched_batch_sizes = []

    i = 0
    n = len(items)
    while i < n:
        L = items[i]["length"]
        # how many of length L can we pack?
        max_bs = max_tokens // L
        if max_bs < 1:
            max_bs = 1

        # collect up to max_bs items with same length
        cur_ids = []
        cur_masks = []
        taken = 0
        j = i
        while j < n and items[j]["length"] == L and taken < max_bs:
            cur_ids.append(items[j]["input_ids"])         # length L
            cur_masks.append(items[j]["attention_mask"])  # length L
            j += 1
            taken += 1

        # now this is one batch: shape (B, L)
        batched_input_ids.append(cur_ids)
        batched_attention_masks.append(cur_masks)
        batched_lengths.append(L)
        batched_batch_sizes.append(taken)

        i = j

    ds = Dataset.from_dict({
        "input_ids": batched_input_ids,           # list of (B, L)
            "attention_mask": batched_attention_masks,
        "seq_length": batched_lengths,
            "batch_size": batched_batch_sizes,
    })
    return ds


# ------------------------------------------------------------
# 4) build batched datasets for each split
# ------------------------------------------------------------
train_ds = build_batched_dataset(train_sequences, max_tokens=MAX_TOKENS_PER_BATCH)
val_ds   = build_batched_dataset(val_sequences,   max_tokens=MAX_TOKENS_PER_BATCH)
test_ds  = build_batched_dataset(test_sequences,  max_tokens=MAX_TOKENS_PER_BATCH)

dsdict = DatasetDict(
    {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
dsdict.save_to_disk(OUTPUT_DIR)
print(f"saved to {OUTPUT_DIR}")
