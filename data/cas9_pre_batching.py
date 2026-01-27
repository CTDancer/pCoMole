import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import EsmTokenizer

import selfies as sf
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import sys
sys.path.append('/scratch/cope/editflows')
from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from smiles_tokenizer.selfies_tokenizers import SelfiesTokenizer

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CSV_PATH = "/scratch/data/cas9/cas9.csv"          # <- your csv
OUTPUT_DIR = "/scratch/data/cas9/"  # where to save with save_to_disk
MAX_TOKENS_PER_BATCH = 8192
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


# ------------------------------------------------------------
# 1) Load CSV and split 8:1:1
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
sequences = df["Sequence"].tolist()

# ---- Build vocab ----
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

random.shuffle(sequences)
n = len(sequences)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_df = sequences[:n_train]
val_df   = sequences[n_train:n_train + n_val]
test_df  = sequences[n_train + n_val:]


# ------------------------------------------------------------
# 2) Tokenizer
# ------------------------------------------------------------

def tokenize_selfies(sequences):
    enc = {"input_ids": [], "attention_mask": []}
    for s in sequences:
        res = tokenizer(s)
        enc["input_ids"].append(res["input_ids"])
        enc["attention_mask"].append(res["attention_mask"])
    # enc["input_ids"] is a list of list[int]
    return enc


# ------------------------------------------------------------
# helper: build batches (one row = one batch)
# ------------------------------------------------------------
def build_batched_dataset(sequences, max_tokens=9182):
    # tokenize all first
    toks = tokenize_selfies(sequences)
    input_ids_list = toks["input_ids"]
    attn_mask_list = toks["attention_mask"]

    # add lengths
    items = []
    for ids, mask in zip(input_ids_list, attn_mask_list):
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
# 3) build batched datasets for each split
# ------------------------------------------------------------
train_ds = build_batched_dataset(train_df, MAX_TOKENS_PER_BATCH)
val_ds   = build_batched_dataset(val_df,   MAX_TOKENS_PER_BATCH)
test_ds  = build_batched_dataset(test_df,  MAX_TOKENS_PER_BATCH)

dsdict = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
dsdict.save_to_disk(OUTPUT_DIR)
print(f"saved to {OUTPUT_DIR}")
