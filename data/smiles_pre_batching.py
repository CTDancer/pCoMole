import os
import math
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

import sys
sys.path.append('/scratch/pranamlab/tong/cope/editflows')
from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
CSV_PATH = "/scratch/pranamlab/tong/data/smiles/28k_mimetics.csv"          # <- your csv
SMILES_COL = "Peptidomimetic_SMILES"
OUTPUT_DIR = "/scratch/pranamlab/tong/data/smiles/28k_mimetics"  # where to save with save_to_disk
MAX_TOKENS_PER_BATCH = 1024
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


# ------------------------------------------------------------
# 1) Load CSV and split 8:1:1
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
assert SMILES_COL in df.columns

df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
n = len(df)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_df = df.iloc[:n_train]
val_df   = df.iloc[n_train:n_train + n_val]
test_df  = df.iloc[n_train + n_val:]


# ------------------------------------------------------------
# 2) Tokenizer
# ------------------------------------------------------------
tokenizer = SMILES_SPE_Tokenizer('/scratch/pranamlab/tong/cope/editflows/smiles_tokenizer/new_vocab.txt',
                                 '/scratch/pranamlab/tong/cope/editflows/smiles_tokenizer/new_splits.txt')

def tokenize_smiles(smiles_list):
    enc = tokenizer(
        smiles_list,
        padding=False,
        truncation=False,
    )
    # enc["input_ids"] is a list of list[int]
    return enc


# ------------------------------------------------------------
# helper: build batches (one row = one batch)
# ------------------------------------------------------------
def build_batched_dataset(smiles_series, max_tokens=1024):
    # tokenize all first
    toks = tokenize_smiles(smiles_series.tolist())
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
train_ds = build_batched_dataset(train_df[SMILES_COL], MAX_TOKENS_PER_BATCH)
val_ds   = build_batched_dataset(val_df[SMILES_COL],   MAX_TOKENS_PER_BATCH)
test_ds  = build_batched_dataset(test_df[SMILES_COL],  MAX_TOKENS_PER_BATCH)

dsdict = DatasetDict({
    "train": train_ds,
    "validation": val_ds,
    "test": test_ds,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
dsdict.save_to_disk(OUTPUT_DIR)
print(f"saved to {OUTPUT_DIR}")