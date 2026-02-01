import os
import random
import pandas as pd
from datasets import Dataset, DatasetDict

import selfies as sf
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from smiles_tokenizer.selfies_tokenizers import SelfiesTokenizer

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', 'path/to/project')
CSV_PATH = os.path.join(PROJECT_ROOT, "data/smiles/28k_mimetics.csv")
SMILES_COL = "Peptidomimetic_SMILES"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/selfies/28k_mimetics")

BATCH_SIZE = 64
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


# ------------------------------------------------------------
# 1) Load CSV and convert SMILES -> SELFIES
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
assert SMILES_COL in df.columns

selfies_list = []
print("Validating SMILES and encoding to SELFIES…")
for s in df[SMILES_COL].tolist():
    if Chem.MolFromSmiles(s) is None:
        continue
    try:
        selfies_list.append(sf.encoder(s))
    except Exception:
        continue

# ---- Build vocab from all valid SELFIES ----
tokenizer = SelfiesTokenizer()
tokenizer.build_vocab(selfies_list, add_semantic_default=False)
tokenizer.save(os.path.join(OUTPUT_DIR, 'tokenizer'), constraints=None)
print(f"Vocabulary Size: {tokenizer.vocab_size}")

PAD_ID = 0

# shuffle before splitting
random.shuffle(selfies_list)
n = len(selfies_list)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_df = selfies_list[:n_train]
val_df   = selfies_list[n_train:n_train + n_val]
test_df  = selfies_list[n_train + n_val:]


# ------------------------------------------------------------
# 2) helper: tokenize SELFIES using the built tokenizer
# ------------------------------------------------------------
def tokenize_selfies(selfies_seq_list):
    enc = {"input_ids": [], "attention_mask": []}
    for s in selfies_seq_list:
        # we already have SELFIES strings
        res = tokenizer.encode(s, already_selfies=True, add_bos_eos=True)
        enc["input_ids"].append(res["input_ids"])
        enc["attention_mask"].append(res["attention_mask"])
    return enc


# ------------------------------------------------------------
# 3) build batched dataset (sorted by length, fixed B=64, pad in-batch)
# ------------------------------------------------------------
def build_batched_dataset(selfies_seq_list, batch_size=64, pad_id=0):
    toks = tokenize_selfies(selfies_seq_list)
    input_ids_list = toks["input_ids"]
    attn_mask_list = toks["attention_mask"]

    # collect items with length
    items = []
    for ids, mask in zip(input_ids_list, attn_mask_list):
        items.append(
            {
                "input_ids": ids,
                "attention_mask": mask,
                "length": len(ids),
            }
        )

    # sort globally by length (ascending)
    items.sort(key=lambda x: x["length"])

    batched_input_ids = []
    batched_attention_masks = []
    batched_seq_lengths = []
    batched_batch_sizes = []

    n = len(items)
    for start in range(0, n, batch_size):
        batch_items = items[start:start + batch_size]

        # max length in this batch
        max_len_in_batch = max(it["length"] for it in batch_items)

        cur_ids = []
        cur_masks = []

        for it in batch_items:
            ids = it["input_ids"]
            mask = it["attention_mask"]

            # pad to max_len_in_batch
            pad_len = max_len_in_batch - len(ids)
            if pad_len > 0:
                ids = ids + [pad_id] * pad_len
                mask = mask + [0] * pad_len

            cur_ids.append(ids)    # now len == max_len_in_batch
            cur_masks.append(mask) # now len == max_len_in_batch

        batched_input_ids.append(cur_ids)               # (B, max_len_in_batch)
        batched_attention_masks.append(cur_masks)       # (B, max_len_in_batch)
        batched_seq_lengths.append(max_len_in_batch)    # store the padded length
        batched_batch_sizes.append(len(batch_items))    # <= batch_size

    ds = Dataset.from_dict(
        {
            "input_ids": batched_input_ids,          # list of (B, Lpadded)
            "attention_mask": batched_attention_masks,
            "seq_length": batched_seq_lengths,       # Lpadded
            "batch_size": batched_batch_sizes,
        }
    )
    return ds


# ------------------------------------------------------------
# 4) build batched datasets for each split
# ------------------------------------------------------------
train_ds = build_batched_dataset(train_df, batch_size=BATCH_SIZE, pad_id=PAD_ID)
val_ds   = build_batched_dataset(val_df,   batch_size=BATCH_SIZE, pad_id=PAD_ID)
test_ds  = build_batched_dataset(test_df,  batch_size=BATCH_SIZE, pad_id=PAD_ID)

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
