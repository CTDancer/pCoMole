import os
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

MAX_TOKENS_PER_BATCH = 4096
SOURCE_DATASET = 'gfp'
MAX_LENGTH = 350
MIN_LENGTH = 150
MAX_DATA_SIZE = None
NUM_REPEATS = None

# Set PROJECT_ROOT or edit paths below
PROJECT_ROOT = os.environ.get('PROJECT_ROOT', 'path/to/project')
# Input directory containing train.fasta, val.fasta, test.fasta
if SOURCE_DATASET == 'gfp':
    INPUT_DIR = os.path.join(PROJECT_ROOT, 'data/gfp/fpbase_pfamPF01353_filtered')
    TRAIN_FASTA = os.path.join(INPUT_DIR, 'train.fasta')
    VAL_FASTA = os.path.join(INPUT_DIR, 'val.fasta')
    TEST_FASTA = os.path.join(INPUT_DIR, 'test.fasta')
else:
    raise ValueError(f"Invalid source dataset: {SOURCE_DATASET}")

# Output directory for tokenized and batched dataset
if MAX_DATA_SIZE is not None:
    if MAX_DATA_SIZE // 1000 > 0:
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, f'data/gfp/gfp_dataset_esm2_tokenized_mt{MAX_TOKENS_PER_BATCH}_leq{MAX_LENGTH}_n{MAX_DATA_SIZE//1000}k')
    else:
        OUTPUT_DIR = os.path.join(PROJECT_ROOT, f'data/gfp/gfp_dataset_esm2_tokenized_mt{MAX_TOKENS_PER_BATCH}_leq{MAX_LENGTH}_n{MAX_DATA_SIZE}')
else:
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, f'data/gfp/gfp_dataset_esm2_tokenized_mt{MAX_TOKENS_PER_BATCH}_leq{MAX_LENGTH}')
# ------------------------------------------------------------
# 1) Load sequences from pre-split FASTA files
# ------------------------------------------------------------
from Bio import SeqIO

def load_and_filter_sequences(fasta_path, split_name):
    """Load sequences from a FASTA file and filter them."""
    sequences = []
    print(f"Loading {split_name} sequences from {fasta_path}...")
    for record in SeqIO.parse(fasta_path, "fasta"):
        s = str(record.seq)
        # skip empty sequences
        if not s:
            continue
        # Replace X with A
        s = s.replace('X', 'A')
        sequences.append(s)
    
    # Remove duplicates within this split
    sequences = list(set(sequences))
    
    # Filter out any sequence that contains non-natural amino acids
    print(f"len({split_name}) before filtering: {len(sequences)}")
    sequences = [
        seq for seq in sequences
        if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq)
        and (MAX_LENGTH is None or len(seq) <= MAX_LENGTH)
        and (MIN_LENGTH is None or len(seq) >= MIN_LENGTH)
    ]
    print(f"len({split_name}) after filtering: {len(sequences)}")
    if sequences:
        print(f"Average length of {split_name} sequences: {sum(len(seq) for seq in sequences) / len(sequences):.1f}")
    
    return sequences

# Load sequences from each split
train_sequences = load_and_filter_sequences(TRAIN_FASTA, "train")
val_sequences = load_and_filter_sequences(VAL_FASTA, "val")
test_sequences = load_and_filter_sequences(TEST_FASTA, "test")

# Apply MAX_DATA_SIZE truncation if specified (applies to each split independently)
if MAX_DATA_SIZE is not None:
    print(f"Truncating each split to {MAX_DATA_SIZE} sequences")
    train_sequences = train_sequences[:MAX_DATA_SIZE]
    val_sequences = val_sequences[:MAX_DATA_SIZE]
    test_sequences = test_sequences[:MAX_DATA_SIZE]
    print(f"len(train_sequences) after truncation: {len(train_sequences)}")
    print(f"len(val_sequences) after truncation: {len(val_sequences)}")
    print(f"len(test_sequences) after truncation: {len(test_sequences)}")

# Apply NUM_REPEATS if specified
if NUM_REPEATS is not None:
    print(f"Repeating each split {NUM_REPEATS} times")
    train_sequences = train_sequences * NUM_REPEATS
    val_sequences = val_sequences * NUM_REPEATS
    test_sequences = test_sequences * NUM_REPEATS

print(f"Final len(train_sequences): {len(train_sequences)}")
print(f"Final len(val_sequences): {len(val_sequences)}")
print(f"Final len(test_sequences): {len(test_sequences)}")

# ---- Initialize ESM tokenizer ----
tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
print(f"Vocabulary Size: {tokenizer.vocab_size}")
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

# Print number of sequences in each split after batching
print("\n" + "=" * 60)
print("Sequences per split after batching:")
print("=" * 60)
for split_name, ds in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
    num_seqs = 0
    for item in ds:
        num_seqs += len(item["input_ids"])
    print(f"  {split_name:12s}: {num_seqs:>6} sequences")
print("=" * 60 + "\n")

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
