import pandas as pd
import random
from collections import Counter

# --------- Load sequences from CSV ----------
df = pd.read_csv("samples.csv")

# Try to automatically pick a sequence column; otherwise take the first column
candidate_cols = [c for c in df.columns if c.lower() in ["samples"]]
col = candidate_cols[0] if candidate_cols else df.columns[0]

seqs = df[col].dropna().astype(str).tolist()
N = len(seqs)
print(f"Loaded {N} sequences from column '{col}'")

# --------- Simple baseline: unique fraction ----------
unique_frac = len(set(seqs)) / max(1, N)
print("Unique fraction:", unique_frac)

# --------- Diversity 1: k-mer Jaccard (alignment-free) ----------
def kgrams(s: str, k: int):
    s = s.strip()
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i+k] for i in range(len(s) - k + 1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0

def diversity_kmer_jaccard(seqs, k=3, pairs=50000, seed=0):
    """
    Diversity = 1 - average Jaccard similarity over random pairs.
    Works for variable-length strings.
    """
    rng = random.Random(seed)
    grams = [kgrams(s, k) for s in seqs]
    n = len(seqs)

    total_sim = 0.0
    for _ in range(pairs):
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        total_sim += jaccard(grams[i], grams[j])

    avg_sim = total_sim / pairs
    return 1.0 - avg_sim, avg_sim

div, avg_sim = diversity_kmer_jaccard(seqs, k=3, pairs=50000, seed=0)
print(f"k-mer Jaccard avg similarity (k=3): {avg_sim:.4f}")
print(f"k-mer Jaccard diversity (1-avg_sim): {div:.4f}")

# --------- Optional Diversity 2: edit-distance similarity (slower) ----------
# If rapidfuzz is available, this is pretty fast and robust.
try:
    from rapidfuzz.distance import Levenshtein

    def diversity_levenshtein(seqs, pairs=20000, seed=0):
        rng = random.Random(seed)
        n = len(seqs)
        total_sim = 0.0
        for _ in range(pairs):
            i = rng.randrange(n)
            j = rng.randrange(n - 1)
            if j >= i:
                j += 1
            # normalized similarity in [0,1]
            sim = Levenshtein.normalized_similarity(seqs[i], seqs[j])
            total_sim += sim
        avg_sim = total_sim / pairs
        return 1.0 - avg_sim, avg_sim

    div2, avg_sim2 = diversity_levenshtein(seqs, pairs=20000, seed=0)
    print(f"Levenshtein avg similarity: {avg_sim2:.4f}")
    print(f"Levenshtein diversity: {div2:.4f}")

except Exception as e:
    print("rapidfuzz not available; skipping Levenshtein diversity.")
