#!/usr/bin/env python3
"""
Script to construct a dataset of uniref50 sequences that matches the 
sequence length distribution found in a cas9 FASTA file.

The script prioritizes matching the length distribution over matching 
a specified dataset size exactly.
"""

import argparse
import sys
from collections import Counter, defaultdict
from Bio import SeqIO
import numpy as np
from tqdm import tqdm


def load_sequences(fasta_path, description=""):
    """Load sequences from a FASTA file."""
    sequences = []
    lengths = []
    print(f"Loading {description} sequences from {fasta_path}...")
    
    for record in tqdm(SeqIO.parse(fasta_path, "fasta"), desc=f"Reading {description}"):
        seq = str(record.seq).upper()
        # Skip empty sequences
        if not seq:
            continue
        # Filter out sequences with non-natural amino acids
        if all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
            sequences.append(seq)
            lengths.append(len(seq))
    
    print(f"Loaded {len(sequences)} {description} sequences")
    if sequences:
        print(f"  Length range: {min(lengths)} - {max(lengths)}")
        print(f"  Average length: {sum(lengths) / len(lengths):.1f}")
    
    return sequences, lengths


def compute_length_distribution(lengths, bins=None):
    """
    Compute the distribution of sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        bins: Optional list of bin edges. If None, uses histogram bins.
    
    Returns:
        bin_edges: List of bin edges
        counts: List of counts for each bin
        normalized_dist: Normalized distribution (probabilities)
    """
    if bins is None:
        # Use histogram to automatically determine bins
        counts, bin_edges = np.histogram(lengths, bins='auto')
    else:
        counts, bin_edges = np.histogram(lengths, bins=bins)
    
    # Normalize to get probability distribution
    total = sum(counts)
    normalized_dist = counts / total if total > 0 else counts
    
    return bin_edges, counts, normalized_dist


def group_sequences_by_length(sequences):
    """Group sequences by their length."""
    length_groups = defaultdict(list)
    for seq in sequences:
        length_groups[len(seq)].append(seq)
    return length_groups


def sample_to_match_distribution(
    source_sequences, 
    source_lengths,
    target_lengths,
    target_size=None,
    bin_edges=None
):
    """
    Sample sequences from source_sequences to match the length distribution 
    of target_lengths.
    
    Args:
        source_sequences: List of sequences to sample from
        source_lengths: List of lengths corresponding to source_sequences
        target_lengths: List of lengths to match the distribution of
        target_size: Optional target number of sequences (approximate)
        bin_edges: Optional bin edges for histogram (if None, computed from target)
    
    Returns:
        sampled_sequences: List of sampled sequences
        sampled_lengths: List of lengths of sampled sequences
    """
    # Compute target distribution
    if bin_edges is None:
        target_bin_edges, target_counts, target_dist = compute_length_distribution(target_lengths)
    else:
        _, target_counts, target_dist = compute_length_distribution(target_lengths, bins=bin_edges)
        target_bin_edges = bin_edges
    
    print(f"\nTarget distribution:")
    print(f"  Number of bins: {len(target_dist)}")
    print(f"  Bin edges: {target_bin_edges[:5]}...{target_bin_edges[-5:]}")
    
    # Group source sequences by length
    print("\nGrouping source sequences by length...")
    length_groups = group_sequences_by_length(source_sequences)
    print(f"  Found {len(length_groups)} unique lengths")
    
    # Determine how many sequences to sample per bin
    if target_size is None:
        # Use the same counts as target
        target_counts_scaled = target_counts.copy()
    else:
        # Scale the distribution to approximately match target_size
        total_target = sum(target_counts)
        scale_factor = target_size / total_target if total_target > 0 else 1.0
        target_counts_scaled = (target_counts * scale_factor).astype(int)
        # Ensure we have at least 1 sequence per non-zero bin
        target_counts_scaled = np.maximum(target_counts_scaled, (target_counts > 0).astype(int))
    
    print(f"\nTarget sample size: {sum(target_counts_scaled)} sequences")
    
    # Sample sequences for each bin
    sampled_sequences = []
    sampled_lengths = []
    
    print("\nSampling sequences to match distribution...")
    for i in tqdm(range(len(target_dist)), desc="Processing bins"):
        bin_start = int(target_bin_edges[i])
        bin_end = int(target_bin_edges[i + 1])
        n_samples = int(target_counts_scaled[i])
        
        if n_samples == 0:
            continue
        
        # Find all source sequences in this length range
        candidates = []
        for length in range(bin_start, bin_end + 1):
            if length in length_groups:
                candidates.extend(length_groups[length])
        
        if not candidates:
            # If no exact match, try to find sequences close to this bin
            # Find the closest available lengths
            available_lengths = sorted(length_groups.keys())
            for length in available_lengths:
                if bin_start <= length <= bin_end:
                    candidates.extend(length_groups[length])
                    break
        
        if not candidates:
            # Still no candidates, try to find sequences in nearby bins
            bin_center = (bin_start + bin_end) / 2
            closest_length = min(available_lengths, key=lambda x: abs(x - bin_center))
            candidates.extend(length_groups[closest_length])
        
        # Sample from candidates
        if len(candidates) >= n_samples:
            sampled = np.random.choice(candidates, size=n_samples, replace=False).tolist()
        else:
            # Not enough candidates, sample with replacement
            sampled = np.random.choice(candidates, size=n_samples, replace=True).tolist()
        
        sampled_sequences.extend(sampled)
        sampled_lengths.extend([len(seq) for seq in sampled])
    
    return sampled_sequences, sampled_lengths


def write_fasta(sequences, output_path, description="sequences"):
    """Write sequences to a FASTA file."""
    print(f"\nWriting {len(sequences)} {description} to {output_path}...")
    with open(output_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">sequence_{i+1}\n")
            f.write(f"{seq}\n")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Match uniref50 sequence length distribution to cas9 distribution"
    )
    parser.add_argument(
        'cas9_fasta',
        type=str,
        help='Path to FASTA file containing cas9 sequences'
    )
    parser.add_argument(
        'uniref50_fasta',
        type=str,
        help='Path to FASTA file containing uniref50 sequences'
    )
    parser.add_argument(
        'output_fasta',
        type=str,
        help='Path to output FASTA file'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=None,
        help='Approximate target dataset size (distribution matching takes precedence)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load cas9 sequences
    cas9_sequences, cas9_lengths = load_sequences(args.cas9_fasta, "cas9")
    if not cas9_sequences:
        print("Error: No valid cas9 sequences found!", file=sys.stderr)
        sys.exit(1)
    
    # Load uniref50 sequences
    uniref50_sequences, uniref50_lengths = load_sequences(args.uniref50_fasta, "uniref50")
    if not uniref50_sequences:
        print("Error: No valid uniref50 sequences found!", file=sys.stderr)
        sys.exit(1)
    
    # Compute cas9 length distribution
    print("\n" + "="*60)
    print("Analyzing cas9 length distribution...")
    print("="*60)
    cas9_bin_edges, cas9_counts, cas9_dist = compute_length_distribution(cas9_lengths)
    print(f"Cas9 distribution statistics:")
    print(f"  Total sequences: {len(cas9_sequences)}")
    print(f"  Number of bins: {len(cas9_dist)}")
    print(f"  Length range: {min(cas9_lengths)} - {max(cas9_lengths)}")
    
    # Sample uniref50 sequences to match cas9 distribution
    print("\n" + "="*60)
    print("Sampling uniref50 sequences to match cas9 distribution...")
    print("="*60)
    sampled_sequences, sampled_lengths = sample_to_match_distribution(
        uniref50_sequences,
        uniref50_lengths,
        cas9_lengths,
        target_size=args.target_size,
        bin_edges=cas9_bin_edges
    )
    
    # Compare distributions
    print("\n" + "="*60)
    print("Distribution comparison:")
    print("="*60)
    sampled_bin_edges, sampled_counts, sampled_dist = compute_length_distribution(
        sampled_lengths, bins=cas9_bin_edges
    )
    
    print(f"\nCas9 distribution:")
    print(f"  Total sequences: {len(cas9_sequences)}")
    print(f"  Mean length: {np.mean(cas9_lengths):.1f}")
    print(f"  Std length: {np.std(cas9_lengths):.1f}")
    
    print(f"\nSampled uniref50 distribution:")
    print(f"  Total sequences: {len(sampled_sequences)}")
    print(f"  Mean length: {np.mean(sampled_lengths):.1f}")
    print(f"  Std length: {np.std(sampled_lengths):.1f}")
    
    # Compute distribution similarity (KL divergence or correlation)
    # Remove zero bins for comparison
    mask = (cas9_counts > 0) & (sampled_counts > 0)
    if mask.sum() > 0:
        cas9_dist_filtered = cas9_dist[mask]
        sampled_dist_filtered = sampled_dist[mask]
        # Normalize filtered distributions
        cas9_dist_filtered = cas9_dist_filtered / cas9_dist_filtered.sum()
        sampled_dist_filtered = sampled_dist_filtered / sampled_dist_filtered.sum()
        
        # Compute correlation
        correlation = np.corrcoef(cas9_dist_filtered, sampled_dist_filtered)[0, 1]
        print(f"\nDistribution correlation: {correlation:.4f}")
    
    # Write output
    write_fasta(sampled_sequences, args.output_fasta, "sampled uniref50")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
