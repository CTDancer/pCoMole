#!/usr/bin/env python3
"""
Script to construct a dataset of uniref50 sequences that matches the 
sequence length distribution found in a cas9 FASTA file.

This variation:
1. Filters uniref sequences to only include those between cas9 min and max length
2. Removes redundancy within the filtered sequences at 80% identity using mmseqs2
3. Uses the remaining sequences to match the cas9 length distribution

The script prioritizes matching the length distribution over matching 
a specified dataset size exactly.
"""

import argparse
import sys
import os
import subprocess
import tempfile
import shutil
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


def filter_and_write_fasta(input_fasta, output_fasta, min_length, max_length, description=""):
    """
    Filter sequences by length range and write to output FASTA file.
    
    Args:
        input_fasta: Input FASTA file path
        output_fasta: Output FASTA file path
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        description: Description for progress messages
    
    Returns:
        Number of sequences written
    """
    print(f"\nFiltering {description} sequences to length range [{min_length}, {max_length}]...")
    
    filtered_count = 0
    with open(output_fasta, 'w') as out_handle:
        for record in tqdm(SeqIO.parse(input_fasta, "fasta"), desc=f"Filtering {description}"):
            seq = str(record.seq).upper()
            # Skip empty sequences
            if not seq:
                continue
            # Filter out sequences with non-natural amino acids
            if not all(aa in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
                continue
            # Filter by length
            seq_len = len(seq)
            if min_length <= seq_len <= max_length:
                SeqIO.write(record, out_handle, "fasta")
                filtered_count += 1
    
    print(f"Filtered {filtered_count} sequences to length range [{min_length}, {max_length}]")
    return filtered_count


def remove_redundancy_mmseqs(input_fasta, output_fasta, identity_threshold=0.8, tmp_dir=None):
    """
    Remove redundancy using mmseqs2 clustering.
    
    Args:
        input_fasta: Input FASTA file
        output_fasta: Output FASTA file (representative sequences)
        identity_threshold: Sequence identity threshold (0.0-1.0)
        tmp_dir: Temporary directory for mmseqs2 files (if None, uses tempfile)
    
    Returns:
        Number of sequences before and after clustering
    """
    print(f"\n{'='*60}")
    print(f"Removing redundancy using mmseqs2 at {identity_threshold*100:.0f}% identity...")
    print(f"{'='*60}")
    
    # Count input sequences
    input_count = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    print(f"Input sequences: {input_count}")
    
    # Create temporary directory if not provided
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="mmseqs_")
        cleanup_tmp = True
    else:
        cleanup_tmp = False
        os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        db_path = os.path.join(tmp_dir, "db")
        cluster_path = os.path.join(tmp_dir, "cluster")
        cluster_rep_path = os.path.join(tmp_dir, "cluster_rep")
        
        # Step 1: Create database
        print("\nStep 1: Creating mmseqs2 database...")
        cmd = ['mmseqs', 'createdb', input_fasta, db_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Step 2: Cluster sequences
        print(f"\nStep 2: Clustering at {identity_threshold*100:.0f}% identity...")
        cmd = [
            'mmseqs', 'cluster',
            db_path,
            cluster_path,
            tmp_dir,
            '--min-seq-id', str(identity_threshold)
            # Note: mmseqs2 uses all available threads by default
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Step 3: Extract representative sequences
        print("\nStep 3: Extracting representative sequences...")
        cmd = ['mmseqs', 'createsubdb', cluster_path, db_path, cluster_rep_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Step 4: Convert to FASTA
        print("\nStep 4: Converting to FASTA...")
        cmd = ['mmseqs', 'convert2fasta', cluster_rep_path, output_fasta]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Count output sequences
        output_count = sum(1 for _ in SeqIO.parse(output_fasta, "fasta"))
        print(f"\nOutput sequences: {output_count}")
        print(f"Redundancy removed: {input_count - output_count} sequences ({100*(input_count-output_count)/input_count:.2f}%)")
        
        return input_count, output_count
        
    except subprocess.CalledProcessError as e:
        print(f"Error running mmseqs2: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("Error: mmseqs2 not found in PATH. Please install mmseqs2.", file=sys.stderr)
        raise
    finally:
        # Cleanup temporary directory
        if cleanup_tmp and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


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
        description="Match uniref50 sequence length distribution to cas9 distribution (with filtering and redundancy removal)"
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
        '--identity',
        type=float,
        default=0.8,
        help='Sequence identity threshold for redundancy removal (default: 0.8)'
    )
    parser.add_argument(
        '--tmp-dir',
        type=str,
        default=None,
        help='Temporary directory for intermediate files (default: auto)'
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
    
    # Create temporary directory for intermediate files
    if args.tmp_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="match_cas9_")
        cleanup_tmp = True
    else:
        tmp_dir = args.tmp_dir
        cleanup_tmp = False
        os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # Load cas9 sequences
        print("\n" + "="*60)
        print("Step 1: Loading cas9 sequences...")
        print("="*60)
        cas9_sequences, cas9_lengths = load_sequences(args.cas9_fasta, "cas9")
        if not cas9_sequences:
            print("Error: No valid cas9 sequences found!", file=sys.stderr)
            sys.exit(1)
        
        # Get cas9 length range
        cas9_min_length = min(cas9_lengths)
        cas9_max_length = max(cas9_lengths)
        print(f"\nCas9 length range: {cas9_min_length} - {cas9_max_length}")
        
        # Filter uniref sequences by length range
        print("\n" + "="*60)
        print("Step 2: Filtering uniref sequences by length range...")
        print("="*60)
        filtered_fasta = os.path.join(tmp_dir, "uniref_filtered.fasta")
        filtered_count = filter_and_write_fasta(
            args.uniref50_fasta,
            filtered_fasta,
            cas9_min_length,
            cas9_max_length,
            "uniref50"
        )
        
        if filtered_count == 0:
            print("Error: No uniref sequences found in the specified length range!", file=sys.stderr)
            sys.exit(1)
        
        # Remove redundancy from filtered sequences
        print("\n" + "="*60)
        print("Step 3: Removing redundancy from filtered sequences...")
        print("="*60)
        dedup_fasta = os.path.join(tmp_dir, "uniref_dedup.fasta")
        input_count, output_count = remove_redundancy_mmseqs(
            filtered_fasta,
            dedup_fasta,
            identity_threshold=args.identity,
            tmp_dir=os.path.join(tmp_dir, "mmseqs_tmp")
        )
        
        # Load deduplicated sequences
        print("\n" + "="*60)
        print("Step 4: Loading deduplicated sequences...")
        print("="*60)
        uniref50_sequences, uniref50_lengths = load_sequences(dedup_fasta, "uniref50 (deduplicated)")
        if not uniref50_sequences:
            print("Error: No valid deduplicated uniref50 sequences found!", file=sys.stderr)
            sys.exit(1)
        
        # Compute cas9 length distribution
        print("\n" + "="*60)
        print("Step 5: Analyzing cas9 length distribution...")
        print("="*60)
        cas9_bin_edges, cas9_counts, cas9_dist = compute_length_distribution(cas9_lengths)
        print(f"Cas9 distribution statistics:")
        print(f"  Total sequences: {len(cas9_sequences)}")
        print(f"  Number of bins: {len(cas9_dist)}")
        print(f"  Length range: {min(cas9_lengths)} - {max(cas9_lengths)}")
        
        # Sample uniref50 sequences to match cas9 distribution
        print("\n" + "="*60)
        print("Step 6: Sampling uniref50 sequences to match cas9 distribution...")
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
        
        # Compute distribution similarity (correlation)
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
        print("\n" + "="*60)
        print("Step 7: Writing final output...")
        print("="*60)
        write_fasta(sampled_sequences, args.output_fasta, "sampled uniref50")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Cas9 sequences: {len(cas9_sequences)}")
        print(f"Cas9 length range: {cas9_min_length} - {cas9_max_length}")
        print(f"Uniref sequences after length filtering: {filtered_count}")
        print(f"Uniref sequences after redundancy removal ({args.identity*100:.0f}%): {output_count}")
        print(f"Final sampled sequences: {len(sampled_sequences)}")
        print("="*60)
        
    finally:
        # Cleanup temporary directory
        if cleanup_tmp and os.path.exists(tmp_dir):
            print(f"\nCleaning up temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
