# constraints.py

import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
# Make selfies optional (only needed for SMILES constraints, not protein constraints)
try:
    import selfies as sf
except ImportError:
    sf = None

# from .is_peptidomimetic import is_peptidomimetic_not_natural
import pdb

# ---- Optional import: Cas9 domain detector (HNH + RuvC) ----
# Expects cope_domain_detector.py to live in the same package as this constraints.py
try:
    from .cope_domain_detector import DomainDetector, DomainDetectorConfig, HitCriteria
except Exception as e:
    import traceback
    print(f"Warning: Failed to import DomainDetector: {e}")
    traceback.print_exc()
    DomainDetector = None
    DomainDetectorConfig = None
    HitCriteria = None


def detokenize_output(x, cfg, tokenizer, bos_id, eos_id, pad_id):
    """
    Convert a single generated sequence (1, L) back to string.
    """
    seq = x[0].tolist()
    # strip padding
    seq = [tok for tok in seq if tok != pad_id]
    # strip BOS/EOS
    if len(seq) > 0 and seq[0] == bos_id:
        seq = seq[1:]
    if len(seq) > 0 and seq[-1] == eos_id:
        seq = seq[:-1]

    if cfg.task == 'protein':
        # esm tokenizer has batch_decode
        return tokenizer.batch_decode([seq], skip_special_tokens=True)[0]
    elif cfg.task in ('smiles', 'selfies'):
        return tokenizer.decode(seq)
    else:
        return " ".join(map(str, seq))


class Peptidomimetic:
    def __init__(self):
        pass

    def __call__(self, smiles_tokens, smiles_seq):
        flag, audit = is_peptidomimetic_not_natural(smiles_seq)
        return 1 if flag else 0


# SMARTS
_AMIDE       = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")               # amide C(=O)-N (includes carbamates/ureas as centers)
_SULFONAMIDE = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[NX3]")       # sulfonamide S(=O)2-N  (optional isostere)


def _centers_and_N(mol, include_sulfonamide=True):
    centers, n2c = set(), {}
    for c_idx, _, n_idx in mol.GetSubstructMatches(_AMIDE):
        centers.add(c_idx)
        n2c.setdefault(n_idx, set()).add(c_idx)
    if include_sulfonamide:
        for s_idx, _, _, n_idx in mol.GetSubstructMatches(_SULFONAMIDE):
            centers.add(s_idx)
            n2c.setdefault(n_idx, set()).add(s_idx)
    return centers, n2c


def _is_sp3_carbon(atom):
    return atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3


def _backbone_edges(mol, centers, n2c):
    """
    Edge between centers i--j exists if there is a single-bond path:
      center_i — N — C(sp3) — center_j    (length 3)
      or center_i — N — (O|S|N) — C(sp3) — center_j  (length 4)
    """
    edges = set()
    for n_idx, left_centers in n2c.items():
        n = mol.GetAtomWithIdx(n_idx)
        for b1 in n.GetBonds():
            if b1.GetBondType() != Chem.BondType.SINGLE:
                continue
            a1 = b1.GetOtherAtom(n)

            # case A: N—C(sp3)
            if _is_sp3_carbon(a1):
                for b2 in a1.GetBonds():
                    if b2.GetBondType() != Chem.BondType.SINGLE:
                        continue
                    other = b2.GetOtherAtom(a1)
                    j = other.GetIdx()
                    if j in centers:
                        for i in left_centers:
                            if i != j:
                                edges.add(tuple(sorted((i, j))))

            # case B: N—(O|S|N)—C(sp3)
            if a1.GetAtomicNum() in (8, 16, 7):  # O, S, N
                for b2 in a1.GetBonds():
                    if b2.GetBondType() != Chem.BondType.SINGLE:
                        continue
                    a2 = b2.GetOtherAtom(a1)
                    if _is_sp3_carbon(a2):
                        for b3 in a2.GetBonds():
                            if b3.GetBondType() != Chem.BondType.SINGLE:
                                continue
                            other = b3.GetOtherAtom(a2)
                            j = other.GetIdx()
                            if j in centers:
                                for i in left_centers:
                                    if i != j:
                                        edges.add(tuple(sorted((i, j))))
    return edges


def _largest_component(centers, edges):
    if not centers:
        return set()
    adj = {i: set() for i in centers}
    for i, j in edges:
        if i in adj and j in adj:
            adj[i].add(j)
            adj[j].add(i)
    seen, best = set(), set()
    for s in adj:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            seen.add(u)
            stack.extend(v for v in adj[u] if v not in comp)
        if len(comp) > len(best):
            best = comp
    return best


def backbone_metrics(smiles: str, include_sulfonamide=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    centers, n2c = _centers_and_N(mol, include_sulfonamide)
    edges = _backbone_edges(mol, centers, n2c)
    main_nodes = _largest_component(centers, edges)
    main_edges = {e for e in edges if e[0] in main_nodes and e[1] in main_nodes}
    return {
        "backbone_centers_main": len(main_nodes),
        "backbone_links_main": len(main_edges),
        "MW": Descriptors.MolWt(mol),
        "HAC": mol.GetNumAtoms(),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def compare_shorter_smaller(orig_smiles: str, cand_smiles: str, include_sulfonamide=True,
                            mw_margin: float = 0.0, hac_margin: int = 0):
    o = backbone_metrics(orig_smiles, include_sulfonamide)
    c = backbone_metrics(cand_smiles, include_sulfonamide)
    shorter_backbone = (c["backbone_centers_main"] < o["backbone_centers_main"]) or \
                       (c["backbone_links_main"]   < o["backbone_links_main"])
    smaller_bulk = (c["MW"] <= o["MW"] - mw_margin) and (c["HAC"] <= o["HAC"] - hac_margin)
    return {"original": o, "candidate": c, "decisions": {
        "shorter_backbone": shorter_backbone,
        "smaller_bulk": smaller_bulk,
        "overall_shorter_or_smaller": (shorter_backbone or smaller_bulk),
        "require_both_shorter_and_smaller": (shorter_backbone and smaller_bulk),
    }}


class Length:
    def __init__(self, orig_smiles, cfg, smiles_tokenizer):
        self.cfg = cfg
        self.smiles_tokenizer = smiles_tokenizer

        self.orig_smiles = sf.decoder(sf.encoder(orig_smiles))
        self.L0 = smiles_tokenizer(orig_smiles, return_tensors='pt')['input_ids'].shape[1]
        print("Initial Length: ", self.L0)

    def __call__(self, smiles_tokens, smiles_seq):
        res = compare_shorter_smaller(self.orig_smiles, smiles_seq)
        return int(res["decisions"]["overall_shorter_or_smaller"] and smiles_tokens.shape[1] >= self.L0 // 2)
        # return int(smiles_tokens.shape[1] <= self.L0 and smiles_tokens.shape[1] >= self.L0 // 2)


class ProteinLength:
    """
    Length constraint for protein sequences.
    
    Identical logic to the SMILES Length constraint, but using protein sequence length.
    Requires candidate to be shorter than original AND at least half the original length.
    """
    def __init__(self, orig_protein_seq, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        
        self.orig_protein_seq = orig_protein_seq.strip().replace(" ", "").replace("\n", "")
        # Store length in amino acids (equivalent to tokenized length for proteins)
        self.L0 = len(self.orig_protein_seq)
        print("Initial Length: ", self.L0)

    def __call__(self, protein_tokens, protein_seq):
        """
        Check if protein sequence is shorter than original AND at least half the original length.
        
        Identical logic to SMILES Length constraint:
        - Candidate must be shorter than original (in amino acid length)
        - AND candidate must be at least half the original length
        
        Args:
            protein_tokens: Unused (kept for interface compatibility, could use for tokenized length)
            protein_seq: Protein sequence string (single sequence)
        
        Returns:
            int: 1 if candidate is shorter and >= L0 // 2, 0 otherwise
        """
        seq_clean = (protein_seq or "").strip().replace(" ", "").replace("\n", "")
        L_cand = len(seq_clean)
        
        # Same logic as SMILES Length: shorter AND at least half
        is_shorter = L_cand < self.L0
        is_at_least_half = L_cand >= self.L0 // 2
        
        return int(is_shorter and is_at_least_half)


class TargetLength:
    """
    Target length constraint for protein sequences.
    
    Requires candidate sequence length to exactly match the target length.
    """
    def __init__(self, target_length: int):
        """
        Args:
            target_length: The exact target length in amino acids
        """
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}")
        self.target_length = int(target_length)
        print(f"Target Length: {self.target_length}")

    def __call__(self, protein_tokens, protein_seq):
        """
        Check if protein sequence length exactly matches the target length.
        
        Args:
            protein_tokens: Unused (kept for interface compatibility)
            protein_seq: Protein sequence string (single sequence)
        
        Returns:
            int: 1 if length matches target, 0 otherwise
        """
        seq_clean = (protein_seq or "").strip().replace(" ", "").replace("\n", "")
        L_cand = len(seq_clean)
        
        return int(L_cand == self.target_length)
    
    def predict_batch(self, seqs: Sequence[str]) -> List[bool]:
        """
        Batch prediction helper for evaluation / faster integration.
        
        Args:
            seqs: List of protein sequence strings
        
        Returns:
            List of bool: True if length matches target, False otherwise
        """
        clean = [(s or "").strip().replace(" ", "").replace("\n", "") for s in seqs]
        return [len(s) == self.target_length for s in clean]


class MaxTargetLength:
    """
    Maximum target length constraint for protein sequences.
    
    Requires candidate sequence length to be strictly less than the maximum target length.
    """
    def __init__(self, max_target_length: int):
        """
        Args:
            max_target_length: The maximum allowed length in amino acids (exclusive)
        """
        if max_target_length <= 0:
            raise ValueError(f"max_target_length must be positive, got {max_target_length}")
        self.max_target_length = int(max_target_length)
        print(f"Max Target Length: {self.max_target_length}")

    def __call__(self, protein_tokens, protein_seq):
        """
        Check if protein sequence length is strictly less than the maximum target length.
        
        Args:
            protein_tokens: Unused (kept for interface compatibility)
            protein_seq: Protein sequence string (single sequence)
        
        Returns:
            int: 1 if length < max_target_length, 0 otherwise
        """
        seq_clean = (protein_seq or "").strip().replace(" ", "").replace("\n", "")
        L_cand = len(seq_clean)
        
        return int(L_cand < self.max_target_length)
    
    def predict_batch(self, seqs: Sequence[str]) -> List[bool]:
        """
        Batch prediction helper for evaluation / faster integration.
        
        Args:
            seqs: List of protein sequence strings
        
        Returns:
            List of bool: True if length < max_target_length, False otherwise
        """
        clean = [(s or "").strip().replace(" ", "").replace("\n", "") for s in seqs]
        return [len(s) < self.max_target_length for s in clean]


class PAMMatchingConstraint:
    """
    Terminal constraint for protein sequences: predicted PAM must match target PAM.
    
    Requires that the predicted PAM of the generated sequence exactly matches the target PAM.
    For positions where target PAM is 'N', any nucleotide is acceptable.
    """
    def __init__(self, pam_matching_obj):
        """
        Args:
            pam_matching_obj: PAMMatching objective object that has predict_pam method and target_pam attribute
        """
        if not hasattr(pam_matching_obj, 'predict_pam'):
            raise ValueError("pam_matching_obj must have a predict_pam method")
        if not hasattr(pam_matching_obj, 'target_pam'):
            raise ValueError("pam_matching_obj must have a target_pam attribute")
        self.pam_matching_obj = pam_matching_obj
        self.target_pam = pam_matching_obj.target_pam.upper()
        print(f"PAM Matching Constraint: sequences must have predicted PAM matching target: {self.target_pam}")

    def _pam_matches(self, predicted_pam: str, target_pam: str) -> bool:
        """
        Check if predicted PAM matches target PAM.
        For positions where target PAM is 'N', any nucleotide is acceptable.
        
        Args:
            predicted_pam: Predicted PAM string (10 nucleotides)
            target_pam: Target PAM string (10 nucleotides, may contain 'N')
        
        Returns:
            bool: True if predicted PAM matches target PAM, False otherwise
        """
        # if len(predicted_pam) != len(target_pam):
        #     return False
        
        # for pred_nuc, target_nuc in zip(predicted_pam, target_pam):
        #     if target_nuc == 'N':
        #         # N in target means any nucleotide is acceptable
        #         continue
        #     if pred_nuc != target_nuc:
        #         return False
        
        # return True
        return predicted_pam.upper() == target_pam.upper()

    def __call__(self, protein_tokens, protein_seq):
        """
        Check if protein sequence's predicted PAM matches the target PAM.
        
        Args:
            protein_tokens: Unused (kept for interface compatibility)
            protein_seq: Protein sequence string (single sequence)
        
        Returns:
            int: 1 if predicted PAM matches target PAM, 0 otherwise
        """
        try:
            # Check if the model is available
            if not hasattr(self.pam_matching_obj, 'model') or self.pam_matching_obj.model is None:
                print(f"[WARNING] PAMMatchingConstraint: model is not initialized. Cannot predict PAM.")
                return 0
            
            if not hasattr(self.pam_matching_obj, 'tokenizer') or self.pam_matching_obj.tokenizer is None:
                print(f"[WARNING] PAMMatchingConstraint: tokenizer is not initialized. Cannot predict PAM.")
                return 0
            
            predicted_pams = self.pam_matching_obj.predict_pam([protein_seq])
            if not predicted_pams:
                print(f"[WARNING] PAMMatchingConstraint: predict_pam returned empty list for sequence.")
                return 0
            
            predicted_pam = predicted_pams[0]
            matches = self._pam_matches(predicted_pam, self.target_pam)
            if not matches:
                # Debug output: show what was predicted vs expected
                print(f"[DEBUG] PAMMatchingConstraint: predicted '{predicted_pam}' != target '{self.target_pam}'")
            return int(matches)
        except Exception as e:
            import traceback
            print(f"[ERROR] PAMMatchingConstraint failed: {e}")
            traceback.print_exc()
            return 0
    
    def predict_batch(self, seqs: Sequence[str]) -> List[bool]:
        """
        Batch prediction helper for evaluation / faster integration.
        
        Args:
            seqs: List of protein sequence strings
        
        Returns:
            List of bool: True if predicted PAM matches target PAM, False otherwise
        """
        try:
            # Check if the model is available
            if not hasattr(self.pam_matching_obj, 'model') or self.pam_matching_obj.model is None:
                print(f"[WARNING] PAMMatchingConstraint.predict_batch: model is not initialized. Cannot predict PAM.")
                return [False] * len(seqs)
            
            if not hasattr(self.pam_matching_obj, 'tokenizer') or self.pam_matching_obj.tokenizer is None:
                print(f"[WARNING] PAMMatchingConstraint.predict_batch: tokenizer is not initialized. Cannot predict PAM.")
                return [False] * len(seqs)
            
            predicted_pams = self.pam_matching_obj.predict_pam(seqs)
            results = []
            for i, predicted_pam in enumerate(predicted_pams):
                if not predicted_pam:
                    print(f"[WARNING] PAMMatchingConstraint.predict_batch: empty prediction for sequence {i}")
                    results.append(False)
                else:
                    results.append(self._pam_matches(predicted_pam, self.target_pam))
            return results
        except Exception as e:
            import traceback
            print(f"[ERROR] PAMMatchingConstraint.predict_batch failed: {e}")
            traceback.print_exc()
            return [False] * len(seqs)


class Cas9ScoreThreshold:
    """
    Terminal constraint for protein sequences: Cas9 classifier score must be above a threshold.
    
    Requires that the Cas9 classifier score (probability) is strictly greater than the threshold.
    """
    def __init__(self, cas9_classifier_obj, threshold: float):
        """
        Args:
            cas9_classifier_obj: Cas9Classification objective object that has get_scores method
            threshold: Minimum Cas9 score threshold (0-1). Sequences must have score > threshold.
        """
        if not hasattr(cas9_classifier_obj, 'get_scores'):
            raise ValueError("cas9_classifier_obj must have a get_scores method")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.cas9_classifier_obj = cas9_classifier_obj
        self.threshold = float(threshold)
        print(f"Cas9 Score Threshold Constraint: sequences must have Cas9 score > {self.threshold}")

    def __call__(self, protein_tokens, protein_seq):
        """
        Check if protein sequence's Cas9 score is above the threshold.
        
        Args:
            protein_tokens: Unused (kept for interface compatibility)
            protein_seq: Protein sequence string (single sequence)
        
        Returns:
            int: 1 if Cas9 score > threshold, 0 otherwise
        """
        scores = self.cas9_classifier_obj.get_scores([protein_seq])
        if not scores:
            return 0
        
        score = scores[0]
        return int(score > self.threshold)
    
    def predict_batch(self, seqs: Sequence[str]) -> List[bool]:
        """
        Batch prediction helper for evaluation / faster integration.
        
        Args:
            seqs: List of protein sequence strings
        
        Returns:
            List of bool: True if Cas9 score > threshold, False otherwise
        """
        scores = self.cas9_classifier_obj.get_scores(seqs)
        return [score > self.threshold for score in scores]


# -------------------------------------------------------------------------
# Cas9 nuclease domain completeness detector (HNH + RuvC), recall-first
# -------------------------------------------------------------------------

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class Cas9DomainCompleteness:
    """
    Hard constraint for protein sequences:
      pass iff sequence contains BOTH HNH and RuvC domains (domain presence),
      using the Option-B detector (Stage1 fast scan + Stage2 --max rescue).

    Intended use:
      - As a *domain presence* gate (high recall)
      - Combine with an MLP classifier to enforce "is Cas9" / other plausibility checks.
    """
    def __init__(
        self,
        hmm_path: str,
        cpu: int = 16,
        # Stage1 acceptance
        ev1: float = 1e-2,
        minlen1: int = 35,
        # Stage2 acceptance (recall-first)
        ev2: float = 10.0,
        minlen2: int = 20,
        # Optional: relax Stage1 HMMER filters (usually leave False)
        stage1_relax_filters: bool = False,
        # Cache results by sequence content to avoid repeated hmmscan calls
        enable_cache: bool = True,
    ):
        if DomainDetector is None:
            raise ImportError(
                "Could not import DomainDetector. Ensure cope_domain_detector.py is in the same package as constraints.py "
                "and that its imports succeed."
            )
        self.hmm_path = hmm_path
        self.enable_cache = enable_cache
        self._cache: Dict[str, int] = {}

        cfg = DomainDetectorConfig(
            hmm_path=hmm_path,
            cpu=cpu,
            stage1=HitCriteria(ievalue_max=ev1, min_dom_len=minlen1),
            stage2=HitCriteria(ievalue_max=ev2, min_dom_len=minlen2),
        )

        if stage1_relax_filters:
            cfg = DomainDetectorConfig(
                hmm_path=hmm_path,
                cpu=cpu,
                stage1=HitCriteria(ievalue_max=ev1, min_dom_len=minlen1),
                stage2=HitCriteria(ievalue_max=ev2, min_dom_len=minlen2),
                stage1_F1=1.0, stage1_F2=1.0, stage1_F3=1.0,
                stage1_report_E=1e6, stage1_report_domE=1e6,
            )

        self.detector = DomainDetector(cfg)

    def __call__(self, protein_tokens, protein_seq: str) -> int:
        # protein_tokens is unused; kept to match existing constraint signature
        seq = (protein_seq or "").strip().replace(" ", "").replace("\n", "")
        if not seq:
            return 0

        if self.enable_cache:
            key = _sha1(seq)
            if key in self._cache:
                return self._cache[key]

        passed = int(self.detector.predict_one(seq, id_="query"))

        if self.enable_cache:
            self._cache[_sha1(seq)] = passed
        return passed

    def predict_batch(self, seqs: Sequence[str], ids: Optional[Sequence[str]] = None) -> List[bool]:
        """
        Batch prediction helper for evaluation / faster integration.
        """
        clean = [(s or "").strip().replace(" ", "").replace("\n", "") for s in seqs]
        return self.detector.predict_batch(clean, ids=ids)
