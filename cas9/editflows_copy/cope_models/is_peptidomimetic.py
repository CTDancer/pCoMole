from rdkit import Chem
from rdkit.Chem import rdchem

# ---------- SMARTS ----------
SMARTS = {
    # Amide (2-atom match: amide N and carbonyl C) — avoids 3-tuple unpacking
    "amide": Chem.MolFromSmarts("[NX3]-[CX3;$(C=O)]"),

    # α-peptide tile: O=C - N - C(sp3) - C(=O)
    "alpha_tile": Chem.MolFromSmarts("[CX3](=O)-[NX3]-[CX4]-[CX3](=O)"),

    # β/γ spacing (both orientations around an amide)
    "beta_fwd": Chem.MolFromSmarts("[NX3]-[CX4]-[CX4]-[CX3](=O)"),
    "beta_rev": Chem.MolFromSmarts("[CX3](=O)-[CX4]-[CX4]-[NX3]"),

    # Backbone isostere motifs
    "depsipeptide": Chem.MolFromSmarts("[CX3](=O)-O-[CX4]-[CX3](=O)"),
    "thioamide":    Chem.MolFromSmarts("[CX3](=S)-[NX3]"),
    "peptoid":      Chem.MolFromSmarts("O=C-N([#6])-[CH2]-C(=O)"),

    # Common peptidomimetic isosteres
    "urea":         Chem.MolFromSmarts("[NX3]-[CX3](=O)-[NX3]"),
    "thiourea":     Chem.MolFromSmarts("[NX3]-[CX3](=S)-[NX3]"),
    "guanidine":    Chem.MolFromSmarts("[NX3]-C(=N)-[NX3]"),
    "sulfonamide":  Chem.MolFromSmarts("[SX4](=O)(=O)-[NX3]"),

    # Linear imide pattern kept (some imides are not caught by ring/di-acyl logic)
    "imide_linear": Chem.MolFromSmarts("[CX3](=O)-[NX3](-[CX3](=O))"),
}

# ---------- helpers ----------
def _count_matches(mol, patt):
    return len(mol.GetSubstructMatches(patt)) if patt is not None else 0

def _amide_matches(mol):
    """Return list of (N_idx, C_idx) for amide pairs using the 2-atom SMARTS."""
    return list(mol.GetSubstructMatches(SMARTS["amide"]))

def _amide_atoms(mol):
    for n_idx, c_idx in _amide_matches(mol):
        yield mol.GetAtomWithIdx(n_idx), mol.GetAtomWithIdx(c_idx)

def _is_proline_like_tertiary_amide(n_atom: rdchem.Atom) -> bool:
    """Tertiary amide N that sits in a 5-member ring with its α-carbon (proline-like)."""
    if n_atom.GetTotalNumHs() != 0 or n_atom.GetDegree() < 2:
        return False
    mol = n_atom.GetOwningMol()
    ri = mol.GetRingInfo()
    if not ri.IsAtomInRingOfSize(n_atom.GetIdx(), 5):
        return False
    for nbr in n_atom.GetNeighbors():
        if nbr.GetAtomicNum() == 6 and nbr.GetHybridization() == rdchem.HybridizationType.SP3:
            if ri.IsAtomInRingOfSize(nbr.GetIdx(), 5):
                return True
    return False

def _has_nonproline_tertiary_amide(mol) -> bool:
    """Any tertiary amide N (no N–H) that is not proline-like."""
    for n_atom, _ in _amide_atoms(mol):
        if n_atom.GetTotalNumHs() == 0 and not _is_proline_like_tertiary_amide(n_atom):
            return True
    return False

def _count_secondary_amides(mol) -> int:
    """Count amide nitrogens with at least one attached hydrogen (secondary amides)."""
    cnt = 0
    for n_atom, _ in _amide_atoms(mol):
        if n_atom.GetTotalNumHs() > 0:
            cnt += 1
    return cnt

def _count_lactam_ring_amides(mol) -> int:
    """
    Count amide pairs where the N and the carbonyl C are in the same ring (lactam/imide ring).
    """
    ri = mol.GetRingInfo()
    nC_pairs = _amide_matches(mol)
    count = 0
    for n_idx, c_idx in nC_pairs:
        # Efficient ring membership test
        if ri.AreAtomsInSameRing(n_idx, c_idx):
            count += 1
    return count

def _count_diacyl_amide_nitrogens(mol) -> int:
    """
    Count amide nitrogens that are bonded to >= 2 carbonyl carbons (C=O or C=S).
    This generalizes 'imide' detection (including cyclic/fused cases).
    """
    count = 0
    for n_atom, _ in _amide_atoms(mol):
        carbonyl_neighbors = 0
        for nbr in n_atom.GetNeighbors():
            if nbr.GetAtomicNum() != 6:
                continue
            # Check if this carbon has a double bond to O or S (C=O/C=S)
            is_carbonyl = False
            for bond in nbr.GetBonds():
                if bond.GetBeginAtomIdx() == nbr.GetIdx():
                    other = bond.GetEndAtom()
                elif bond.GetEndAtomIdx() == nbr.GetIdx():
                    other = bond.GetBeginAtom()
                else:
                    continue
                if bond.GetBondType() == rdchem.BondType.DOUBLE and other.GetAtomicNum() in (8, 16):
                    is_carbonyl = True
                    break
            if is_carbonyl:
                carbonyl_neighbors += 1
        if carbonyl_neighbors >= 2:
            count += 1
    return count

def _is_alpha_peptide_only(mol) -> bool:
    """
    Stricter α-peptide criterion to avoid small lactams/imides:
      - amide_count >= 2  (need chain-like character)
      - at least one secondary amide (N–H) present
      - no β/γ spacing
      - alpha_tiles cover all but up to two terminal amides
    """
    amide_count = _count_matches(mol, SMARTS["amide"])
    if amide_count < 2:  # dipeptide or larger; single-amide scaffolds aren't treated as α-peptides here
        return False

    if _count_secondary_amides(mol) == 0:
        return False

    beta_tiles = (_count_matches(mol, SMARTS["beta_fwd"]) +
                  _count_matches(mol, SMARTS["beta_rev"]))
    if beta_tiles > 0:
        return False

    alpha_tiles = _count_matches(mol, SMARTS["alpha_tile"])
    terminal_allowance = min(2, amide_count)
    required_alpha = max(0, amide_count - terminal_allowance)
    return alpha_tiles >= required_alpha

# ---------- main classification ----------
def classify_molecule(smiles: str):
    """
    Return (label, audit) where label ∈ {
        'peptidomimetic_not_natural', 'natural_peptide', 'non_peptide_or_uncertain'
    }
    """
    audit = {"smiles": smiles, "parsed": False, "errors": []}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            audit["errors"].append("SMILES parsing returned None")
            return "non_peptide_or_uncertain", audit
        audit["parsed"] = True
    except Exception as e:
        audit["errors"].append(f"RDKit parse error: {e}")
        return "non_peptide_or_uncertain", audit

    # Counts from SMARTS
    amide_count   = _count_matches(mol, SMARTS["amide"])
    alpha_tiles   = _count_matches(mol, SMARTS["alpha_tile"])
    beta_tiles    = (_count_matches(mol, SMARTS["beta_fwd"]) +
                     _count_matches(mol, SMARTS["beta_rev"]))
    depsis        = _count_matches(mol, SMARTS["depsipeptide"])
    thioamides    = _count_matches(mol, SMARTS["thioamide"])
    peptoids      = _count_matches(mol, SMARTS["peptoid"])
    ureas         = _count_matches(mol, SMARTS["urea"])
    thioureas     = _count_matches(mol, SMARTS["thiourea"])
    guanidines    = _count_matches(mol, SMARTS["guanidine"])
    sulfonamides  = _count_matches(mol, SMARTS["sulfonamide"])
    imide_linear  = _count_matches(mol, SMARTS["imide_linear"])

    # Structural, non-SMARTS detectors
    tert_nonpro   = _has_nonproline_tertiary_amide(mol)
    lactam_rings  = _count_lactam_ring_amides(mol)
    diacyl_Ns     = _count_diacyl_amide_nitrogens(mol)

    audit.update({
        "counts": {
            "amide": amide_count,
            "alpha_tiles": alpha_tiles,
            "beta_or_gamma_tiles": beta_tiles,
            "depsipeptide_links": depsis,
            "thioamide_links": thioamides,
            "peptoid_motifs": peptoids,
            "urea": ureas,
            "thiourea": thioureas,
            "guanidine": guanidines,
            "sulfonamide": sulfonamides,
            "imide_linear": imide_linear,
            "lactam_ring_amides": lactam_rings,
            "diacyl_amide_nitrogens": diacyl_Ns,
        },
        "nonproline_tertiary_amide": tert_nonpro
    })

    alpha_only = _is_alpha_peptide_only(mol)
    audit["alpha_only_backbone"] = alpha_only

    mimetic_indicators = {
        "beta_or_gamma_backbone": beta_tiles > 0,
        "depsipeptide_in_backbone": depsis > 0,
        "thioamide_present": thioamides > 0,
        "peptoid_motif_present": peptoids > 0,
        "nonproline_tertiary_amide": tert_nonpro,
        "urea_or_thiourea": (ureas + thioureas) > 0,
        "guanidine_like": guanidines > 0,
        "sulfonamide_present": sulfonamides > 0,
        "imide_or_lactam_isostere": (imide_linear > 0) or (lactam_rings > 0) or (diacyl_Ns > 0),
    }
    audit["mimetic_indicators"] = mimetic_indicators
    is_mimetic = any(mimetic_indicators.values())

    if alpha_only and not is_mimetic:
        label = "natural_peptide"
    elif is_mimetic:
        label = "peptidomimetic_not_natural"
    else:
        label = "non_peptide_or_uncertain"

    return label, audit

def is_peptidomimetic_not_natural(smiles: str):
    label, audit = classify_molecule(smiles)
    return label == "peptidomimetic_not_natural", audit


# ---------- quick demo ----------
if __name__ == "__main__":
    sequences = ['CCC(C)[C@H](NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)[C@H](C)N)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)ON']
    for seq in sequences:
        flag, audit = is_peptidomimetic_not_natural(seq)
        print("-----------------")
        print(f"{seq}\nIs Peptidomimetic: {flag}\n{audit['mimetic_indicators']}\nalpha_only={audit['alpha_only_backbone']}\n")
