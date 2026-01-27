import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import selfies as sf
import numpy as np

from is_peptidomimetic import is_peptidomimetic_not_natural
import pdb

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

    def __call__(self, smiles_tokens, smiles_seqs):
        scores = []
        for smiles in smiles_seqs:
            flag, audit = is_peptidomimetic_not_natural(smiles)
            score = 1 if flag else 0
            scores.append(score)
        return scores


# SMARTS
_AMIDE       = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")               # amide C(=O)-N (includes carbamates/ureas as centers)
_SULFONAMIDE = Chem.MolFromSmarts("[SX4](=[OX1])(=[OX1])[NX3]")       # sulfonamide S(=O)2-N  (optional isostere)

def _centers_and_N(mol, include_sulfonamide=True):
    centers, n2c = set(), {}
    for c_idx, _, n_idx in mol.GetSubstructMatches(_AMIDE):
        centers.add(c_idx); n2c.setdefault(n_idx, set()).add(c_idx)
    if include_sulfonamide:
        for s_idx, _, _, n_idx in mol.GetSubstructMatches(_SULFONAMIDE):
            centers.add(s_idx); n2c.setdefault(n_idx, set()).add(s_idx)
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
    if not centers: return set()
    adj = {i: set() for i in centers}
    for i, j in edges:
        if i in adj and j in adj:
            adj[i].add(j); adj[j].add(i)
    seen, best = set(), set()
    for s in adj:
        if s in seen: 
            continue
        stack, comp = [s], set()
        while stack:
            u = stack.pop()
            if u in comp: 
                continue
            comp.add(u); seen.add(u)
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

    def __call__(self, smiles_tokens, smiles_seqs):
        scores = []
        smiles_tokens_np = smiles_tokens.detach().cpu().numpy()    # converting to numpy for easy trimming 0s at the end
        
        for i in range(len(smiles_seqs)):
            smiles_token = smiles_tokens_np[i]
            smiles_token = np.trim_zeros(smiles_token, 'b')
            
            res = compare_shorter_smaller(self.orig_smiles, smiles_seqs[i])
            score = int(res["decisions"]["overall_shorter_or_smaller"] and len(smiles_token) >= self.L0 // 2 and len(smiles_token) < self.L0)
            scores.append(score)

        return scores

class Pass:
    def __init__(self):
        pass

    def __call__(self, smiles_tokens, smiles_seqs):
        return [1] * len(smiles_seqs)