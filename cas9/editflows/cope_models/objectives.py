# import warnings
# import logging
# warnings.filterwarnings("ignore", message="to-Python converter.*already registered", category=RuntimeWarning)
# warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# from rdkit import RDLogger, rdBase
# RDLogger.DisableLog("rdApp.*")
# logging.basicConfig(level=logging.ERROR)
# logging.getLogger("chemprop").setLevel(logging.ERROR)
# logging.getLogger("hyperopt").setLevel(logging.ERROR)
# rdBase.DisableLog('rdApp.error')

import torch
import torch.nn.functional as F
import logging
from rdkit import Chem
# Make chemprop imports conditional to avoid import errors when not needed
try:
    from chemprop import data, models
    # Try to import featurizers separately as it may not exist in all versions
    try:
        from chemprop import featurizers
    except (ImportError, AttributeError):
        featurizers = None
    CHEMPROP_AVAILABLE = True
except (ImportError, AttributeError) as e:
    CHEMPROP_AVAILABLE = False
    # Create dummy objects to avoid NameError if someone tries to use them
    data = None
    featurizers = None
    models = None
from lightning import pytorch as pl

# ---- QUIET MODE (put these lines at the top of your script) ----
import os, warnings, logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Make PyTorch stop suggesting Tensor Core settings
torch.set_float32_matmul_precision("high")

# Silence Python warnings (fine-tune as needed)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=r".*predict_dataloader.*many workers.*")
warnings.filterwarnings("ignore", message=r"Dropping last batch of size .*")

# Quiet RDKit
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Quiet common loggers (Lightning, Chemprop, etc.)
logging.basicConfig(level=logging.ERROR, force=True)
for name in [
    "lightning", "pytorch_lightning", "lightning.pytorch",
    "chemprop", "rdkit", "urllib3", "torch"
]:
    logging.getLogger(name).setLevel(logging.ERROR)
# ---------------------------------------------------------------


# from admet_ai import ADMETModel
import sys
import os
# Optional: add path to ReDi_discrete/smiles if using related objectives
# sys.path.append('path/to/ReDi_discrete/smiles')
import xgboost as xgb
import numpy as np
from transformers import AutoModelForMaskedLM
import warnings
import numpy as np
import esm
import torch.nn as nn
from rdkit import Chem
from collections import defaultdict
import pdb

import math
import sys
import os
from pathlib import Path

# Add cas_predictor to path for imports
cas_predictor_path = Path(__file__).parent.parent.parent.parent.parent / "predictors" / "cas_predictor"
if str(cas_predictor_path) not in sys.path:
    sys.path.insert(0, str(cas_predictor_path))

# Add protein2pam to path for imports
protein2pam_path = Path(__file__).parent.parent.parent.parent.parent / "predictors" / "protein2pam-0.2.0"
if protein2pam_path.exists():
    protein2pam_path = protein2pam_path.resolve()  # Resolve symlinks/relative paths
    protein2pam_path_str = str(protein2pam_path)
    # Check if path (or original unresolved path) is already in sys.path
    original_path_str = str(Path(__file__).parent.parent.parent.parent.parent / "predictors" / "protein2pam-0.2.0")
    if protein2pam_path_str not in sys.path and original_path_str not in sys.path:
        sys.path.insert(0, protein2pam_path_str)

try:
    # Import using importlib to avoid conflicts with editflows/model package
    import importlib.util
    import importlib
    
    # Temporarily add cas_predictor to sys.path at the front to ensure it takes precedence
    cas_predictor_str = str(cas_predictor_path)
    original_path = sys.path.copy()
    if cas_predictor_str not in sys.path:
        sys.path.insert(0, cas_predictor_str)
    
    try:
        # Load model.py directly - use cas_predictor.model as the module name
        model_file = cas_predictor_path / "model.py"
        spec = importlib.util.spec_from_file_location("cas_predictor.model", model_file)
        cas_model = importlib.util.module_from_spec(spec)
        # Set __file__ to help with relative imports
        cas_model.__file__ = str(model_file)
        # Register the module in sys.modules before exec_module so imports work
        sys.modules['cas_predictor.model'] = cas_model
        sys.modules['model'] = cas_model  # Also register as 'model' for lightning_module imports
        spec.loader.exec_module(cas_model)
        Cas9Classifier = cas_model.Cas9Classifier
        
        # Load lightning_module.py - it will import from 'model' which should now resolve correctly
        lightning_file = cas_predictor_path / "lightning_module.py"
        spec_lightning = importlib.util.spec_from_file_location("cas_predictor.lightning_module", lightning_file)
        cas_lightning = importlib.util.module_from_spec(spec_lightning)
        # Set __file__ to help with relative imports
        cas_lightning.__file__ = str(lightning_file)
        # Register the module in sys.modules before exec_module
        sys.modules['cas_predictor.lightning_module'] = cas_lightning
        spec_lightning.loader.exec_module(cas_lightning)
        Cas9ClassifierModule = cas_lightning.Cas9ClassifierModule
        
        import yaml
        try:
            from easydict import EasyDict as edict
        except ImportError:
            # Fallback: use regular dict if easydict not available
            class EasyDict(dict):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                def __getattr__(self, key):
                    try:
                        return self[key]
                    except KeyError:
                        raise AttributeError(key)
                def __setattr__(self, key, value):
                    self[key] = value
            edict = EasyDict
        CAS9_PREDICTOR_AVAILABLE = True
    finally:
        # Restore original sys.path
        sys.path[:] = original_path
except (ImportError, Exception) as e:
    print(f"Warning: Could not import Cas9 predictor modules: {e}")
    import traceback
    traceback.print_exc()
    CAS9_PREDICTOR_AVAILABLE = False

# SMARTS patterns
_AMIDE_SMARTS        = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")                     # C(=O)-N
_CARBONYL_C_SMARTS   = Chem.MolFromSmarts("[CX3](=[OX1])")                          # carbonyl C
_DIPEPTIDE_SMARTS    = Chem.MolFromSmarts("[CX3](=[OX1])N[#6X4][CX3](=[OX1])N")     # amide–C(sp3)–amide

def _amide_bond_indices(mol, ignore_ring_amides=False):
    ids = set()
    for c_idx, _, n_idx in mol.GetSubstructMatches(_AMIDE_SMARTS):
        b = mol.GetBondBetweenAtoms(c_idx, n_idx)
        if b and b.GetBondType() == Chem.rdchem.BondType.SINGLE:
            if ignore_ring_amides and b.IsInRing():
                continue
            ids.add(b.GetIdx())
    return ids

def _carbonyl_c_indices(mol):
    return {m[0] for m in mol.GetSubstructMatches(_CARBONYL_C_SMARTS)}

def _carbonyl_neighbor_stats(mol, c_indices):
    stats = {"total": 0, "with_N": 0, "with_O": 0, "with_S": 0, "pure_amide": 0}
    for c_idx in c_indices:
        c = mol.GetAtomWithIdx(c_idx)
        stats["total"] += 1
        hasN = hasO = hasS = False
        for b in c.GetBonds():
            if b.GetBondType() != Chem.rdchem.BondType.SINGLE:
                continue
            z = b.GetOtherAtom(c).GetAtomicNum()
            if   z == 7:  hasN = True
            elif z == 8:  hasO = True
            elif z == 16: hasS = True
        stats["with_N"] += int(hasN)
        stats["with_O"] += int(hasO)
        stats["with_S"] += int(hasS)
        if hasN and not (hasO or hasS):
            stats["pure_amide"] += 1
    return stats

def _adjacent_amide_pairs(mol):
    # Count distinct amide–C(sp3)–amide windows (dedup by central carbon)
    centers = set()
    for match in mol.GetSubstructMatches(_DIPEPTIDE_SMARTS):
        centers.add(match[3])  # central sp3 carbon index
    return len(centers)

def analyze_peptide_likeness(smiles: str,
                             ignore_ring_amides: bool = False,
                             amide_density_target: float = 0.12):
    """
    Compute peptide-likeness metrics and a continuous score in [0,1].
    - ignore_ring_amides=False: include macro/cyclic peptides by default.
    - amide_density_target: amide-per-atom density to hit score~1 for peptides
      (~0.10–0.15 works well; default 0.12 ≈ 1 amide per ~8 heavy atoms).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    n_heavy_atoms  = mol.GetNumAtoms()
    n_heavy_bonds  = mol.GetNumBonds()

    amide_bonds    = _amide_bond_indices(mol, ignore_ring_amides=ignore_ring_amides)
    n_amide_bonds  = len(amide_bonds)

    carbonyl_cs    = _carbonyl_c_indices(mol)
    cstats         = _carbonyl_neighbor_stats(mol, carbonyl_cs)
    total_carb     = max(1, cstats["total"])

    # Core features
    f1 = cstats["with_N"] / total_carb  # acyl-N fraction ∈ [0,1]
    amide_per_atom = n_amide_bonds / max(1, n_heavy_atoms)
    f2 = min(1.0, amide_per_atom / max(1e-8, amide_density_target))  # saturate at 1
    n_adjacent = _adjacent_amide_pairs(mol)
    f3 = 1.0 - math.exp(-n_adjacent)  # 0, 0.63, 0.86, 0.95, ... as pairs increase

    # Penalty for non-peptidic carbonyls (carbamates/anhydrides/thioesters)
    pure_amide_fraction = cstats["pure_amide"] / total_carb
    penalty = 1.0 - pure_amide_fraction  # 0 (all pure amide) … 1 (no pure amide)

    # Final heuristic score in [0,1]
    score = 0.55 * f1 + 0.25 * f2 + 0.20 * f3 - 0.25 * penalty
    score = max(0.0, min(1.0, score))

    return {
        "n_heavy_atoms": n_heavy_atoms,
        "n_heavy_bonds": n_heavy_bonds,
        "n_carbonyls": cstats["total"],
        "n_amide_bonds": n_amide_bonds,
        "amide_bond_ratio_all_bonds": n_amide_bonds / max(1, n_heavy_bonds),
        "acyl_N_fraction": f1,
        "pure_amide_fraction": pure_amide_fraction,
        "amide_per_atom": amide_per_atom,
        "n_adjacent_amide_pairs": n_adjacent,
        "peptide_likeness": score,  # <<< continuous score in [0,1]
    }


def score_combination(ratios, scores, admet_scores):
    high_mask = ratios > 0.6
    low_mask  = ratios < 0.1
    mid_mask  = ~(high_mask | low_mask)

    # start with zeros
    final_scores = torch.zeros_like(scores)

    # high-peptide: use peptide scores
    final_scores[high_mask] = scores[high_mask]

    # low-peptide: use admet scores
    final_scores[low_mask] = admet_scores[low_mask]

    # middle band: linear blend
    if mid_mask.any():
        r_mid = ratios[mid_mask]
        alpha = (r_mid - 0.1) / 0.5          # in [0, 1]
        blended = alpha * scores[mid_mask] + (1 - alpha) * admet_scores[mid_mask]
        final_scores[mid_mask] = blended

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


class Toxicity:
    def __init__(self, device):
        self.predictor = xgb.Booster(model_file=os.environ.get('TOXICITY_MODEL_PATH', 'path/to/toxicity/best_model_f1.json'))
        self.emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device)
        self.emb_model.eval()

    def get_scores(self, x):
        scores = np.zeros(len(x))
        features = np.array(self.emb_model(input_ids=x).last_hidden_state.mean(dim=1).detach().cpu())
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores.item()
    
    def __call__(self, smiles_tokens, smiles_seq):
        score = self.get_scores(smiles_tokens)

        return 'non_toxicity', 1 - score


class Solubility:
    def __init__(self, device):
        self.predictor = xgb.Booster(model_file=os.environ.get('SOLUBILITY_MODEL_PATH', 'path/to/solubility/best_model_f1.json'))
        self.emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device)
        self.emb_model.eval()

    def get_scores(self, x):
        scores = np.zeros(len(x))
        features = np.array(self.emb_model(input_ids=x).last_hidden_state.mean(dim=1).detach().cpu())
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores.item()
    
    def __call__(self, smiles_tokens, smiles_seq):        
        score = self.get_scores(smiles_tokens)
        return 'solubility', score
    
class Permeability:
    
    def __init__(self, device):
        self.predictor = xgb.Booster(model_file=os.environ.get('PERMEABILITY_MODEL_PATH', 'path/to/permeability/best_model.json'))
        self.emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device)
        self.emb_model.eval()
    
    def get_scores(self, x):
        scores = -10 * np.ones(len(x))
        features = np.array(self.emb_model(input_ids=x).last_hidden_state.mean(dim=1).detach().cpu())
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores.item()
    
    def __call__(self, smiles_tokens, smiles_seq):        
        score = self.get_scores(smiles_tokens)
        score = (10 + score) / 10
        return 'permeability', score


class Halflife:
    def __init__(self, device):
        self.predictor = xgb.Booster(model_file=os.environ.get('HALFLIFE_MODEL_PATH', 'path/to/halflife/best_model.json'))
        self.emb_model = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all").roformer.to(device)
        self.emb_model.eval()

    def get_scores(self, x):
        scores = np.zeros(len(x))
        features = np.array(self.emb_model(input_ids=x).last_hidden_state.mean(dim=1).detach().cpu())
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores.item()
    
    def __call__(self, smiles_tokens, smiles_seq):
        score = max(0, min(2, self.get_scores(smiles_tokens))) / 2
        return 'halflife', score

# class Stability:
#     def __init__(self, cfg, device, smiles_tokenizer):
#         self.predictor = xgb.Booster(model_file='path/to/stability/stability_best.json')
#         self.emb_model = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all").roformer.to(device)
#         self.emb_model.eval()
#         self.smiles_tokenizer = smiles_tokenizer

#         self.cfg = cfg
#         self.device = device

#     def get_scores(self, x):
#         scores = np.zeros(len(x))
#         features = np.array(self.emb_model(input_ids=x).last_hidden_state.mean(dim=1).detach().cpu())
        
#         if len(features) == 0:
#             return scores
        
#         features = np.nan_to_num(features, nan=0.)
#         features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
#         features = xgb.DMatrix(features)
        
#         scores = self.predictor.predict(features)
#         return scores
    
#     def __call__(self, sequences):
#         ratios = torch.tensor([peptide_bond_ratio(smiles)[0] for smiles in sequences]).to(self.device)
#         x = self.smiles_tokenizer(sequences)['input_ids']
#         scores = torch.tensor(self.get_scores(x)).to(self.device)

#         admet_scores = torch.tensor([max(0, min(-4, self.admet(smiles)['Solubility_AqSolDB'])) / 4 for smiles in sequences]).to(self.device)
#         final_scores = score_combination(ratios, scores, admet_scores)
#         return final_scores
    



class ImprovedBindingPredictor(nn.Module):
    def __init__(self, 
                 esm_dim=1280,
                 smiles_dim=768,
                 hidden_dim=512,
                 n_heads=8,
                 n_layers=3,
                 dropout=0.1):
        super().__init__()
        
        # Define binding thresholds
        self.tight_threshold = 7.5    # Kd/Ki/IC50 ≤ ~30nM
        self.weak_threshold = 6.0     # Kd/Ki/IC50 > 1μM
        
        # Project to same dimension
        self.smiles_projection = nn.Linear(smiles_dim, hidden_dim)
        self.protein_projection = nn.Linear(esm_dim, hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.smiles_norm = nn.LayerNorm(hidden_dim)
        
        # Cross attention blocks with layer norm
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(n_layers)
        ])
        
        # Prediction heads
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Regression head
        self.regression_head = nn.Linear(hidden_dim, 1)
        
        # Classification head (3 classes: tight, medium, loose binding)
        self.classification_head = nn.Linear(hidden_dim, 3)
        
    def get_binding_class(self, affinity):
        """Convert affinity values to class indices
        0: tight binding (>= 7.5)
        1: medium binding (6.0-7.5)
        2: weak binding (< 6.0)
        """
        if isinstance(affinity, torch.Tensor):
            tight_mask = affinity >= self.tight_threshold
            weak_mask = affinity < self.weak_threshold
            medium_mask = ~(tight_mask | weak_mask)
            
            classes = torch.zeros_like(affinity, dtype=torch.long)
            classes[medium_mask] = 1
            classes[weak_mask] = 2
            return classes
        else:
            if affinity >= self.tight_threshold:
                return 0  # tight binding
            elif affinity < self.weak_threshold:
                return 2  # weak binding
            else:
                return 1  # medium binding
        
    def forward(self, protein_emb, smiles_emb):
        protein = self.protein_norm(self.protein_projection(protein_emb))
        smiles = self.smiles_norm(self.smiles_projection(smiles_emb))
        
        #protein = protein.transpose(0, 1)
        #smiles = smiles.transpose(0, 1)
        
        # Cross attention layers
        for layer in self.cross_attention_layers:
            # Protein attending to SMILES
            attended_protein = layer['attention'](
                protein, smiles, smiles
            )[0]
            protein = layer['norm1'](protein + attended_protein)
            protein = layer['norm2'](protein + layer['ffn'](protein))
            
            # SMILES attending to protein
            attended_smiles = layer['attention'](
                smiles, protein, protein
            )[0]
            smiles = layer['norm1'](smiles + attended_smiles)
            smiles = layer['norm2'](smiles + layer['ffn'](smiles))
        
        # Get sequence-level representations
        protein_pool = torch.mean(protein, dim=0)
        smiles_pool = torch.mean(smiles, dim=0)
        
        # Concatenate both representations
        combined = torch.cat([protein_pool, smiles_pool], dim=-1)
        
        # Shared features
        shared_features = self.shared_head(combined)
        
        regression_output = self.regression_head(shared_features)
        classification_logits = self.classification_head(shared_features)
        
        return regression_output, classification_logits
    
class BindingAffinity:
    def __init__(self, prot_seq, device):
        super().__init__()
    
        # peptide embeddings
        self.pep_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device)
        self.model = ImprovedBindingPredictor().to(device)
        checkpoint = torch.load(os.environ.get('BINDING_MODEL_PATH', 'path/to/binding/best_model.pt'), weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Use HuggingFace transformers API instead of old esm.pretrained API
        from transformers import EsmModel, EsmTokenizer
        esm_model_name = "facebook/esm2_t33_650M_UR50D"
        self.esm_model = EsmModel.from_pretrained(esm_model_name).to(device)
        self.esm_model.eval()
        self.prot_tokenizer = EsmTokenizer.from_pretrained(esm_model_name)

        # Tokenize protein sequence
        encoded = self.prot_tokenizer(prot_seq, return_tensors="pt", add_special_tokens=True)
        prot_tokens = encoded["input_ids"].to(device)
        
        with torch.no_grad():
            # Get hidden states from all layers
            results = self.esm_model(prot_tokens, output_hidden_states=True)
            # Layer 33 is the last layer (0-indexed, so it's hidden_states[33])
            prot_emb = results.hidden_states[33]
            
        self.prot_emb = prot_emb[0]
        self.prot_emb = torch.mean(self.prot_emb, dim=0, keepdim=True).to(device)
                
    def forward(self, x):        
        with torch.no_grad():
            scores = []
            pep_emb = self.pep_model(input_ids=x, output_hidden_states=True).last_hidden_state.mean(dim=1, keepdim=True)
            for pep in pep_emb:
                score, logits = self.model.forward(self.prot_emb, pep)
                scores.append(min(10, score.item()) / 10)
        
        return scores[0]
    
    def __call__(self, smiles_tokens, smiles_seq):
        score = self.forward(smiles_tokens)
        return 'affinity', score

# class Admet_AI:
#     def __init__(self, cfg, smiles_tokenizer, selfies_tokenizer):
#         self.admet = ADMETModel()
#         self.smiles_tokenizer = smiles_tokenizer
#         self.selfies_tokenizer = selfies_tokenizer
#         self.cfg = cfg

#     def __call__(self, x):
#         if self.cfg.task == 'selfies':
#             seq = self.selfies_tokenizer.decode(x[0].tolist()[1:-1])
#         elif self.cfg.task == 'smiles':
#             seq = self.smiles_tokenizer.decode(x[0])
#         else:
#             raise NotImplementedError

#         res = self.admet.predict(smiles=seq)
#         toxicity = 1 - res['ClinTox']
#         solubility = (min(0, max(-4, res['Solubility_AqSolDB'])) + 4) / 4
#         permeability = res['Caco2_Wang']
#         halflife = max(0, min(2, np.log10(max(1e-6, res['Half_Life_Obach'])))) / 2
        
#         ratio = analyze_peptide_likeness(seq)['peptide_like']
        
#         return 'admet', {
#             'toxicity': toxicity,
#             'solubility': solubility,
#             'permeability': permeability,
#             'halflife': halflife,
#             'ratio': ratio,
#         }

class Admetica:
    def __init__(self):
        self.trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="cuda", devices=1)
        self.models = self.load_models(ckpt_dir=os.environ.get('ADMETICA_MODELS_DIR', 'path/to/admetica/Models'))

    def load_models(self, ckpt_dir):
        if not CHEMPROP_AVAILABLE:
            raise ImportError("chemprop is required for Admetica objective but is not available. "
                           "Please install chemprop or use a different objective.")
        toxicity_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'ld50.ckpt'))
        solubility_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'solubility.ckpt'))
        permeability_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'caco2.ckpt'))
        halflife_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'half-life.ckpt'))

        return toxicity_model, solubility_model, permeability_model, halflife_model

    def is_valid_smiles(self, smiles):
        """Check if the given SMILES string is valid."""
        try:
            return Chem.MolFromSmiles(smiles) is not None
        except Exception as e:
            logging.error(f"Error validating SMILES '{smiles}': {str(e)}")
            return False

    def prediction(self, smiles_list, trainer, model):
        if not CHEMPROP_AVAILABLE:
            raise ImportError("chemprop is required for Admetica objective but is not available. "
                           "Please install chemprop or use a different objective.")
        valid_smiles = [smi for smi in smiles_list if self.is_valid_smiles(smi)]
        valid_indices = [i for i, smi in enumerate(smiles_list) if self.is_valid_smiles(smi)]
        invalid_indices = [i for i in range(len(smiles_list)) if i not in valid_indices]

        if not valid_smiles:
            return np.full(len(smiles_list), "", dtype=object)

        test_data = [data.MoleculeDatapoint.from_smi(smi) for smi in valid_smiles]
        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dataset = data.MoleculeDataset(test_data, featurizer=featurizer)
        test_loader = data.build_dataloader(test_dataset, shuffle=False)

        with torch.no_grad():
            predictions = trainer.predict(model, test_loader)
        
        predictions = [pred.item() for batch in predictions for pred in batch]
        for index in invalid_indices:
            predictions.insert(index, "")
        
        return predictions

    def non_toxicity_from_log10mgkg(self, x, lo=1.0, hi=4.0):
        """
        x : predicted log10(mg/kg)
        lo ~ 1  (≈10 mg/kg: very toxic)
        hi ~ 4  (≈10,000 mg/kg: low acute toxicity)
        returns ∈ [0,1]: higher = safer (non-toxic)
        """
        x = max(lo, min(hi, x))
        return (x - lo) / (hi - lo)

    def __call__(self, smiles_tokens, smiles_seq):
        scores = []
        for model in self.models:
            scores.append(self.prediction([smiles_seq], self.trainer, model)[0])

        non_toxicity = self.non_toxicity_from_log10mgkg(scores[0])
        solubility = (max(-12, min(2, scores[1])) + 12) / 14
        permeability = (max(-8, min(-3, scores[2])) + 8) / 5
        halflife = max(0, min(2, np.log10(max(1e-6, scores[3])))) / 2
        
        ratio = analyze_peptide_likeness(smiles_seq)['peptide_likeness']
        
        return 'admet', {
            'non_toxicity': non_toxicity,
            'solubility': solubility,
            'permeability': permeability,
            'halflife': halflife,
            'ratio': ratio,
        }


class Cas9Classification:
    """
    Cas9 binary classification objective.
    
    This objective works with protein sequences (not SMILES).
    It uses ESM-2 embeddings and a trained Cas9Classifier to predict
    whether a protein sequence is Cas9-like.
    
    Note: This objective expects protein sequences, not SMILES sequences.
    If used with SMILES-based generation, you'll need to convert SMILES to proteins first.
    """
    def __init__(self, device, checkpoint_path=None, config_path=None, shared_esm_model=None):
        if not CAS9_PREDICTOR_AVAILABLE:
            raise ImportError(
                "Cas9 predictor modules not available. "
                "Ensure cas_predictor is in the correct location."
            )
        
        self.device = device
        
        # Default checkpoint path if not provided
        if checkpoint_path is None:
            checkpoint_path = (
                os.environ.get("CAS_PREDICTOR_CKPT") or
                "path/to/predictors/cas_predictor/checkpoint/last.ckpt"
            )
        
        # Load config
        config = None
        if config_path:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
                config = edict(config_dict)
        else:
            # Try to load from checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if "hyper_parameters" in checkpoint:
                    config_dict = checkpoint["hyper_parameters"]
                    if isinstance(config_dict, dict):
                        config = edict(config_dict)
            except Exception as e:
                print(f"Warning: Could not load config from checkpoint: {e}")
        
        # If config still not found, create a default config with reasonable defaults
        if config is None:
            print("Warning: Could not load config from checkpoint or config_path. Using default config.")
            config = edict({
                "model": edict({
                    "esm_model_name": "facebook/esm2_t33_650M_UR50D",
                    "freeze_esm": True,
                    "hidden_size": 512,
                    "dropout": 0.1,
                    "pooling_type": "mean"
                }),
                "optim": edict({
                    "lr": 5e-5,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1
                })
            })
        
        # Load model from checkpoint
        # If shared_esm_model is provided, we need to manually instantiate and load
        # to avoid creating a duplicate ESM model
        if shared_esm_model is not None:
            # Manually instantiate with shared ESM model
            self.model = Cas9ClassifierModule(config, shared_esm_model=shared_esm_model)
            # Load checkpoint state dict (excluding ESM weights since we're using shared model)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            # Filter out ESM weights to avoid loading them (we're using shared model)
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                                 if not k.startswith("model.esm_emb")}
            self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded Cas9Classifier checkpoint with shared ESM model. "
                  f"Filtered out {len(state_dict) - len(filtered_state_dict)} ESM parameters.")
        else:
            # Original behavior: use load_from_checkpoint (creates its own ESM model)
            self.model = Cas9ClassifierModule.load_from_checkpoint(
                checkpoint_path,
                config=config,
                strict=False,
            )
        
        self.model.eval()
        self.model.to(device)
        
        # Load ESM tokenizer (same approach as BindingAffinity)
        # We only need the alphabet for tokenization, not the full model
        # If we have a shared ESM model, we can get the tokenizer from it
        # Otherwise, load the ESM model just to get the alphabet
        if shared_esm_model is not None:
            # Use the shared ESM model's tokenizer if available
            # For transformers.EsmModel, we need to use EsmTokenizer separately
            from transformers import EsmTokenizer
            esm_model_name = getattr(config.model, "esm_model_name", "facebook/esm2_t33_650M_UR50D")
            tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
            # Create a simple batch converter function
            def batch_converter(data):
                # data is list of (name, sequence) tuples
                sequences = [seq for _, seq in data]
                encoded = tokenizer(sequences, padding=True, return_tensors="pt", add_special_tokens=True)
                # Return format: (names, sequences, tokens)
                names = [name for name, _ in data]
                return names, sequences, encoded["input_ids"]
            self.batch_converter = batch_converter
        else:
            # Use HuggingFace transformers API instead of old esm.pretrained API
            from transformers import EsmTokenizer
            esm_model_name = getattr(config.model, "esm_model_name", "facebook/esm2_t33_650M_UR50D")
            tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
            # Create a simple batch converter function
            def batch_converter(data):
                # data is list of (name, sequence) tuples
                sequences = [seq for _, seq in data]
                encoded = tokenizer(sequences, padding=True, return_tensors="pt", add_special_tokens=True)
                # Return format: (names, sequences, tokens)
                names = [name for name, _ in data]
                return names, sequences, encoded["input_ids"]
            self.batch_converter = batch_converter
    
    def get_scores(self, protein_seqs):
        """
        Get Cas9 classification scores for protein sequences.
        
        Args:
            protein_seqs: List of protein sequence strings (amino acid sequences) or single string
        
        Returns:
            scores: List of probabilities (0-1) that each sequence is Cas9-like
        """
        # Handle single string input
        if isinstance(protein_seqs, str):
            protein_seqs = [protein_seqs]
        
        if not protein_seqs:
            return []
        
        scores = []
        
        with torch.no_grad():
            # Tokenize sequences using ESM batch converter
            data = [(f"seq_{i}", seq) for i, seq in enumerate(protein_seqs)]
            _, _, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            # Create attention mask (True for valid tokens, False for padding)
            # ESM uses 1 for padding, so we check for non-padding tokens
            # Convert to bool tensor explicitly (model expects BoolTensor)
            attention_mask = (batch_tokens != 1).bool().to(self.device)  # 1 is ESM's padding token
            
            # Debug: Print shapes and values for first sequence if debugging
            if len(protein_seqs) == 1 and len(protein_seqs[0]) < 2000:  # Only debug for single sequences and reasonable lengths
                import os
                if os.environ.get("DEBUG_CAS9", "0") == "1":
                    print(f"[DEBUG] Cas9 classifier - sequence length: {len(protein_seqs[0])}")
                    print(f"[DEBUG] Cas9 classifier - batch_tokens shape: {batch_tokens.shape}")
                    print(f"[DEBUG] Cas9 classifier - attention_mask shape: {attention_mask.shape}")
                    print(f"[DEBUG] Cas9 classifier - attention_mask sum (valid tokens): {attention_mask.sum().item()}")
                    print(f"[DEBUG] Cas9 classifier - batch_tokens min/max: {batch_tokens.min().item()}/{batch_tokens.max().item()}")
            
            # Forward pass
            logits = self.model(batch_tokens, attention_mask)
            
            # Debug: Print logits if debugging
            if len(protein_seqs) == 1 and len(protein_seqs[0]) < 2000:
                import os
                if os.environ.get("DEBUG_CAS9", "0") == "1":
                    print(f"[DEBUG] Cas9 classifier - logits shape: {logits.shape}")
                    print(f"[DEBUG] Cas9 classifier - logits value: {logits.item()}")
            
            probs = torch.sigmoid(logits).cpu()
            
            # Debug: Print probabilities if debugging
            if len(protein_seqs) == 1 and len(protein_seqs[0]) < 2000:
                import os
                if os.environ.get("DEBUG_CAS9", "0") == "1":
                    print(f"[DEBUG] Cas9 classifier - probabilities: {probs.tolist()}")
            
            scores = probs.tolist()
        
        return scores
    
    def __call__(self, protein_tokens, protein_seqs):
        """
        Objective call interface.
        
        Args:
            protein_tokens: Unused (kept for interface compatibility)
            protein_seqs: List of protein sequence strings
        
        Returns:
            Tuple of ('cas9', scores) where scores is a list of probabilities
        """
        scores = self.get_scores(protein_seqs)
        return 'cas9', scores


class DeletionCount:
    """
    Objective that maximizes the number of deletions (length reduction) in protein sequences.
    
    This objective computes the difference between the original sequence length and
    the current sequence length, normalized by a maximum deletion percentage to produce
    a value between 0 and 1.
    
    Args:
        original_seq: The original protein sequence string (before any edits)
        max_deletion_percentage: Maximum deletion percentage (0-1). The objective value will be computed as
                                 (deletion_count / (original_length * max_deletion_percentage)), clamped to [0, 1].
                                 If None, defaults to 1.0 (100% deletion possible).
    """
    def __init__(self, original_seq, max_deletion_percentage=None):
        # Store original sequence length (excluding spaces)
        self.original_length = len(original_seq.replace(' ', ''))
        # Set max_deletion_percentage to 1.0 (100%) if not provided
        if max_deletion_percentage is None:
            self.max_deletion_percentage = 1.0
        else:
            if max_deletion_percentage <= 0 or max_deletion_percentage > 1.0:
                raise ValueError(f"max_deletion_percentage must be in (0, 1], got {max_deletion_percentage}")
            self.max_deletion_percentage = float(max_deletion_percentage)
        
        # Calculate max_deletion as absolute value for backward compatibility
        self.max_deletion = self.original_length * self.max_deletion_percentage
    
    def __call__(self, protein_tokens, protein_seqs):
        """
        Objective call interface.
        
        Args:
            protein_tokens: Unused (kept for interface compatibility)
            protein_seqs: List of protein sequence strings
        
        Returns:
            Tuple of ('deletion_count', scores) where scores is a list of normalized deletion
            percentages (deletion_count / (original_length * max_deletion_percentage)), clamped to [0, 1]
        """
        # Handle single string input
        if isinstance(protein_seqs, str):
            protein_seqs = [protein_seqs]
        
        if not protein_seqs:
            return 'deletion_count', []
        
        scores = []
        for seq in protein_seqs:
            # Remove spaces and compute current length
            current_length = len(seq.replace(' ', ''))
            # Number of deletions = original_length - current_length
            deletion_count = self.original_length - current_length
            # Normalize by max_deletion and clamp to [0, 1]
            normalized_score = max(0.0, min(1.0, deletion_count / self.max_deletion))
            scores.append(float(normalized_score))
        
        return 'deletion_count', scores


class PAMMatching:
    """
    PAM matching objective using temperature-scaled log probabilities.
    
    This objective uses the HuggingFace protein2pam model (cas9_full) to predict PAM logits
    for protein sequences. The score is computed using temperature-scaled log probabilities:
    at each target position (non-N), we compute log P(target_nucleotide | position) after
    applying temperature scaling to the logits. The final score is the geometric mean of
    probabilities (exp of mean log prob) over target positions.
    
    This approach amplifies small improvements, making them more visible in the weighted sum
    during optimization, which helps when candidates show only marginal improvements.
    
    Args:
        device: torch device
        target_pam: Target PAM sequence string of length 10 (e.g., "NGGNNNNNNN")
        model_name: HuggingFace model identifier (default: "Profluent-Bio/protein2pam-cas9_full")
        sigmoid_temperature: Temperature for scaling logits before softmax. Lower values (0.1-0.3)
                            sharpen the distribution, making small improvements more visible.
                            Default 0.2. (Note: parameter name kept for backward compatibility)
        pam_prediction_temperature: Temperature for PAM prediction (in predict_pam method).
                                    Values < 1.0 make distributions sharper (more confident),
                                    > 1.0 make them softer. Default 1.0 (no temperature scaling).
                                    Note: This only affects prediction, not scoring.
    """
    def __init__(self, device, target_pam, model_name="Profluent-Bio/protein2pam-cas9_full", sigmoid_temperature=0.2, use_entropy_for_n_positions=True, use_ce_loss=False, pam_min_confidence=0.55, pam_prediction_temperature=1.0):
        self.device = device
        self.target_pam = target_pam.upper()
        
        # Validate target PAM length
        if len(self.target_pam) != 10:
            raise ValueError(f"Target PAM must be exactly 10 nucleotides, got {len(self.target_pam)}")
        
        # Validate nucleotides
        valid_nucleotides = set('ACGTN')
        if not all(nuc in valid_nucleotides for nuc in self.target_pam):
            raise ValueError(f"Target PAM contains invalid nucleotides. Only ACGTN allowed.")
        
        # Convert target PAM to class indices (ACGT = 0,1,2,3, N = -1 for masking)
        nucleotides = ['A', 'C', 'G', 'T']
        self.target_indices = []
        self.target_mask = []
        for nuc in self.target_pam:
            if nuc == 'N':
                # N means any nucleotide - we'll mask this position in loss calculation
                self.target_indices.append(-1)  # Placeholder, will be masked
                self.target_mask.append(False)
            else:
                self.target_indices.append(nucleotides.index(nuc))
                self.target_mask.append(True)
        
        self.target_indices = torch.tensor(self.target_indices, device=device, dtype=torch.long)
        self.target_mask = torch.tensor(self.target_mask, device=device, dtype=torch.bool)
        self.sigmoid_temperature = float(sigmoid_temperature)
        self.use_entropy_for_n_positions = bool(use_entropy_for_n_positions)
        self.use_ce_loss = bool(use_ce_loss)
        self.pam_min_confidence = float(pam_min_confidence)
        self.pam_prediction_temperature = float(pam_prediction_temperature)
        
        # Load HuggingFace model
        # First, verify the protein2pam path exists
        # Try multiple strategies to find predictors/protein2pam-0.2.0:
        # 1. Try 3 levels up (most common: editflows/cope_models/objectives.py -> parent of editflows/)
        # 2. Try 5 levels up (original expected structure)
        # 3. Try other levels (3-8) as fallback
        
        protein2pam_path_resolved = None
        current_file = Path(__file__).resolve()
        
        # Strategy 1: Try 3 levels up first (for structure: editflows/cope_models/objectives.py, predictors/ is sibling of editflows/)
        test_path_3 = current_file.parent.parent.parent / "predictors" / "protein2pam-0.2.0"
        if test_path_3.exists():
            protein2pam_path_resolved = test_path_3.resolve()
            print(f"[DEBUG] Found protein2pam at 3 levels up: {protein2pam_path_resolved}")
        
        # Strategy 2: Try 5 levels up (original expected structure)
        if protein2pam_path_resolved is None:
            test_path_5 = current_file.parent.parent.parent.parent.parent / "predictors" / "protein2pam-0.2.0"
            if test_path_5.exists():
                protein2pam_path_resolved = test_path_5.resolve()
                print(f"[DEBUG] Found protein2pam at 5 levels up: {protein2pam_path_resolved}")
        
        # Strategy 3: Try other levels (3-8) as fallback
        if protein2pam_path_resolved is None:
            for levels_up in range(3, 9):
                if levels_up == 3 or levels_up == 5:  # Skip already checked
                    continue
                alt_path = current_file
                for _ in range(levels_up):
                    alt_path = alt_path.parent
                test_path = alt_path / "predictors" / "protein2pam-0.2.0"
                if test_path.exists():
                    protein2pam_path_resolved = test_path.resolve()
                    print(f"[DEBUG] Found protein2pam at {levels_up} levels up: {protein2pam_path_resolved}")
                    break
        
        if protein2pam_path_resolved is None:
            raise ImportError(
                f"protein2pam-0.2.0 directory not found. "
                f"Expected at: {protein2pam_path} "
                f"Please ensure protein2pam-0.2.0 is in the correct location relative to this file."
            )
        
        # Ensure the path is in sys.path
        protein2pam_path_str = str(protein2pam_path_resolved)
        if protein2pam_path_str not in sys.path:
            sys.path.insert(0, protein2pam_path_str)
            print(f"[DEBUG] Added protein2pam path to sys.path: {protein2pam_path_str}")
        
        try:
            # Try importing directly from huggingface submodule to avoid protein2pam/__init__.py
            # which imports PAMOracle (requires torch at import time)
            import importlib.util
            
            # Import the modules directly without going through __init__.py
            huggingface_dir = protein2pam_path_resolved / "protein2pam" / "huggingface"
            
            # Load configuration_esm
            config_file = huggingface_dir / "configuration_esm.py"
            if config_file.exists():
                spec_config = importlib.util.spec_from_file_location("protein2pam.huggingface.configuration_esm", config_file)
                if spec_config and spec_config.loader:
                    # Create package structure
                    if 'protein2pam' not in sys.modules:
                        sys.modules['protein2pam'] = type(sys)('protein2pam')
                    if 'protein2pam.huggingface' not in sys.modules:
                        sys.modules['protein2pam.huggingface'] = type(sys)('protein2pam.huggingface')
                    
                    config_module = importlib.util.module_from_spec(spec_config)
                    sys.modules['protein2pam.huggingface.configuration_esm'] = config_module
                    spec_config.loader.exec_module(config_module)
            
            # Load modeling_esm (depends on configuration_esm)
            modeling_file = huggingface_dir / "modeling_esm.py"
            if modeling_file.exists():
                spec_modeling = importlib.util.spec_from_file_location("protein2pam.huggingface.modeling_esm", modeling_file)
                if spec_modeling and spec_modeling.loader:
                    modeling_module = importlib.util.module_from_spec(spec_modeling)
                    sys.modules['protein2pam.huggingface.modeling_esm'] = modeling_module
                    spec_modeling.loader.exec_module(modeling_module)
            
            # Load tokenizer
            tokenizer_file = huggingface_dir / "tokenizer.py"
            if tokenizer_file.exists():
                spec_tokenizer = importlib.util.spec_from_file_location("protein2pam.huggingface.tokenizer", tokenizer_file)
                if spec_tokenizer and spec_tokenizer.loader:
                    tokenizer_module = importlib.util.module_from_spec(spec_tokenizer)
                    sys.modules['protein2pam.huggingface.tokenizer'] = tokenizer_module
                    spec_tokenizer.loader.exec_module(tokenizer_module)
            
            # Now import from the loaded modules
            EsmForSequenceClassification = sys.modules['protein2pam.huggingface.modeling_esm'].EsmForSequenceClassification
            get_tokenizer = sys.modules['protein2pam.huggingface.tokenizer'].get_tokenizer
            
            self.tokenizer = get_tokenizer()
            print(f"Loading PAM prediction model: {model_name}...")
            self.model = EsmForSequenceClassification.from_pretrained(model_name, device_map=device)
            self.model.eval()
            print(f"PAM prediction model loaded successfully.")
        except ImportError as e:
            # Show more details about the import error
            import traceback
            print(f"[DEBUG] protein2pam import error: {e}")
            print(f"[DEBUG] sys.path entries containing 'protein2pam': {[p for p in sys.path if 'protein2pam' in p]}")
            print(f"[DEBUG] Expected protein2pam path: {protein2pam_path_resolved}")
            print(f"[DEBUG] Path exists: {protein2pam_path_resolved.exists() if protein2pam_path_resolved else False}")
            traceback.print_exc()
            raise ImportError(
                f"protein2pam package not available: {e}. "
                "Please ensure protein2pam is installed and in the correct location."
            )
        except Exception as e:
            import traceback
            print(f"[DEBUG] Failed to load PAM prediction model: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to load PAM prediction model: {e}")
    
    def get_pam_probability_distributions(self, protein_seqs):
        """
        Get detailed probability distributions for PAM predictions.
        
        Args:
            protein_seqs: List of protein sequence strings
        
        Returns:
            List of dictionaries, each containing:
            - 'probabilities': (10, 4) tensor of probabilities for each position and nucleotide
            - 'predicted_pam': Predicted PAM string
            - 'per_position': List of dicts with position info (nucleotide probs, entropy, etc.)
        """
        if not protein_seqs:
            return []
        
        # Handle single string input
        if isinstance(protein_seqs, str):
            protein_seqs = [protein_seqs]
        
        nucleotides = ['A', 'C', 'G', 'T']
        results = []
        
        with torch.no_grad():
            # Tokenize sequences
            encodings = self.tokenizer.encode_batch(protein_seqs)
            input_batch = dict(
                input_ids=torch.tensor([encoding.ids for encoding in encodings], device=self.device),
                attention_mask=torch.tensor([encoding.attention_mask for encoding in encodings], device=self.device),
            )
            
            # Get PAM predictions
            output = self.model(**input_batch)
            logits = output.logits  # (batch_size, 10, 4)
            probabilities = F.softmax(logits, dim=-1)  # (batch_size, 10, 4)
            log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, 10, 4)
            
            # Process each sequence
            for i in range(probabilities.shape[0]):
                probs = probabilities[i]  # (10, 4)
                log_probs_seq = log_probs[i]  # (10, 4)
                pam_seq = []
                per_position = []
                
                for pos_idx in range(probs.shape[0]):
                    pos_probs = probs[pos_idx, :]  # (4,)
                    pos_log_probs = log_probs_seq[pos_idx, :]  # (4,)
                    
                    # Compute entropy for this position
                    entropy = -(pos_probs * pos_log_probs).sum().item()
                    max_entropy = math.log(4)  # Maximum entropy for uniform distribution
                    normalized_entropy = entropy / max_entropy
                    
                    max_prob = pos_probs.max().item()
                    max_idx = pos_probs.argmax().item()
                    predicted_nuc = nucleotides[max_idx]
                    
                    # Create position info
                    pos_info = {
                        'position': pos_idx,
                        'probabilities': {nuc: pos_probs[j].item() for j, nuc in enumerate(nucleotides)},
                        'predicted': predicted_nuc,
                        'max_probability': max_prob,
                        'entropy': entropy,
                        'normalized_entropy': normalized_entropy,
                    }
                    per_position.append(pos_info)
                    
                    # Predict nucleotide (use instance min_confidence)
                    if max_prob < self.pam_min_confidence:
                        pam_seq.append('N')
                    else:
                        pam_seq.append(predicted_nuc)
                
                results.append({
                    'probabilities': probs.cpu(),
                    'predicted_pam': ''.join(pam_seq),
                    'per_position': per_position,
                })
        
        return results
    
    def predict_pam(self, protein_seqs, min_confidence=None):
        """
        Predict PAM sequences for protein sequences, with support for 'N' when confidence is low.
        
        Uses temperature scaling (pam_prediction_temperature) to control distribution sharpness.
        Lower temperatures (< 1.0) produce sharper distributions and more confident predictions.
        
        Args:
            protein_seqs: List of protein sequence strings
            min_confidence: Minimum probability threshold for predicting a specific nucleotide.
                           If max probability < min_confidence, predict 'N'.
                           If None, uses self.pam_min_confidence (default: None)
        
        Returns:
            predicted_pams: List of predicted PAM strings (10 nucleotides each)
        """
        if min_confidence is None:
            min_confidence = self.pam_min_confidence
        if not protein_seqs:
            return []
        
        # Handle single string input
        if isinstance(protein_seqs, str):
            protein_seqs = [protein_seqs]
        
        # Check if model and tokenizer are initialized
        if not hasattr(self, 'model') or self.model is None:
            print(f"[ERROR] PAMMatching.predict_pam: model is not initialized. Cannot predict PAM.")
            return []
        
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            print(f"[ERROR] PAMMatching.predict_pam: tokenizer is not initialized. Cannot predict PAM.")
            return []
        
        predicted_pams = []
        nucleotides = ['A', 'C', 'G', 'T']
        
        try:
            with torch.no_grad():
                # Tokenize sequences
                encodings = self.tokenizer.encode_batch(protein_seqs)
                input_batch = dict(
                    input_ids=torch.tensor([encoding.ids for encoding in encodings], device=self.device),
                    attention_mask=torch.tensor([encoding.attention_mask for encoding in encodings], device=self.device),
                )
                
                # Get PAM predictions
                output = self.model(**input_batch)
                logits = output.logits  # (batch_size, 10, 4) - 10 positions, 4 nucleotides (ACGT)
                # Apply temperature scaling for prediction (sharper distributions when < 1.0)
                scaled_logits = logits / self.pam_prediction_temperature
                probabilities = F.softmax(scaled_logits, dim=-1)  # (batch_size, 10, 4)
                
                # Predict PAM for each sequence
                for i in range(probabilities.shape[0]):
                    probs = probabilities[i]  # (10, 4)
                    pam_seq = []
                    
                    for pos_idx in range(probs.shape[0]):
                        pos_probs = probs[pos_idx, :]  # (4,)
                        max_prob = pos_probs.max().item()
                        max_idx = pos_probs.argmax().item()
                        
                        # If confidence is low, predict 'N'
                        if max_prob < min_confidence:
                            pam_seq.append('N')
                        else:
                            pam_seq.append(nucleotides[max_idx])
                    
                    predicted_pams.append(''.join(pam_seq))
        except Exception as e:
            import traceback
            print(f"[ERROR] PAMMatching.predict_pam failed: {e}")
            traceback.print_exc()
            return []
        
        return predicted_pams
    
    def get_scores(self, protein_seqs):
        """
        Get PAM matching scores using either cross-entropy loss or log probability approach.
        
        If use_ce_loss=True:
            Uses cross-entropy loss between predicted logits and target PAM indices.
            Score = exp(-mean_ce_loss) to convert to [0, 1] range (higher is better).
        
        If use_ce_loss=False (default):
            Uses raw probabilities (no temperature scaling) to maintain high probability values.
            For non-N positions: score is the geometric mean of probabilities (exp of mean log prob).
            For N positions: if use_entropy_for_n_positions=True, score is based on entropy.
        
        Args:
            protein_seqs: List of protein sequence strings
        
        Returns:
            scores: List of scores in [0, 1] (higher = better match to target PAM)
        """
        if not protein_seqs:
            return []
        
        # Handle single string input
        if isinstance(protein_seqs, str):
            protein_seqs = [protein_seqs]
        
        scores = []
        max_entropy = math.log(4)  # Maximum entropy for uniform distribution over 4 nucleotides
        
        with torch.no_grad():
            # Tokenize sequences
            encodings = self.tokenizer.encode_batch(protein_seqs)
            input_batch = dict(
                input_ids=torch.tensor([encoding.ids for encoding in encodings], device=self.device),
                attention_mask=torch.tensor([encoding.attention_mask for encoding in encodings], device=self.device),
            )
            
            # Get PAM predictions (logits)
            output = self.model(**input_batch)
            logits = output.logits  # (batch_size, 10, 4) - 10 positions, 4 nucleotides (ACGT)
            
            for i in range(logits.shape[0]):
                seq_logits = logits[i]  # (10, 4)
                
                if self.use_ce_loss:
                    # Cross-entropy loss approach
                    # Separate non-N and N positions
                    non_n_mask = self.target_mask  # True for non-N positions
                    n_mask = ~non_n_mask  # True for N positions
                    
                    score_components = []
                    
                    # For non-N positions: compute cross-entropy loss against target nucleotides
                    if non_n_mask.any():
                        masked_logits = seq_logits[non_n_mask]  # (K, 4)
                        masked_targets = self.target_indices[non_n_mask]  # (K,) in {0,1,2,3}
                        
                        # Compute cross-entropy loss per position (reduction='none')
                        ce_loss_per_pos = F.cross_entropy(
                            masked_logits, masked_targets, reduction='none'
                        )  # (K,)
                        
                        # Mean cross-entropy loss over non-N positions
                        mean_ce_loss = ce_loss_per_pos.mean().item()
                        
                        # Convert to score in [0, 1] range using exp(-ce_loss)
                        # Lower CE loss = higher score (better match)
                        non_n_score = math.exp(-mean_ce_loss)
                        score_components.append(non_n_score)
                    
                    # For N positions: handle based on entropy flag
                    if n_mask.any():
                        if self.use_entropy_for_n_positions:
                            # Use entropy scoring (encourage uniform distribution)
                            log_probs = F.log_softmax(seq_logits, dim=-1)  # (10, 4)
                            probs = torch.exp(log_probs)  # (10, 4)
                            n_probs = probs[n_mask]  # (M, 4) where M is number of N positions
                            n_log_probs = log_probs[n_mask]  # (M, 4)
                            entropy_per_pos = -(n_probs * n_log_probs).sum(dim=-1)  # (M,)
                            mean_entropy = entropy_per_pos.mean().item()
                            n_score = mean_entropy / max_entropy
                            score_components.append(n_score)
                        else:
                            # Use CE loss against uniform distribution (consistent with CE loss approach)
                            # This encourages uniform distribution over all 4 nucleotides
                            n_logits = seq_logits[n_mask]  # (M, 4) where M is number of N positions
                            
                            # Compute cross-entropy loss against uniform distribution [0.25, 0.25, 0.25, 0.25]
                            # CE_uniform = -sum(0.25 * log(p_i)) = -0.25 * sum(log(p_i))
                            # Lower CE loss (closer to uniform) = higher score
                            log_probs = F.log_softmax(n_logits, dim=-1)  # (M, 4)
                            uniform_target = 0.25  # Uniform probability for each nucleotide
                            ce_loss_uniform_per_pos = -(uniform_target * log_probs).sum(dim=-1)  # (M,)
                            
                            # Mean cross-entropy loss over N positions
                            mean_ce_loss_uniform = ce_loss_uniform_per_pos.mean().item()
                            
                            # Convert to score in [0, 1] range using exp(-ce_loss)
                            # Lower CE loss (more uniform) = higher score
                            n_score = math.exp(-mean_ce_loss_uniform)
                            score_components.append(n_score)
                    
                    # Combine scores: if both non-N and N positions exist, take weighted average
                    # Weight by number of positions of each type
                    if len(score_components) == 2:
                        num_non_n = non_n_mask.sum().item()
                        num_n = n_mask.sum().item()
                        total_positions = num_non_n + num_n
                        score = (score_components[0] * num_non_n + score_components[1] * num_n) / total_positions
                    elif len(score_components) == 1:
                        score = score_components[0]
                    else:
                        score = 0.0
                    
                    scores.append(score)
                else:
                    # Original log probability approach (default)
                    # Use raw probabilities (no temperature scaling) to maintain high probabilities
                    # This treats all probability values equally, appropriate for maintaining
                    # high probability rather than optimizing low probabilities upward
                    log_probs = F.log_softmax(seq_logits, dim=-1)  # (10, 4)
                    probs = torch.exp(log_probs)  # (10, 4) - needed for entropy calculation
                    
                    # Separate non-N and N positions
                    non_n_mask = self.target_mask  # True for non-N positions
                    n_mask = ~non_n_mask  # True for N positions
                    
                    score_components = []
                    
                    # For non-N positions: maximize log probability of target nucleotide
                    if non_n_mask.any():
                        masked_log_probs = log_probs[non_n_mask]  # (K, 4)
                        masked_targets = self.target_indices[non_n_mask]  # (K,) in {0,1,2,3}
                        
                        # Get log probability of target nucleotide at each position
                        target_log_probs = masked_log_probs.gather(
                            1, masked_targets.unsqueeze(1)
                        ).squeeze(1)  # (K,)
                        
                        # Mean log probability (geometric mean in probability space)
                        mean_log_prob = target_log_probs.mean().item()
                        
                        # Convert to score in [0, 1] range using geometric mean probability
                        # This gives equal weight to maintaining high probabilities
                        non_n_score = math.exp(mean_log_prob)
                        score_components.append(non_n_score)
                    
                    # For N positions: maximize entropy (encourage uniform distribution) if enabled
                    if n_mask.any() and self.use_entropy_for_n_positions:
                        n_probs = probs[n_mask]  # (M, 4) where M is number of N positions
                        
                        # Compute entropy for each N position: H = -sum(p_i * log(p_i))
                        # Use log_probs for numerical stability: H = -sum(p_i * log_p_i)
                        n_log_probs = log_probs[n_mask]  # (M, 4)
                        entropy_per_pos = -(n_probs * n_log_probs).sum(dim=-1)  # (M,)
                        
                        # Mean entropy across N positions
                        mean_entropy = entropy_per_pos.mean().item()
                        
                        # Normalize by max entropy to get score in [0, 1]
                        # Higher entropy (more uniform) = higher score
                        n_score = mean_entropy / max_entropy
                        score_components.append(n_score)
                    
                    # Combine scores: if both non-N and N positions exist, take weighted average
                    # Weight by number of positions of each type
                    # If entropy is disabled, only use non-N score
                    if len(score_components) == 2:
                        # Weighted average based on number of positions
                        num_non_n = non_n_mask.sum().item()
                        num_n = n_mask.sum().item()
                        total_positions = num_non_n + num_n
                        score = (score_components[0] * num_non_n + score_components[1] * num_n) / total_positions
                    elif len(score_components) == 1:
                        score = score_components[0]
                    else:
                        # No positions to evaluate (shouldn't happen, but handle gracefully)
                        score = 0.0
                    
                    scores.append(score)
        
        return scores
    
    def get_score_for_pam(self, protein_seqs, pam_sequence, use_temperature_scaling=False):
        """
        Get probability score for a specific PAM sequence (not necessarily the target).
        
        This computes the log probability score for an arbitrary PAM sequence.
        By default, uses raw probabilities (no temperature scaling) for more
        interpretable results. Set use_temperature_scaling=True to match the
        optimization objective.
        
        Args:
            protein_seqs: List of protein sequence strings
            pam_sequence: PAM sequence string of length 10 (e.g., "NGGNNNNNNN")
            use_temperature_scaling: If True, apply temperature scaling (default: False for display)
        
        Returns:
            scores: List of scores in [0, 1] (probability of the given PAM sequence)
        """
        if not protein_seqs:
            return []
        
        # Handle single string input
        if isinstance(protein_seqs, str):
            protein_seqs = [protein_seqs]
        
        # Validate and convert PAM sequence to indices
        pam_sequence = pam_sequence.upper()
        if len(pam_sequence) != 10:
            raise ValueError(f"PAM sequence must be exactly 10 nucleotides, got {len(pam_sequence)}")
        
        nucleotides = ['A', 'C', 'G', 'T']
        pam_indices = []
        pam_mask = []
        for nuc in pam_sequence:
            if nuc == 'N':
                pam_indices.append(-1)
                pam_mask.append(False)
            else:
                if nuc not in nucleotides:
                    raise ValueError(f"Invalid nucleotide in PAM sequence: {nuc}")
                pam_indices.append(nucleotides.index(nuc))
                pam_mask.append(True)
        
        pam_indices = torch.tensor(pam_indices, device=self.device, dtype=torch.long)
        pam_mask = torch.tensor(pam_mask, device=self.device, dtype=torch.bool)
        
        scores = []
        
        with torch.no_grad():
            # Tokenize sequences
            encodings = self.tokenizer.encode_batch(protein_seqs)
            input_batch = dict(
                input_ids=torch.tensor([encoding.ids for encoding in encodings], device=self.device),
                attention_mask=torch.tensor([encoding.attention_mask for encoding in encodings], device=self.device),
            )
            
            # Get PAM predictions (logits)
            output = self.model(**input_batch)
            logits = output.logits  # (batch_size, 10, 4) - 10 positions, 4 nucleotides (ACGT)
            
            for i in range(logits.shape[0]):
                seq_logits = logits[i]  # (10, 4)
                
                # Apply temperature scaling only if requested (for display, use raw probabilities)
                if use_temperature_scaling:
                    scaled_logits = seq_logits / self.sigmoid_temperature
                    log_probs = F.log_softmax(scaled_logits, dim=-1)  # (10, 4)
                else:
                    # Use raw probabilities (no temperature scaling) for more interpretable results
                    log_probs = F.log_softmax(seq_logits, dim=-1)  # (10, 4)
                
                if pam_mask.any():
                    masked_log_probs = log_probs[pam_mask]  # (K, 4)
                    masked_pam_indices = pam_indices[pam_mask]  # (K,) in {0,1,2,3}
                    
                    # Get log probability of PAM nucleotide at each position
                    pam_log_probs = masked_log_probs.gather(
                        1, masked_pam_indices.unsqueeze(1)
                    ).squeeze(1)  # (K,)
                    
                    # Mean log probability (linear in log space)
                    mean_log_prob = pam_log_probs.mean().item()
                    
                    # Convert to score in [0, 1] range using geometric mean probability
                    score = math.exp(mean_log_prob)
                else:
                    # All positions are N: use neutral score
                    score = 0.0
                
                scores.append(score)
        
        return scores
        
        # ========================================================================
        # OLD IMPLEMENTATION (logit-margin-based with sigmoid) - COMMENTED OUT
        # ========================================================================
        # This was the previous implementation that struggled to show small improvements
        # because sigmoid saturates quickly. Kept for reference.
        #
        # def get_scores_old(self, protein_seqs):
        #     """
        #     Get PAM matching scores (logit-margin-based, in [0, 1]) for protein sequences.
        #     
        #     At each target (non-N) position: margin = (target PAM logit) - (largest non-target logit),
        #     then score_pos = sigmoid(margin / sigmoid_temperature). The sequence score is the mean
        #     over those positions. Higher is better.
        #     
        #     Args:
        #         protein_seqs: List of protein sequence strings
        #     
        #     Returns:
        #         scores: List of scores in [0, 1] (higher = target PAM preferred over alternatives)
        #     """
        #     if not protein_seqs:
        #         return []
        #     
        #     # Handle single string input
        #     if isinstance(protein_seqs, str):
        #         protein_seqs = [protein_seqs]
        #     
        #     scores = []
        #     
        #     with torch.no_grad():
        #         # Tokenize sequences
        #         encodings = self.tokenizer.encode_batch(protein_seqs)
        #         input_batch = dict(
        #             input_ids=torch.tensor([encoding.ids for encoding in encodings], device=self.device),
        #             attention_mask=torch.tensor([encoding.attention_mask for encoding in encodings], device=self.device),
        #         )
        #         
        #         # Get PAM predictions (logits)
        #         output = self.model(**input_batch)
        #         logits = output.logits  # (batch_size, 10, 4) - 10 positions, 4 nucleotides (ACGT)
        #         
        #         for i in range(logits.shape[0]):
        #             seq_logits = logits[i]  # (10, 4)
        #             masked_positions = self.target_mask
        #             if masked_positions.any():
        #                 masked_logits = seq_logits[masked_positions]  # (K, 4)
        #                 masked_targets = self.target_indices[masked_positions]  # (K,) in {0,1,2,3}
        #                 # Target logit at each position
        #                 target_logits = masked_logits.gather(1, masked_targets.unsqueeze(1)).squeeze(1)  # (K,)
        #                 # Largest non-target logit: mask out target class then max over dim=-1
        #                 logits_copy = masked_logits.clone()
        #                 logits_copy.scatter_(1, masked_targets.unsqueeze(1), -1e9)
        #                 max_non_target = logits_copy.max(dim=1).values  # (K,)
        #                 margin = target_logits - max_non_target  # (K,)
        #                 margin = margin / self.sigmoid_temperature
        #                 score_per_pos = torch.sigmoid(margin)
        #                 score = score_per_pos.mean().item()
        #             else:
        #                 # All positions are N: no target to match, use neutral score
        #                 score = 0.5
        #             scores.append(score)
        #     
        #     return scores
    
    def __call__(self, protein_tokens, protein_seqs):
        """
        Objective call interface.
        
        Args:
            protein_tokens: Unused (kept for interface compatibility)
            protein_seqs: List of protein sequence strings
        
        Returns:
            Tuple of ('pam_matching', scores) where scores is a list of values in [0, 1]:
            - For non-N positions: temperature-scaled log probability (geometric mean over positions)
            - For N positions: if use_entropy_for_n_positions=True, normalized entropy (encouraging uniform distribution)
            - If use_entropy_for_n_positions=False, N positions are ignored (score only considers non-N positions)
            - Combined score is weighted average if both types exist and entropy is enabled
            Higher scores indicate better match to target PAM.
        """
        scores = self.get_scores(protein_seqs)
        return 'pam_matching', scores

