import torch
import logging
import pickle
from rdkit import Chem
import chemprop
from lightning import pytorch as pl
from transformers import AutoTokenizer, AutoModelWithLMHead, EsmModel
from BindEvaluator_models import * 
from DeepDTAGen_models import *

import sys
sys.path.append('/scratch/pCoMol/')
from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

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
sys.path.append('/scratch/ReDi_discrete/smiles')
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


class TransformerClassifier(nn.Module):
    def __init__(self, d_model=256, nhead=8, layers=2, ff=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(768, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, X, M):
        # X: (B,L,768), M: (B,L) bool, True=valid token, False=pad/special
        pad_mask = ~M  # True = ignore
        Z = self.proj(X)
        Z = self.enc(Z, src_key_padding_mask=pad_mask)

        Mf = M.unsqueeze(-1).float()
        denom = Mf.sum(dim=1).clamp(min=1.0)
        pooled = (Z * Mf).sum(dim=1) / denom
        return self.head(pooled).squeeze(-1)  # logits


class Toxicity:
    def __init__(self, device):
        self.device = device

        self.tokenizer = SMILES_SPE_Tokenizer(
            '/scratch/pCoMol/smiles_tokenizer/new_vocab.txt', 
            '/scratch/pCoMol/smiles_tokenizer/new_splits.txt'
        )
        self.embedding_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer
        self.embedding_model.to(self.device).eval()

        ckpt = torch.load('/scratch/pCoMol/peptidomimetics/ckpt/toxicity.pt', map_location="cuda", weights_only=False)
        best_params = ckpt["best_params"]
        self.best_params = dict(best_params)

        self.classifier = TransformerClassifier(
            d_model=int(best_params["d_model"]),
            nhead=int(best_params["nhead"]),
            layers=int(best_params["layers"]),
            ff=int(best_params["ff"]),
            dropout=float(best_params.get("dropout", 0.1)),
        )
        self.classifier.load_state_dict(ckpt["state_dict"])
        self.classifier.to(self.device).eval()

    @torch.no_grad()
    def _embed_unpooled(self, smiles_tokens):
        attention_mask = (smiles_tokens != 0).to(self.device)  # (B,L)

        out = self.embedding_model(input_ids=smiles_tokens, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state.float()                           # (B,L,768)

        return last_hidden, attention_mask
    
    @torch.no_grad()
    def predict_proba(self, smiles_tokens):
        X, M = self._embed_unpooled(smiles_tokens)
        logits = self.classifier(X, M)            # (B,)
        probs = torch.sigmoid(logits).tolist()

        return probs
    
    def predict_label(self, smiles, threshold = 0.5):
        p = self.predict_proba(smiles)
        if isinstance(p, float):
            return int(p >= threshold)
        return (p >= threshold).astype(np.int64)

    def __call__(self, smiles_tokens, smiles_seqs):
        return 'non_toxicity', [1 - score for score in self.predict_proba(smiles_tokens)]

class Solubility:
    
    def __init__(self, device):
        self.predictor = xgb.Booster(model_file='/scratch/PeptiVerse/src/solubility/best_model_f1.json')
        self.emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device)
        self.emb_model.eval()

        self.device = device
    
    def get_scores(self, x):
        # pdb.set_trace()
        scores = np.zeros(len(x))
        attention_mask = (x != 0).to(self.device)
        features = np.array(self.emb_model(input_ids=x, attention_mask=attention_mask).last_hidden_state.mean(dim=1).detach().cpu())
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores
    
    def __call__(self, smiles_tokens, smiles_seqs):        
        scores = self.get_scores(smiles_tokens)
        return 'solubility', scores
        
class Permeability:
    
    def __init__(self, device):
        self.predictor = xgb.Booster(model_file='/scratch/PeptiVerse/src/permeability/best_model.json')
        self.emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device)
        self.emb_model.eval()

        self.device = device
    
    def get_scores(self, x):
        # pdb.set_trace()
        scores = -10 * np.ones(len(x))
        attention_mask = (x != 0).to(self.device)
        features = np.array(self.emb_model(input_ids=x, attention_mask=attention_mask).last_hidden_state.mean(dim=1).detach().cpu())
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores
    
    def __call__(self, smiles_tokens, smiles_seqs):        
        scores = self.get_scores(smiles_tokens)
        scores = [(10 + score) / 10 for score in scores]
        return 'permeability', scores


class Halflife:
    def __init__(self, device=None, apply_log1p=True):
        self.apply_log1p = apply_log1p
        self.device = device

        self.predictor = xgb.Booster(model_file="/scratch/pCoMol/peptidomimetics/ckpt/halflife.json")

        base = AutoModelForMaskedLM.from_pretrained("aaronfeller/PeptideCLM-23M-all")
        self.emb_model = base.roformer.to(self.device).eval()

        self.tokenizer = SMILES_SPE_Tokenizer(
            "/scratch/PeptiVerse/functions/tokenizer/new_vocab.txt",
            "/scratch/PeptiVerse/functions/tokenizer/new_splits.txt",
        )

    @torch.no_grad()
    def generate_embeddings(self, smiles_tokens):
        attention_mask = (smiles_tokens != 0).to(self.device)

        out = self.emb_model(input_ids=smiles_tokens, attention_mask=attention_mask)
        embs = out.last_hidden_state.mean(dim=1).detach().cpu().numpy().astype(np.float32)

        if len(embs) == 0:
            return np.zeros((0, 768), dtype=np.float32)
        return embs

    def predict_log1p(self, smiles_tokens):
        X = self.generate_embeddings(smiles_tokens)
        if X.shape[0] == 0:
            return np.array([], dtype=np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dmat = xgb.DMatrix(X)
        pred = self.predictor.predict(dmat).astype(np.float32)  # regression output
        return pred

    def predict_hours(self, smiles_tokens):
        pred = self.predict_log1p(smiles_tokens)
        if self.apply_log1p:
            return np.expm1(pred)  # convert log1p(hours) -> hours
        return pred

    def __call__(self, smiles_tokens, smiles_seqs):
        return 'halflife', self.predict_hours(smiles_tokens)


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
        checkpoint = torch.load('/scratch/classifiers/binding/best_model.pt', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.prot_tokenizer = alphabet.get_batch_converter() # load esm tokenizer

        data = [("target", prot_seq)]  
        # get tokenized protein
        _, _, prot_tokens = self.prot_tokenizer(data)
        with torch.no_grad():
            results = self.esm_model.forward(prot_tokens, repr_layers=[33]) 
            prot_emb = results["representations"][33]
            
        self.prot_emb = prot_emb[0]
        self.prot_emb = torch.mean(self.prot_emb, dim=0, keepdim=True).to(device)
        self.device = device
                
    def forward(self, x):        
        with torch.no_grad():
            attention_mask = (x != 0).to(self.device)
            scores = []
            pep_emb = self.pep_model(input_ids=x, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state.mean(dim=1, keepdim=True)
            for pep in pep_emb:
                score, logits = self.model.forward(self.prot_emb, pep)
                scores.append(min(10, score.item()) / 10)
        
        return scores
    
    def __call__(self, smiles_tokens, smiles_seqs):
        scores = self.forward(smiles_tokens)
        return 'affinity', scores


class SmallMolecule:
    def __init__(self, protein_sequence, device):
        # Admetica
        self.trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="cuda", devices=1)
        self.models = self.load_models(ckpt_dir='/scratch/miniconda3/envs/admetica/lib/python3.11/site-packages/admetica/Models')

        # DeepDTAGen
        model_path = f'/scratch/DeepDTAGen/models/deepdtagen_model_bindingdb.pth'
        tokenizer_path = f'/scratch/DeepDTAGen/data/bindingdb_tokenizer.pkl'

        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        self.deep_dta_gen = DeepDTAGen(tokenizer)
        self.deep_dta_gen.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        self.deep_dta_gen.to(device)
        self.deep_dta_gen.eval()

        self.protein_sequence = protein_sequence
        self.device = device

    def load_models(self, ckpt_dir):
        toxicity_model = chemprop.models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'ld50.ckpt'))
        solubility_model = chemprop.models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'solubility.ckpt'))
        permeability_model = chemprop.models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'caco2.ckpt'))
        halflife_model = chemprop.models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'half-life.ckpt'))

        return toxicity_model, solubility_model, permeability_model, halflife_model

    def is_valid_smiles(self, smiles):
        """Check if the given SMILES string is valid."""
        try:
            return Chem.MolFromSmiles(smiles) is not None
        except Exception as e:
            logging.error(f"Error validating SMILES '{smiles}': {str(e)}")
            return False

    def prediction(self, smiles_list, trainer, model):
        valid_smiles = [smi for smi in smiles_list if self.is_valid_smiles(smi)]
        valid_indices = [i for i, smi in enumerate(smiles_list) if self.is_valid_smiles(smi)]
        invalid_indices = [i for i in range(len(smiles_list)) if i not in valid_indices]

        if not valid_smiles:
            return np.full(len(smiles_list), "", dtype=object)

        test_data = [chemprop.data.MoleculeDatapoint.from_smi(smi) for smi in valid_smiles]
        featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
        test_dataset = chemprop.data.MoleculeDataset(test_data, featurizer=featurizer)
        test_loader = chemprop.data.build_dataloader(test_dataset, shuffle=False)

        with torch.no_grad():
            predictions = trainer.predict(model, test_loader)
        
        predictions = [pred.item() for batch in predictions for pred in batch]
        for index in invalid_indices:
            predictions.insert(index, "")
        
        return predictions

    def non_toxicity_from_log10mgkg(self, scores, lo=1.0, hi=4.0):
        """
        x : predicted log10(mg/kg)
        lo ~ 1  (≈10 mg/kg: very toxic)
        hi ~ 4  (≈10,000 mg/kg: low acute toxicity)
        returns ∈ [0,1]: higher = safer (non-toxic)
        """
        res = []
        for score in scores:
            score = max(lo, min(hi, score))
            res.append((score - lo) / (hi - lo))
        return res

    def __call__(self, smiles_tokens, smiles_seqs):
        # Admetica
        scores = []
        for model in self.models:
            scores.append(self.prediction(smiles_seqs, self.trainer, model))

        non_toxicity = self.non_toxicity_from_log10mgkg(scores[0])
        solubility = [(max(-12, min(2, score)) + 12) / 14 for score in scores[1]]
        permeability = [(max(-8, min(-3, score)) + 8) / 5 for score in scores[2]]
        # halflife = [max(0, min(2, np.log10(max(1e-6, score)))) / 2 for score in scores[3]]
        halflife = scores[3]
        ratio = [analyze_peptide_likeness(smiles)['peptide_likeness'] for smiles in smiles_seqs]
        
        # DeepDTAGen
        protein_sequences = [self.protein_sequence] * len(smiles_seqs)
        batch = process_latent_batch(smiles_seqs, protein_sequences).to(self.device)

        with torch.no_grad():
            affinity = self.deep_dta_gen(batch).squeeze(-1)
        affinity = [min(10, score.item()) / 10 for score in affinity]

        return 'small_molecule', {
            'non_toxicity': non_toxicity,
            'solubility': solubility,
            'permeability': permeability,
            'halflife': halflife,
            'affinity': affinity,
            'ratio': ratio,
        }
    

def parse_motifs(motif: str) -> list:
    parts = motif.split(',')
    result = []

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    # result = [pos-1 for pos in result]
    print(f'Target Motifs: {result}')
    return torch.tensor(result)
    

class BindEvaluator(pl.LightningModule):
    def __init__(self, cfg):
        super(BindEvaluator, self).__init__()

        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.chemberta_model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k").roberta.eval()
        for param in self.chemberta_model.parameters():
            param.requires_grad = False

        self.repeated_module = RepeatedModule(cfg.model.n_layers, cfg.model.d_model, cfg.model.d_hidden,
                                               cfg.model.n_head, cfg.model.d_k, cfg.model.d_v, cfg.model.d_inner, dropout=cfg.model.dropout)

        self.final_attention_layer = MultiHeadAttentionSequence(cfg.model.n_head, cfg.model.d_model,
                                                                cfg.model.d_k, cfg.model.d_v, dropout=cfg.model.dropout)

        self.final_ffn = FFN(cfg.model.d_model, cfg.model.d_inner, dropout=cfg.model.dropout)

        self.output_projection_prot = nn.Linear(cfg.model.d_model, 1)

    def forward(self, binder_tokens, target_tokens):
        peptide_sequence = self.chemberta_model(**binder_tokens).last_hidden_state
        protein_sequence = self.esm_model(**target_tokens).last_hidden_state

        binder_mask = binder_tokens["attention_mask"]      # [B, Ls]
        target_mask = target_tokens["attention_mask"]      # [B, Lp]
    
        prot_enc, sequence_enc, sequence_attention_list, prot_attention_list, \
            prot_seq_attention_list, seq_prot_attention_list = self.repeated_module(
                peptide_sequence,
                protein_sequence,
                peptide_mask=binder_mask,
                protein_mask=target_mask,
            )
    
        # final cross-attention: protein queries attend to binder keys
        prot_enc, final_prot_seq_attention = self.final_attention_layer(
            prot_enc, sequence_enc, sequence_enc,
            key_padding_mask=binder_mask,
            query_padding_mask=target_mask,
        )
    
        prot_enc = self.final_ffn(prot_enc, padding_mask=target_mask)
        prot_enc = self.output_projection_prot(prot_enc)
        return prot_enc
    
class MotifModel:
    def __init__(self, cfg, target, motifs, device, specificity):
        self.cfg = cfg
        self.threshold = 0.918
        self.device = device
        self.specificity = specificity
        self.chemberta_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k")
        self.esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.target = self.esm_tokenizer(target, return_tensors='pt').to(device)
        self.motifs = parse_motifs(motifs).to(device)
        self.bindevaluator = BindEvaluator.load_from_checkpoint(cfg.inference.ckpt, cfg=cfg, map_location=device)
    
    def __call__(self, smiles_tokens, smiles_seqs):
        binder = self.chemberta_tokenizer(smiles_seqs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        batch_size = binder['input_ids'].shape[0]
        L = self.target['input_ids'].shape[1]
        target_input_ids = self.target['input_ids'].expand(batch_size, L)
        target_attention_mask = self.target['attention_mask'].expand(batch_size, L)
        target = {"input_ids": target_input_ids, "attention_mask": target_attention_mask}

        # pdb.set_trace()
        prediction = self.bindevaluator(binder, target).squeeze(-1) 
        probs = torch.sigmoid(prediction)   # (B, L)

        motif_scores = probs[:, self.motifs].mean(dim=1)
        if self.specificity:
            non_motif_probs = probs[:, [i for i in range(probs.shape[1]) if i not in self.motifs]]
            mask = non_motif_probs >= self.threshold
            count = mask.sum(dim=-1)
            specificity = 1 - count / (L-2)

            return "motif", (motif_scores.tolist(), specificity.tolist())
        else:
            return "motif", motif_scores.tolist()
            
class DeepDTAGenModel:
    def __init__(self, protein_sequence, device):
        model_path = f'/scratch/DeepDTAGen/models/deepdtagen_model_bindingdb.pth'
        tokenizer_path = f'/scratch/DeepDTAGen/data/bindingdb_tokenizer.pkl'

        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        self.model = DeepDTAGen(tokenizer)
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False)).to(device)
        self.model.eval()

        self.protein_sequence = protein_sequence
        self.device = device
        
    def __call__(self, smiles_tokens, smiles_seqs):
        protein_sequences = [self.protein_sequence] * len(smiles_seqs)
        batch = process_latent_batch(smiles_seqs, protein_sequences).to(self.device)

        with torch.no_grad():
            scores = self.model(batch).squeeze(-1)

        return "affinity", scores
