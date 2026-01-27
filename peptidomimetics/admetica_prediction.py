import torch
import numpy as np
import logging
from rdkit import Chem
from admetica.constants import mean_vectors
from chemprop import data, featurizers, models
from lightning import pytorch as pl
import pdb

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


def load_models(ckpt_dir):
    toxicity_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'ld50.ckpt'))
    solubility_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'solubility.ckpt'))
    permeability_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'caco2.ckpt'))
    halflife_model = models.MPNN.load_from_checkpoint(os.path.join(ckpt_dir, 'half-life.ckpt'))

    return toxicity_model, solubility_model, permeability_model, halflife_model

def is_valid_smiles(smiles):
    """Check if the given SMILES string is valid."""
    try:
        return Chem.MolFromSmiles(smiles) is not None
    except Exception as e:
        logging.error(f"Error validating SMILES '{smiles}': {str(e)}")
        return False

def prediction(smiles_list, trainer, model):
    valid_smiles = [smi for smi in smiles_list if is_valid_smiles(smi)]
    valid_indices = [i for i, smi in enumerate(smiles_list) if is_valid_smiles(smi)]
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
    
    return torch.tensor(predictions)

def main():
    smiles_list = ['CSCC[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)CNC(=O)CNC(=O)[C@@H](N)Cc1ccc(O)cc1)C(=O)N[C@@H](CCC(N)=O)C(N)=O']
    trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="cuda", devices=1)
    models = load_models(ckpt_dir='/scratch/miniconda3/envs/admetica/lib/python3.11/site-packages/admetica/Models')
    for model in models:
        res = prediction(smiles_list, trainer, model)
        print(res)

if __name__ == '__main__':
    main()
