import warnings
import logging

# --- Silence Python Warnings ---
# 1. Silence the RDKit RuntimeWarning
# This is triggered by the C++ to Python conversion layer
warnings.filterwarnings("ignore", 
                        message="to-Python converter.*already registered", 
                        category=RuntimeWarning)

# 2. Silence the Hyperopt/pkg_resources UserWarning
# This is triggered by hyperopt importing an old package
warnings.filterwarnings("ignore", 
                        message="pkg_resources is deprecated", 
                        category=UserWarning)

# 3. Silence any other FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Silence RDKit's Internal Logger (Your #1) ---
# This is still a good idea, but it's separate from the 'warnings' system.
# Do this *after* importing RDKit.
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# --- Silence Library Loggers (Your #3) ---
# This controls 'print-like' output, not 'warning' banners.
# This is also fine to keep.
logging.basicConfig(level=logging.ERROR)
logging.getLogger("chemprop").setLevel(logging.ERROR)
logging.getLogger("hyperopt").setLevel(logging.ERROR)

from admet_ai import ADMETModel
import pdb

model = ADMETModel()

smiles_list = [
    'CC(C)CC(NC(=O)[C@H1](CC(N)=O)NC(=O)[C@H1](CO)NC(=O)[C@@H1](N)CO)([C@@H1])CCCNC(=N)NC=O',
]

for smiles in smiles_list:
    preds = model.predict(smiles=smiles)
    # print(f"Solubility: {preds['Solubility_AqSolDB']}")
    pdb.set_trace()
    print(preds.keys())