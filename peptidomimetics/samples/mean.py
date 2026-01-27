import pandas as pd
import pdb
# Load the dataset
file_path = '/scratch/pranamlab/tong/pCoMol/peptidomimetics/samples/known_binders_4_2_4_0.2_4_2_1/1AYC.csv'  # Replace with your actual file name
df = pd.read_csv(file_path, index_col=False)
# df = df.sort_values(by='Affinity', ascending=False)
# df.to_csv(file_path, index=False)
print(len(df))

# List of columns to average
target_columns = [
    'Length', 'Toxicity', 'Solubility', 'Permeability', 
    'Halflife', 'Affinity', 'Motif', 'Specificity', 
    'Peptidomimetic', 'Shorter'
]

averages = df[target_columns].mean()

print("--- Average Values ---")
print(averages.round(4))