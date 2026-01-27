import pandas as pd

# Load your CSV
df = pd.read_csv('/scratch/pranamlab/tong/pCoMol/gfp/samples/objective_ablation/length.csv')

# Open a new FASTA file to write
with open('/scratch/pranamlab/tong/pCoMol/gfp/samples/objective_ablation/length.fasta', 'w') as f:
    for index, row in df.iterrows():
        # Create a header (e.g., >Sequence_0, >Sequence_1, etc.)
        # If you have an 'ID' column, use row['ID'] instead of index
        f.write(f">Sequence_{index}\n")
        f.write(f"{row['Sequence']}\n")