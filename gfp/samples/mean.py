# import pandas as pd
# import pdb

# # Load the dataset
# file_path = '/scratch/pranamlab/tong/pCoMol/gfp/samples/objective_ablation/length_excitation.csv'  # Replace with your actual file name
# df = pd.read_csv(file_path, index_col=False)
# # df = df.sort_values(by='Affinity', ascending=False)
# # df.to_csv(file_path, index=False)
# print(len(df))

# # List of columns to average
# target_columns = [
#     'Length', 
#     'excitation',
#     # 'brightness'
# ]

# averages = df[target_columns].mean()

# print("--- Average Values ---")
# print(averages.round(4))

import pandas as pd

# Define the file path
file_path = '/scratch/pranamlab/tong/pCoMol/gfp/samples/specific_length/length_excitation_brightness.csv'

# Load the dataset
df = pd.read_csv(file_path)[:5]

# Compute Mean and Standard Deviation for the specific columns
stats = df[['excitation', 'brightness']].agg(['mean', 'std'])

print("Statistics for Excitation and Brightness:")
print(stats)

# Alternatively, to access individual values:
mean_exc = stats.loc['mean', 'excitation']
std_exc = stats.loc['std', 'excitation']

print(f"\nMean Excitation: {mean_exc:.4f}")
print(f"Std Dev Excitation: {std_exc:.4f}")