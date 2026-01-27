import pandas as pd
from constraints import GFP
from pathlib import Path
import torch
import numpy as np

def read_fasta_sequences(fasta_path: str):
    fasta_path = Path(fasta_path)
    seq_ids, seqs = [], []

    cur_id = None
    cur_chunks = []

    with fasta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if cur_id is not None:
                    seq_ids.append(cur_id)
                    seqs.append("".join(cur_chunks).replace(" ", "").upper())

                header = line[1:].strip()
                cur_id = header.split()[0]
                cur_chunks = []
            else:
                cur_chunks.append(line)

    if cur_id is not None:
        seq_ids.append(cur_id)
        seqs.append("".join(cur_chunks).replace(" ", "").upper())

    return seq_ids, seqs


df = pd.read_csv('/scratch/pranamlab/tong/pCoMol/gfp/FPredX/summary_pred_mean_scisor_1.csv')
_, seqs = read_fasta_sequences('/scratch/pranamlab/tong/pCoMol/gfp/scisor_gfp_1.fasta')
# seqs = seqs[:100]

assert len(df) == len(seqs), f"Length mismatch: len(df)={len(df)} vs len(seqs)={len(seqs)}"

df['ex'] = -(df['ex_model'] - 488).abs()
df['em'] = df['em_model'].between(470, 550).astype(int)

gfp = GFP('cuda:0')

# --- compute validity in batches ---
def predict_validity(model, seqs, batch_size=334):
    preds = []
    model_out = None
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i+batch_size]
            model_out = model(batch)  # expects list[str]
            # normalize output -> 1D numpy int array
            if isinstance(model_out, torch.Tensor):
                out = model_out.detach().to("cpu").view(-1).numpy()
            else:
                out = np.asarray(model_out).reshape(-1)
            preds.append((out == 1).astype(np.int64))
    return np.concatenate(preds, axis=0)

df['validity'] = predict_validity(gfp, seqs, batch_size=256)

# Score: (ex + bright_model) * validity
df['Score'] = (df['ex'] + df['bright_model']) * df['validity']

# Print Mean and Standard Deviation
print(f"Excitation Offset:       {df['ex'].mean():.4f} ± {df['ex'].std():.4f}")
print(f"Emission Range Match %:  {df['em'].mean():.4f} ± {df['em'].std():.4f}")
print(f"Brightness:              {df['bright_model'].mean():.4f} ± {df['bright_model'].std():.4f}")
print(f"Validity rate:           {df['validity'].mean():.4f} ± {df['validity'].std():.4f}")
print(f"Score:                   {df['Score'].mean():.4f} ± {df['Score'].std():.4f}")

# (Optional) attach sequence for easier inspection
# df['seq'] = seqs

# # --- print best row within first k rows ---
ks = [1, 10, 50, 100, 200, 342]
cols_to_show = ['Score', 'validity', 'ex', 'em', 'bright_model', 'ex_model', 'em_model']

for k in ks:
    k = min(k, len(df))
    sub = df.head(k)
    best_idx = sub['Score'].idxmax()
    best = df.loc[best_idx, cols_to_show]
    print(f"\n=== Best within first {k} rows (index={best_idx}) ===")
    print(best.to_string())
    # If you enabled df['seq']:
    # print("seq:", df.loc[best_idx, 'seq'])
