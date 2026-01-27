import os
import pandas as pd
import subprocess

CSV_PATH = "/scratch/pranamlab/tong/pCoMol/peptidomimetics/samples/7JVS.csv"
OUT_DIR = "/scratch/pranamlab/tong/pCoMol/peptidomimetics/boltz_files/7JVS" 

PROTEIN_SEQ = (
    "MITVDITVNDEGKVTDVIMDGHADHGEYGHDIVSAGASAVLFGSVNAIIGLTSERPDINYDDQGGHFHIRSVDTNNDEAQLILQTMLVSLQTIEEEYNENIRLNYK"
)

BOLTZ_CACHE = "/scratch/pranamlab/tong/.cache/boltz/"
SEED = "42"

def yaml_single_quote(s: str) -> str:
    """YAML-safe single-quoted string."""
    return "'" + s.replace("'", "''") + "'"

def write_yaml(smiles: str, out_path: str) -> None:
    yaml_text = f"""sequences:
  - ligand:
      id: A
      smiles: {yaml_single_quote(smiles)}

  - protein:
      id: B
      sequence: {yaml_single_quote(PROTEIN_SEQ)}

properties:
  - affinity:
      binder: A
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)

def run_boltz_predict():
    cmd = [
        "boltz", "predict", OUT_DIR,
        "--use_msa_server",
        "--seed", SEED,
        "--cache", BOLTZ_CACHE,
        # "--override",
    ]
    # check=True will raise if boltz exits non-zero
    subprocess.run(cmd, check=True)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    if "SMILES" not in df.columns:
        raise ValueError(f"CSV must contain a column named 'SMILES'. Found: {list(df.columns)}")

    for i, raw in enumerate(df["SMILES"].tolist(), start=1):
        if pd.isna(raw):
            continue
        smiles = str(raw).strip()
        if not smiles:
            continue

        yaml_path = os.path.join(OUT_DIR, f"{i}.yaml")
        write_yaml(smiles, yaml_path)

        print(f"[{i}] Wrote {yaml_path} | running boltz predict ...")
        run_boltz_predict()

    print("Done.")

if __name__ == "__main__":
    main()