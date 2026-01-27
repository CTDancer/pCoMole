import os
import glob
import tempfile
import subprocess
from typing import Iterable, List, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib

def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to joblib.load model: {path}\n"
            f"Original error: {repr(e)}\n\n"
            f"Tip: these pickles can be Python/scikit-learn-version sensitive; "
            f"use the same environment used to train/export the models."
        )

@staticmethod
def clean_seq(s: str) -> str:
    s = (s or "").strip().upper().replace(" ", "").replace("\n", "").replace("\r", "")
    if len(s) == 0:
        raise ValueError("Encountered an empty sequence.")
    return s

@staticmethod
def write_fasta(path: str, seqs: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(seqs):
            f.write(f">query_{i}\n")
            f.write(seq + "\n")

@staticmethod
def parse_fasta(path: str) -> pd.DataFrame:
    names, seqs = [], []
    cur_name, cur_seq = None, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_name is not None:
                    names.append(cur_name)
                    seqs.append("".join(cur_seq))
                cur_name = line[1:].strip()
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_name is not None:
        names.append(cur_name)
        seqs.append("".join(cur_seq))

    return pd.DataFrame({"name": names, "seq_align": seqs})


class GFPExcitationPred:
    """
    Predict GFP excitation maximum (ex_model) using the FPredX pipeline:
      1) MAFFT add query fasta to reference alignment (FPredX_mafft.fasta)
      2) One-hot encoding on the *combined* alignment
      3) Keep only features in available_res.csv
      4) Slice last N rows (the added sequences)
      5) Predict with every model in em_model/ and take the mean
    """

    def __init__(
        self,
        root_dir: str = ".",
        laser: float = 488,
        mafft_path: str = "/scratch/bin/mafft",
        ref_alignment_fasta: str = "FPredX_mafft.fasta",
        available_res_csv: str = "available_res.csv",
        model_dir: str = "ex_model",
        preload_models: bool = True,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.mafft_path = mafft_path
        self.ref_alignment_fasta = os.path.join(self.root_dir, ref_alignment_fasta)
        self.available_res_csv = os.path.join(self.root_dir, available_res_csv)
        self.model_dir = os.path.join(self.root_dir, model_dir)

        self.laser = laser

        if not os.path.isfile(self.mafft_path):
            raise FileNotFoundError(f"MAFFT not found at: {self.mafft_path}")

        if not os.path.isfile(self.ref_alignment_fasta):
            raise FileNotFoundError(f"Reference alignment fasta not found: {self.ref_alignment_fasta}")

        if not os.path.isfile(self.available_res_csv):
            raise FileNotFoundError(f"available_res.csv not found: {self.available_res_csv}")

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.available_res = pd.read_csv(self.available_res_csv, index_col=0)

        self.model_paths = sorted(glob.glob(os.path.join(self.model_dir, "*")))
        if len(self.model_paths) == 0:
            raise FileNotFoundError(f"No model files found in: {self.model_dir}")

        self._models = None
        if preload_models:
            self._models = [load_model(p) for p in self.model_paths]

    def __call__(self, seqs: Union[str, Iterable[str]]) -> Union[float, np.ndarray]:
        """
        If seqs is a single sequence string -> returns float (predicted emission max).
        If seqs is an iterable of sequences -> returns np.ndarray of floats.
        """
        return self.get_score(seqs)

    def get_score(
        self,
        seqs: Union[str, Iterable[str]],
        return_debug: bool = False,
    ) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Predict emission maximum.

        Args:
          seqs: str or iterable[str] of raw (unaligned) amino-acid sequences
          return_debug: if True, returns dict with extra fields (unavailable_list, per_model_preds, etc.)

        Returns:
          float if input is a single sequence, else np.ndarray
          or dict if return_debug=True
        """
        single = isinstance(seqs, str)
        seq_list = [seqs] if single else list(seqs)

        # Basic cleanup
        seq_list = [clean_seq(s) for s in seq_list]
        n = len(seq_list)
        if n == 0:
            raise ValueError("No sequences provided.")

        # Build features via the same MAFFT+onehot pipeline as the script
        X, unavailable_list = self._featurize_with_mafft(seq_list)

        # Predict with all em_model models and average
        per_model = self._predict_all_models(X)  # shape (n_models, n)
        mean_pred = per_model.mean(axis=0)       # shape (n,)

        if return_debug:
            return {
                "mean_pred": mean_pred if not single else float(mean_pred[0]),
                "per_model_pred": per_model,
                "model_paths": self.model_paths,
                "unavailable_list": unavailable_list,
                "n_sequences": n,
                "n_features": X.shape[1],
            }

        return -1 * abs(mean_pred - self.laser)

    def _run_mafft_add(self, query_fasta: str, out_fasta: str) -> None:
        # Equivalent to:
        # mafft --add <query_fasta> --keeplength FPredX_mafft.fasta > out_fasta
        cmd = [
            self.mafft_path,
            "--add",
            query_fasta,
            "--keeplength",
            self.ref_alignment_fasta,
        ]
        with open(out_fasta, "w", encoding="utf-8") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                "MAFFT failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr}"
            )

    def _featurize_with_mafft(self, seqs: List[str]):
        with tempfile.TemporaryDirectory() as td:
            q_fa = os.path.join(td, "query.fasta")
            out_fa = os.path.join(td, "FPredX_mafft_predict.fasta")

            write_fasta(q_fa, seqs)
            self._run_mafft_add(q_fa, out_fa)

            # Read the combined alignment (ref + queries)
            seq_list_df = parse_fasta(out_fa)

            # One-hot on the whole combined alignment (same as script)
            bypos = seq_list_df["seq_align"].apply(lambda x: pd.Series(list(x)))
            one_hot = pd.get_dummies(bypos)

            # Keep only features in available_res.csv (same logic as script)
            one_hot_trim = pd.DataFrame()
            unavailable_list = []
            avail_idx = set(self.available_res.index)

            for col in one_hot.columns:
                if col in avail_idx:
                    one_hot_trim = pd.concat([one_hot_trim, one_hot[col]], axis=1)
                else:
                    unavailable_list.append(col)

            # Take last N rows corresponding to the added sequences (same as script)
            n = len(seqs)
            one_hot_trim = one_hot_trim.iloc[-n:, :].reset_index(drop=True)

            # Model expects numeric matrix
            X = np.asarray(one_hot_trim, dtype=float)
            return X, unavailable_list

    def _predict_all_models(self, X: np.ndarray) -> np.ndarray:
        models = self._models
        if models is None:
            models = [self._load_model(p) for p in self.model_paths]

        preds = []
        for m in models:
            y = m.predict(X)
            y = np.asarray(y).reshape(-1)
            preds.append(y)

        return np.stack(preds, axis=0)


class GFPBrightPred:
    """
    Predict GFP brightness (bright_model) using the FPredX pipeline:
      1) MAFFT add query fasta to reference alignment (FPredX_mafft.fasta)
      2) One-hot encoding on the *combined* alignment
      3) Keep only features in available_res.csv
      4) Slice last N rows (the added sequences)
      5) Predict with every model in em_model/ and take the mean
    """

    def __init__(
        self,
        root_dir: str = ".",
        mafft_path: str = "/scratch/bin/mafft",
        ref_alignment_fasta: str = "FPredX_mafft.fasta",
        available_res_csv: str = "available_res.csv",
        model_dir: str = "bright_model",
        preload_models: bool = True,
    ):
        self.root_dir = os.path.abspath(root_dir)
        self.mafft_path = mafft_path
        self.ref_alignment_fasta = os.path.join(self.root_dir, ref_alignment_fasta)
        self.available_res_csv = os.path.join(self.root_dir, available_res_csv)
        self.model_dir = os.path.join(self.root_dir, model_dir)

        if not os.path.isfile(self.mafft_path):
            raise FileNotFoundError(f"MAFFT not found at: {self.mafft_path}")

        if not os.path.isfile(self.ref_alignment_fasta):
            raise FileNotFoundError(f"Reference alignment fasta not found: {self.ref_alignment_fasta}")

        if not os.path.isfile(self.available_res_csv):
            raise FileNotFoundError(f"available_res.csv not found: {self.available_res_csv}")

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        self.available_res = pd.read_csv(self.available_res_csv, index_col=0)

        self.model_paths = sorted(glob.glob(os.path.join(self.model_dir, "*")))
        if len(self.model_paths) == 0:
            raise FileNotFoundError(f"No model files found in: {self.model_dir}")

        self._models = None
        if preload_models:
            self._models = [load_model(p) for p in self.model_paths]

    def __call__(self, seqs: Union[str, Iterable[str]]) -> Union[float, np.ndarray]:
        """
        If seqs is a single sequence string -> returns float (predicted emission max).
        If seqs is an iterable of sequences -> returns np.ndarray of floats.
        """
        return self.get_score(seqs)

    def get_score(
        self,
        seqs: Union[str, Iterable[str]],
        return_debug: bool = False,
    ) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Predict emission maximum.

        Args:
          seqs: str or iterable[str] of raw (unaligned) amino-acid sequences
          return_debug: if True, returns dict with extra fields (unavailable_list, per_model_preds, etc.)

        Returns:
          float if input is a single sequence, else np.ndarray
          or dict if return_debug=True
        """
        single = isinstance(seqs, str)
        seq_list = [seqs] if single else list(seqs)

        # Basic cleanup
        seq_list = [clean_seq(s) for s in seq_list]
        n = len(seq_list)
        if n == 0:
            raise ValueError("No sequences provided.")

        # Build features via the same MAFFT+onehot pipeline as the script
        X, unavailable_list = self._featurize_with_mafft(seq_list)

        # Predict with all em_model models and average
        per_model = self._predict_all_models(X)  # shape (n_models, n)
        mean_pred = per_model.mean(axis=0)       # shape (n,)

        if return_debug:
            return {
                "mean_pred": mean_pred if not single else float(mean_pred[0]),
                "per_model_pred": per_model,
                "model_paths": self.model_paths,
                "unavailable_list": unavailable_list,
                "n_sequences": n,
                "n_features": X.shape[1],
            }

        return mean_pred

    def _run_mafft_add(self, query_fasta: str, out_fasta: str) -> None:
        # Equivalent to:
        # mafft --add <query_fasta> --keeplength FPredX_mafft.fasta > out_fasta
        cmd = [
            self.mafft_path,
            "--add",
            query_fasta,
            "--keeplength",
            self.ref_alignment_fasta,
        ]
        with open(out_fasta, "w", encoding="utf-8") as out:
            proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                "MAFFT failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr}"
            )

    def _featurize_with_mafft(self, seqs: List[str]):
        with tempfile.TemporaryDirectory() as td:
            q_fa = os.path.join(td, "query.fasta")
            out_fa = os.path.join(td, "FPredX_mafft_predict.fasta")

            write_fasta(q_fa, seqs)
            self._run_mafft_add(q_fa, out_fa)

            # Read the combined alignment (ref + queries)
            seq_list_df = parse_fasta(out_fa)

            # One-hot on the whole combined alignment (same as script)
            bypos = seq_list_df["seq_align"].apply(lambda x: pd.Series(list(x)))
            one_hot = pd.get_dummies(bypos)

            # Keep only features in available_res.csv (same logic as script)
            one_hot_trim = pd.DataFrame()
            unavailable_list = []
            avail_idx = set(self.available_res.index)

            for col in one_hot.columns:
                if col in avail_idx:
                    one_hot_trim = pd.concat([one_hot_trim, one_hot[col]], axis=1)
                else:
                    unavailable_list.append(col)

            # Take last N rows corresponding to the added sequences (same as script)
            n = len(seqs)
            one_hot_trim = one_hot_trim.iloc[-n:, :].reset_index(drop=True)

            # Model expects numeric matrix
            X = np.asarray(one_hot_trim, dtype=float)
            return X, unavailable_list

    def _predict_all_models(self, X: np.ndarray) -> np.ndarray:
        models = self._models
        if models is None:
            models = [self._load_model(p) for p in self.model_paths]

        preds = []
        for m in models:
            y = m.predict(X)
            y = np.asarray(y).reshape(-1)
            preds.append(y)

        return np.stack(preds, axis=0)


class GFPLength:
    def __init__(self, orig_seq):
        self.orig_seq = orig_seq
        print("Initial Length: ", len(self.orig_seq))

    def __call__(self, seqs):
        return [len(self.orig_seq) - len(seq) for seq in seqs]
