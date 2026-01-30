# cope_domain_detector.py
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class HitCriteria:
    ievalue_max: float
    min_dom_len: int


@dataclass(frozen=True)
class DomainDetectorConfig:
    hmm_path: str
    cpu: int = 8

    # Stage1 acceptance
    stage1: HitCriteria = HitCriteria(ievalue_max=1e-2, min_dom_len=35)

    # Stage2 acceptance (recall-first)
    stage2: HitCriteria = HitCriteria(ievalue_max=10.0, min_dom_len=20)

    # Stage2 hmmscan reporting knobs (big so rescue sees weak hits)
    stage2_report_E: float = 1e6
    stage2_report_domE: float = 1e6
    stage2_incE: float = 1e6
    stage2_incdomE: float = 1e6

    # Optional Stage1 sensitivity knobs (usually leave None)
    stage1_report_E: Optional[float] = None
    stage1_report_domE: Optional[float] = None
    stage1_F1: Optional[float] = None
    stage1_F2: Optional[float] = None
    stage1_F3: Optional[float] = None

    # Model name prefixes in your HMM library
    hnh_prefix: str = "HNH_"
    ruvc_prefix: str = "RuvC_"


class DomainDetector:
    """
    Cas9 nuclease domain presence detector (HNH + RuvC), Option-B (recall-first).

    Uses hmmscan:
      - Stage1: fast scan over batch
      - Stage2: --max scan only on Stage1 misses (rescue)
    """

    def __init__(self, config: DomainDetectorConfig):
        self.cfg = config
        self._hmmscan = self._check_exe("hmmscan")
        self._validate_hmm(self.cfg.hmm_path)

    @staticmethod
    def _check_exe(name: str) -> str:
        exe = shutil.which(name)
        if not exe:
            raise RuntimeError(f"Required executable not found on PATH: {name}")
        return exe

    @staticmethod
    def _validate_hmm(hmm_path: str) -> None:
        if not os.path.exists(hmm_path):
            raise FileNotFoundError(f"HMM file not found: {hmm_path}")
        if os.path.getsize(hmm_path) == 0:
            raise ValueError(f"HMM file is empty: {hmm_path}")

    @staticmethod
    def _write_fasta_from_pairs(pairs: Sequence[Tuple[str, str]], path: str) -> None:
        with open(path, "w") as out:
            for sid, seq in pairs:
                out.write(f">{sid}\n")
                seq = seq.strip().replace(" ", "").replace("\n", "")
                for i in range(0, len(seq), 80):
                    out.write(seq[i:i+80] + "\n")

    def _run_hmmscan(
        self,
        fasta_path: str,
        domtblout_path: str,
        *,
        max_mode: bool,
        report_E: Optional[float] = None,
        report_domE: Optional[float] = None,
        incE: Optional[float] = None,
        incdomE: Optional[float] = None,
        F1: Optional[float] = None,
        F2: Optional[float] = None,
        F3: Optional[float] = None,
    ) -> None:
        cmd: List[str] = [
            self._hmmscan,
            "--cpu", str(self.cfg.cpu),
            "--domtblout", domtblout_path,
        ]
        if max_mode:
            cmd.append("--max")

        if F1 is not None:
            cmd += ["--F1", str(F1)]
        if F2 is not None:
            cmd += ["--F2", str(F2)]
        if F3 is not None:
            cmd += ["--F3", str(F3)]

        if report_E is not None:
            cmd += ["-E", str(report_E)]
        if report_domE is not None:
            cmd += ["--domE", str(report_domE)]
        if incE is not None:
            cmd += ["--incE", str(incE)]
        if incdomE is not None:
            cmd += ["--incdomE", str(incdomE)]

        cmd += [self.cfg.hmm_path, fasta_path]

        # discard stdout; errors go to CalledProcessError
        with open(os.devnull, "w") as devnull:
            subprocess.run(cmd, check=True, stdout=devnull, stderr=subprocess.DEVNULL)

    def _parse_domtblout_presence(self, domtblout_path: str, crit: HitCriteria) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """
        Returns (has_hnh, has_ruvc) dicts keyed by query ID.
        domtblout columns (1-based):
          1=model, 4=query, 13=i-Evalue, 20=env_from, 21=env_to
        """
        has_hnh: Dict[str, bool] = {}
        has_ruvc: Dict[str, bool] = {}

        with open(domtblout_path, "r") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 21:
                    continue
                model = parts[0]
                qid = parts[3]
                try:
                    ie = float(parts[12])
                    env_from = int(parts[19])
                    env_to = int(parts[20])
                except ValueError:
                    continue
                dom_len = env_to - env_from + 1
                if ie > crit.ievalue_max or dom_len < crit.min_dom_len:
                    continue

                if model.startswith(self.cfg.hnh_prefix):
                    has_hnh[qid] = True
                elif model.startswith(self.cfg.ruvc_prefix):
                    has_ruvc[qid] = True

        return has_hnh, has_ruvc

    def predict_batch(
        self,
        seqs: Sequence[str],
        ids: Optional[Sequence[str]] = None,
    ) -> List[bool]:
        """
        Classify sequences as pass/fail (domains present: HNH AND RuvC).
        - seqs: list of amino acid sequences
        - ids: optional list of IDs (same length). If omitted, uses seq0..seqN.

        Returns: List[bool] aligned with input order.
        """
        if ids is None:
            ids = [f"seq{i}" for i in range(len(seqs))]
        if len(ids) != len(seqs):
            raise ValueError("ids and seqs must have same length")

        pairs = list(zip(ids, seqs))

        with tempfile.TemporaryDirectory() as td:
            fa1 = os.path.join(td, "stage1.faa")
            dom1 = os.path.join(td, "stage1.domtblout")

            self._write_fasta_from_pairs(pairs, fa1)
            self._run_hmmscan(
                fasta_path=fa1,
                domtblout_path=dom1,
                max_mode=False,
                report_E=self.cfg.stage1_report_E,
                report_domE=self.cfg.stage1_report_domE,
                F1=self.cfg.stage1_F1,
                F2=self.cfg.stage1_F2,
                F3=self.cfg.stage1_F3,
            )

            h1, r1 = self._parse_domtblout_presence(dom1, self.cfg.stage1)

            stage1_pass = {sid for sid in ids if h1.get(sid, False) and r1.get(sid, False)}
            miss = [sid for sid in ids if sid not in stage1_pass]

            stage2_pass: set[str] = set()
            if miss:
                miss_pairs = [(sid, seqs[ids.index(sid)]) for sid in miss]  # stable small sizes
                fa2 = os.path.join(td, "stage2.faa")
                dom2 = os.path.join(td, "stage2.domtblout")

                self._write_fasta_from_pairs(miss_pairs, fa2)
                self._run_hmmscan(
                    fasta_path=fa2,
                    domtblout_path=dom2,
                    max_mode=True,
                    report_E=self.cfg.stage2_report_E,
                    report_domE=self.cfg.stage2_report_domE,
                    incE=self.cfg.stage2_incE,
                    incdomE=self.cfg.stage2_incdomE,
                )
                h2, r2 = self._parse_domtblout_presence(dom2, self.cfg.stage2)
                stage2_pass = {sid for sid in miss if h2.get(sid, False) and r2.get(sid, False)}

            final_pass = stage1_pass | stage2_pass
            return [sid in final_pass for sid in ids]

    def predict_one(self, seq: str, id_: str = "seq0") -> bool:
        return self.predict_batch([seq], ids=[id_])[0]
