# cas9_pam_hmm.py
from __future__ import annotations

import os
import re
import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch


# ----------------------------
# Data structures
# ----------------------------
@dataclass(frozen=True)
class HMMHit:
    """
    Represents one domain hit line from HMMER domtblout.
    Coordinates are 1-based (HMMER convention).
    """
    model: str
    accession: str
    query_id: str
    query_len: int

    # Per-domain statistics
    i_evalue: float
    bitscore: float

    # Alignment coordinates
    hmm_from: int
    hmm_to: int
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int

    # Optional
    hmm_len: Optional[int] = None
    bias: Optional[float] = None


# ----------------------------
# domtblout parsing
# ----------------------------
class DomTbloutParser:
    """
    Robust parser for HMMER3 --domtblout files.
    Assumes standard HMMER3 domtblout layout (space-delimited; description is trailing).
    """

    @staticmethod
    def parse(domtbl_path: str) -> List[HMMHit]:
        hits: List[HMMHit] = []
        with open(domtbl_path, "r") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.strip().split()

                # Standard domtblout columns (HMMER3):
                # 1 target name        cols[0]
                # 2 target accession   cols[1]
                # 3 tlen               cols[2]
                # 4 query name         cols[3]
                # 5 query accession    cols[4]
                # 6 qlen               cols[5]
                # 7 E-value (full)     cols[6]
                # 8 score (full)       cols[7]
                # 9 bias (full)        cols[8]
                # 10 #                 cols[9]
                # 11 of                cols[10]
                # 12 c-Evalue          cols[11]
                # 13 i-Evalue          cols[12]
                # 14 score (domain)    cols[13]
                # 15 bias (domain)     cols[14]
                # 16 hmm_from          cols[15]
                # 17 hmm_to            cols[16]
                # 18 ali_from          cols[17]
                # 19 ali_to            cols[18]
                # 20 env_from          cols[19]
                # 21 env_to            cols[20]
                # 22 acc               cols[21]
                # 23+ description      cols[22:]

                try:
                    model = cols[0]
                    accession = cols[1]
                    hmm_len = int(cols[2])
                    query_id = cols[3]
                    query_len = int(cols[5])

                    i_evalue = float(cols[12])
                    bitscore = float(cols[13])
                    bias = float(cols[14])

                    hmm_from = int(cols[15])
                    hmm_to = int(cols[16])
                    ali_from = int(cols[17])
                    ali_to = int(cols[18])
                    env_from = int(cols[19])
                    env_to = int(cols[20])
                except (IndexError, ValueError) as e:
                    # If your HMMER output is slightly different, raise with context.
                    raise RuntimeError(f"Failed parsing domtblout line:\n{line}\nError: {e}") from e

                hits.append(
                    HMMHit(
                        model=model,
                        accession=accession,
                        query_id=query_id,
                        query_len=query_len,
                        i_evalue=i_evalue,
                        bitscore=bitscore,
                        hmm_from=hmm_from,
                        hmm_to=hmm_to,
                        ali_from=ali_from,
                        ali_to=ali_to,
                        env_from=env_from,
                        env_to=env_to,
                        hmm_len=hmm_len,
                        bias=bias,
                    )
                )
        return hits


# ----------------------------
# hmmscan runner
# ----------------------------
class HMMSCAN:
    """
    Thin wrapper around hmmscan for 1 or many sequences.
    Uses --domtblout for machine-parseable results.
    """

    def __init__(
        self,
        hmm_db_path: str,
        hmmscan_bin: str = "hmmscan",
        cpu: int = 1,
        extra_args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        self.hmm_db_path = hmm_db_path
        self.hmmscan_bin = hmmscan_bin
        self.cpu = int(cpu)
        self.extra_args = extra_args or []
        self.env = env

        if not os.path.exists(self.hmm_db_path):
            raise FileNotFoundError(f"HMM DB not found: {self.hmm_db_path}")

    def scan_fasta(self, fasta_path: str, domtblout_path: str) -> None:
        cmd = [
            self.hmmscan_bin,
            "--cpu",
            str(self.cpu),
            "--domtblout",
            domtblout_path,
            self.hmm_db_path,
            fasta_path,
        ] + self.extra_args

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else e.stderr
            stdout_msg = e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else e.stdout
            raise RuntimeError(
                f"hmmscan failed with exit code {e.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{stdout_msg}\n"
                f"STDERR:\n{error_msg}\n"
                f"HMM DB path: {self.hmm_db_path}\n"
                f"FASTA path: {fasta_path}\n"
                f"FASTA exists: {os.path.exists(fasta_path)}"
            ) from e

    def scan_sequences(
        self,
        seqs: Union[str, List[str]],
        ids: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[str, str]:
        """
        Writes temporary FASTA and runs hmmscan.
        Returns: (domtblout_path, fasta_path) inside a tempdir.
        Caller is responsible for tempdir cleanup if they want persistence.
        """
        if isinstance(seqs, str):
            seq_list = [seqs]
        else:
            seq_list = list(seqs)

        if ids is None:
            id_list = [f"q{i}" for i in range(len(seq_list))]
        elif isinstance(ids, str):
            id_list = [ids]
        else:
            id_list = list(ids)

        if len(id_list) != len(seq_list):
            raise ValueError("ids must have same length as seqs")

        td = tempfile.TemporaryDirectory()
        fasta_path = os.path.join(td.name, "queries.faa")
        domtbl_path = os.path.join(td.name, "out.domtbl")

        with open(fasta_path, "w") as f:
            for qid, seq in zip(id_list, seq_list):
                seq = seq.replace(" ", "").strip()
                f.write(f">{qid}\n{seq}\n")

        self.scan_fasta(fasta_path, domtbl_path)

        # Return temp paths; also return tempdir object by attaching it for lifecycle
        # (so it doesn't get GC'ed early).
        setattr(self, "_last_tempdir", td)
        return domtbl_path, fasta_path


# ----------------------------
# PI-domain masking policy ("Option 1")
# ----------------------------
class Cas9PIMasker:
    """
    Builds a compact 'no deletion' mask for the PAM-interacting (PI) region.

    Option 1 implemented:
      - choose highest-confidence PI-family hit
      - then mask a capped window around the alignment core (ali_from..ali_to)
        using max_mask_len

    Notes:
      - HMMER coords are 1-based AA positions.
      - Assumes your token tensor x is [BOS] + AAs + [EOS] (+ pad),
        so AA position p corresponds to token index p.
    """

    DEFAULT_MODEL_PREFERENCE = ["Cas9_PI", "Cas9_PI_C", "Cas9_PI2", "CjCas9_PI_CTD"]

    def __init__(
        self,
        hmmscan: HMMSCAN,
        *,
        model_preference: Optional[List[str]] = None,
        use_env_coords: bool = False,      # False = ali coords (smaller)
        max_mask_len: int = 200,           # cap masked window length
        evalue_cutoff: float = 1e-5,       # per-domain i-evalue cutoff
        min_ali_len: int = 30,             # ignore tiny alignments
        fallback_last_n: Optional[int] = 250,  # if no hit, optionally mask last N AAs
        cache_size: int = 2048,
    ):
        self.hmmscan = hmmscan
        self.model_preference = model_preference or self.DEFAULT_MODEL_PREFERENCE
        self.model_rank = {m: i for i, m in enumerate(self.model_preference)}
        self.use_env_coords = use_env_coords
        self.max_mask_len = int(max_mask_len)
        self.evalue_cutoff = float(evalue_cutoff)
        self.min_ali_len = int(min_ali_len)
        self.fallback_last_n = fallback_last_n
        self.cache_size = int(cache_size)

        # Simple LRU-ish cache keyed by sha1(sequence)
        self._cache: Dict[str, Optional[Tuple[int, int]]] = {}
        self._cache_order: List[str] = []

    # ---- selection logic ----
    def _hit_interval(self, hit: HMMHit) -> Tuple[int, int]:
        if self.use_env_coords:
            s, t = hit.env_from, hit.env_to
        else:
            s, t = hit.ali_from, hit.ali_to
        if s > t:
            s, t = t, s
        return s, t

    def _hit_ali_len(self, hit: HMMHit) -> int:
        s, t = self._hit_interval(hit)
        return t - s + 1

    def _is_candidate(self, hit: HMMHit) -> bool:
        if hit.model not in self.model_rank:
            return False
        if hit.i_evalue > self.evalue_cutoff:
            return False
        if self._hit_ali_len(hit) < self.min_ali_len:
            return False
        return True

    def _choose_best_hit(self, hits: List[HMMHit]) -> Optional[HMMHit]:
        """
        Pick the highest-confidence PI-family hit.

        Ranking:
          1) model preference (Cas9_PI preferred over PI_C over PI2 over CTD)
          2) lowest i-evalue
          3) highest bitscore
        """
        candidates = [h for h in hits if self._is_candidate(h)]
        if not candidates:
            return None

        def key(h: HMMHit):
            return (self.model_rank[h.model], h.i_evalue, -h.bitscore)

        return min(candidates, key=key)

    def _cap_interval(self, s: int, t: int, seq_len: int) -> Tuple[int, int]:
        """
        Cap [s,t] to length max_mask_len around the alignment core midpoint.
        """
        if s > t:
            s, t = t, s

        s = max(1, min(s, seq_len))
        t = max(1, min(t, seq_len))

        L = t - s + 1
        if L <= self.max_mask_len:
            return s, t

        mid = (s + t) // 2
        half = self.max_mask_len // 2
        s2 = mid - half
        t2 = s2 + self.max_mask_len - 1

        if s2 < 1:
            s2 = 1
            t2 = min(seq_len, self.max_mask_len)
        if t2 > seq_len:
            t2 = seq_len
            s2 = max(1, seq_len - self.max_mask_len + 1)

        return s2, t2

    # ---- caching ----
    def _seq_key(self, seq: str) -> str:
        return hashlib.sha1(seq.encode("utf-8")).hexdigest()

    def _cache_get(self, seq: str) -> Optional[Tuple[int, int]]:
        k = self._seq_key(seq)
        if k in self._cache:
            return self._cache[k]
        return None

    def _cache_put(self, seq: str, interval: Optional[Tuple[int, int]]) -> None:
        k = self._seq_key(seq)
        if k in self._cache:
            self._cache[k] = interval
            return
        self._cache[k] = interval
        self._cache_order.append(k)
        if len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

    # ---- public API ----
    def pi_core_interval(self, seq: str) -> Optional[Tuple[int, int]]:
        """
        Returns (start,end) 1-based AA coordinates of the compact PI-core region,
        after applying the max_mask_len cap.

        If no PI hit is found and fallback_last_n is set, returns last N AAs.
        """
        seq = seq.replace(" ", "").strip()
        if not seq:
            return None

        cached = self._cache_get(seq)
        if cached is not None or (self._seq_key(seq) in self._cache):
            # cached could be None meaning "no hit"
            return cached

        domtbl, _ = self.hmmscan.scan_sequences(seq, ids="q")
        hits = DomTbloutParser.parse(domtbl)
        best = self._choose_best_hit(hits)

        if best is None:
            if self.fallback_last_n is None:
                self._cache_put(seq, None)
                return None
            n = int(self.fallback_last_n)
            L = len(seq)
            s = max(1, L - n + 1)
            t = L
            interval = (s, t)
            self._cache_put(seq, interval)
            return interval

        s, t = self._hit_interval(best)
        interval = self._cap_interval(s, t, seq_len=len(seq))
        self._cache_put(seq, interval)
        return interval

    def build_no_del_mask(
        self,
        x: torch.Tensor,
        aa_seq: str,
        pad_id: int,
        *,
        bos_at_index0: bool = True,
    ) -> torch.Tensor:
        """
        Build (B,Lmax) bool mask where True => deletion disallowed.
        Currently supports B=1 (single accepted sequence), which matches your use.
        """
        if x.dim() != 2 or x.shape[0] != 1:
            raise ValueError(f"Expected x shape (1,Lmax), got {tuple(x.shape)}")

        Lmax = x.shape[1]
        device = x.device
        mask = torch.zeros((1, Lmax), device=device, dtype=torch.bool)

        aa_seq = aa_seq.replace(" ", "").strip()
        seq_len = len(aa_seq)

        interval = self.pi_core_interval(aa_seq)
        if interval is None:
            return mask

        s, t = interval  # 1-based AA positions
        s = max(1, min(s, seq_len))
        t = max(1, min(t, seq_len))
        if s > t:
            s, t = t, s

        # Token mapping:
        # if BOS at index 0, AA position p is at token index p
        # else AA position p is at token index p-1
        offset = 0 if bos_at_index0 else -1
        ts = s + offset
        tt = t + offset

        ts = max(0, min(ts, Lmax - 1))
        tt = max(0, min(tt, Lmax - 1))
        if ts > tt:
            ts, tt = tt, ts

        mask[0, ts:tt + 1] = True

        # Don't bother masking PAD; your deletion rates are already masked on PAD tokens.
        # But if you want, you can explicitly clear PAD positions:
        mask[0, x[0] == pad_id] = False

        return mask
