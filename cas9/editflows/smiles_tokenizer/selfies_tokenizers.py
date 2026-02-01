import os
import math
import pandas as pd
from datasets import Dataset, DatasetDict
from typing import List, Dict, Any, Iterable, Set
from tqdm import tqdm
import numpy as np
import selfies as sf
import json
from rdkit import Chem
from rdkit import RDLogger
import pdb

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

class SelfiesTokenizer:
    """
    Minimal SELFIES tokenizer:
      - build_vocab(selfies_list) -> token->id mapping
      - encode(smiles_or_selfies, already_selfies=False) -> {'input_ids': List[int]}
      - decode(ids) -> selfies string (no BOS/EOS returned)
    """
    def __init__(self, vocab: Dict[str,int] = None):
        self.vocab = vocab or {}
        # filled after build_vocab if not provided
        self.inv_vocab = {i:t for t,i in (self.vocab or {}).items()}
        self.vocab_size = len(self.vocab)

    @staticmethod
    def smiles_to_selfies(s: str) -> str:
        return sf.encoder(s)

    @staticmethod
    def selfies_tokens(s: str) -> List[str]:
        return list(sf.split_selfies(s))

    # def vocab_size(self) -> int:
    #     return len(self.vocab)

    def build_vocab(self, selfies_corpus: Iterable[str], add_semantic_default: bool = False):
        tokens: Set[str] = set()
        for s in selfies_corpus:
            tokens.update(self.selfies_tokens(s))

        if add_semantic_default:
            tokens.update(sf.get_semantic_robust_alphabet())

        # special tokens first with fixed ids
        vocab_list = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + sorted(tokens)
        self.vocab = {tok: i for i, tok in enumerate(vocab_list)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, already_selfies: bool = True, add_bos_eos: bool = True) -> Dict[str, List[int]]:
        if not already_selfies:
            # optionally validate SMILES with RDKit to avoid weird cases
            if text and Chem.MolFromSmiles(text) is None:
                return {"input_ids": []}  # skip invalid
            text = sf.encoder(text)

        toks = self.selfies_tokens(text)
        ids = []
        if add_bos_eos:
            ids.append(self.vocab[BOS_TOKEN])
        for t in toks:
            if t not in self.vocab:
                # unseen token -> extend vocab or skip; here we skip sample
                return {"input_ids": []}
            ids.append(self.vocab[t])
        if add_bos_eos:
            ids.append(self.vocab[EOS_TOKEN])
        
        attention_mask = [1] * len(ids)
        return {"input_ids": ids, "attention_mask": attention_mask}

    def save(self, folder, constraints = None, version = None):
        os.makedirs(folder, exist_ok=True)
        meta = {
            "type": "selfies-wordlevel",
            "version": version or getattr(sf, "__version__", None),
            "special_tokens": {
                "pad": PAD_TOKEN, "bos": BOS_TOKEN, "eos": EOS_TOKEN
            },
            "semantic_constraints": constraints or None,
        }
        with open(os.path.join(folder, "vocab.json"), "w") as f:
            json.dump(self.vocab, f, indent=2, sort_keys=True)
        with open(os.path.join(folder, "tokenizer_config.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, file) -> "SelfiesTokenizer":
        with open(file) as f:
            vocab = json.load(f)
        tok = cls(vocab=vocab)
        return tok
    
    def _ids_to_selfies_tokens(
        self,
        ids: list[int],
        stop_at_eos: bool = True,
    ) -> list[str]:
        """Map ids -> SELFIES tokens, filtering PAD/BOS/EOS and (optionally) stopping at EOS."""
        toks = []
        for i in ids:
            tok = self.inv_vocab.get(i, None)
            # if tok is None:
            #     continue
            # if tok == '<pad>':
            #     continue
            # if tok == '<bos>':
            #     continue
            # if tok == '<eos>':
            #     if stop_at_eos:
            #         break
            #     else:
            #         continue
            toks.append(tok)
        return toks

    def decode(
        self,
        ids: list[int],
        return_smiles: bool = True,
        stop_at_eos: bool = False,
        sanitize_smiles: bool = False,
    ) -> str:
        """
        Decode a sequence of token ids.

        Args:
            ids: list of token ids.
            return_smiles: if True, return canonical SMILES; else return SELFIES string.
            stop_at_eos: stop decoding when <eos> is encountered.
            sanitize_smiles: if True (and RDKit available), canonicalize decoded SMILES.

        Returns:
            str: SMILES (default) or SELFIES string.
        """
        # 1) ids -> SELFIES tokens -> SELFIES string
        toks = self._ids_to_selfies_tokens(ids, stop_at_eos=stop_at_eos)
        selfies_str = "".join(toks)
        print(selfies_str)
        # pdb.set_trace()

        if not return_smiles:
            return selfies_str

        # 2) SELFIES -> SMILES
        try:
            smiles = sf.decoder(selfies_str)
        except Exception:
            return ""  # decoding failed

        if not sanitize_smiles:
            return smiles

        # 3) Optional RDKit sanitization / canonicalization
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            # Canonical SMILES (Kekule off, isomeric on to keep stereo)
            can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            return can
        except Exception:
            return smiles  # fall back to raw decoded SMILES

    # (optional) convenience for batches
    def batch_decode(
        self,
        batch_ids: list[list[int]],
        return_smiles: bool = True,
        stop_at_eos: bool = True,
        sanitize_smiles: bool = True,
    ) -> list[str]:
        return [
            self.decode(
                ids,
                return_smiles=return_smiles,
                stop_at_eos=stop_at_eos,
                sanitize_smiles=sanitize_smiles,
            )
            for ids in batch_ids
        ]