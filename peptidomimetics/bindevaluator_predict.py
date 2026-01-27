import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelWithLMHead, EsmModel
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, accuracy_score
import yaml
from easydict import EasyDict as edict
from BindEvaluator_models import * 

import pdb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from transformers.utils import logging
logging.set_verbosity_error()

def parse_motif(motif: str) -> list:
    parts = motif.split(',')
    result = []

    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))

    return result


class BindEvaluator(pl.LightningModule):
    def __init__(self, cfg):
        super(BindEvaluator, self).__init__()

        self.esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval()
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.chemberta_model = AutoModelWithLMHead.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k").roberta.eval()
        for param in self.chemberta_model.parameters():
            param.requires_grad = False

        self.repeated_module = RepeatedModule(cfg.model.n_layers, cfg.model.d_model, cfg.model.d_hidden,
                                               cfg.model.n_head, cfg.model.d_k, cfg.model.d_v, cfg.model.d_inner, dropout=cfg.model.dropout)

        self.final_attention_layer = MultiHeadAttentionSequence(cfg.model.n_head, cfg.model.d_model,
                                                                cfg.model.d_k, cfg.model.d_v, dropout=cfg.model.dropout)

        self.final_ffn = FFN(cfg.model.d_model, cfg.model.d_inner, dropout=cfg.model.dropout)

        self.output_projection_prot = nn.Linear(cfg.model.d_model, 1)

        self.class_weights = torch.tensor([7.803861120886587, 0.5342284711965681])  # binding_site weights, non-bidning site weights

        self.cfg = cfg
        self._total_steps = None

    def forward(self, binder_tokens, target_tokens):
        peptide_sequence = self.chemberta_model(**binder_tokens).last_hidden_state
        protein_sequence = self.esm_model(**target_tokens).last_hidden_state

        binder_mask = binder_tokens["attention_mask"]      # [B, Ls]
        target_mask = target_tokens["attention_mask"]      # [B, Lp]
    
        prot_enc, sequence_enc, sequence_attention_list, prot_attention_list, \
            prot_seq_attention_list, seq_prot_attention_list = self.repeated_module(
                peptide_sequence,
                protein_sequence,
                peptide_mask=binder_mask,
                protein_mask=target_mask,
            )
    
        # final cross-attention: protein queries attend to binder keys
        prot_enc, final_prot_seq_attention = self.final_attention_layer(
            prot_enc, sequence_enc, sequence_enc,
            key_padding_mask=binder_mask,
            query_padding_mask=target_mask,
        )
    
        prot_enc = self.final_ffn(prot_enc, padding_mask=target_mask)
        prot_enc = self.output_projection_prot(prot_enc)
        return prot_enc


def compute_metrics(true_residues, predicted_residues, length):
    # Initialize the true and predicted lists with 0
    true_list = [0] * length
    predicted_list = [0] * length

    # Set the values to 1 based on the provided lists
    for index in true_residues:
        true_list[index] = 1
    for index in predicted_residues:
        predicted_list[index] = 1

    # Compute the metrics
    accuracy = accuracy_score(true_list, predicted_list)
    f1 = f1_score(true_list, predicted_list)
    mcc = matthews_corrcoef(true_list, predicted_list)

    return accuracy, f1, mcc


def main():
    CONFIG_PATH = './bindevaluator_config.yaml'
    with open(CONFIG_PATH, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = edict(config_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BindEvaluator.load_from_checkpoint(cfg.inference.ckpt, cfg=cfg, map_location=device)
    model.eval()

    binder_seq = cfg.inference.binder
    target_seq = cfg.inference.target

    smiles_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa_zinc250k_v2_40k")
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    binder = smiles_tokenizer(binder_seq, return_tensors='pt').to(device)
    target = esm_tokenizer(target_seq, return_tensors='pt').to(device)

    prediction = model(binder, target).squeeze(-1)
    probs = torch.sigmoid(prediction)   # (1, L)

    threshold = cfg.inference.threshold
    binding_site = []
    for i in range(probs.shape[1]):
        if probs[0][i] >= threshold:
            binding_site.append(i)

    print("Predicted Binding Sites: ", binding_site)
    if cfg.inference.ground_truth is not None:
        print("Ground Truth Binding Sites: ", cfg.inference.ground_truth)
        motifs = parse_motif(cfg.inference.ground_truth)

        motif_scores = probs[:, motifs].mean(dim=1)
        non_motif_probs = probs[:, [i for i in range(probs.shape[1]) if i not in motifs]]
        mask = non_motif_probs >= threshold
        count = mask.sum(dim=-1)
        specificity = 1 - count / len(target_seq)

        print(f"Motif Score: {motif_scores.item()}")
        print(f"Specificity Score: {specificity.item()}")
if __name__ == "__main__":
    main()