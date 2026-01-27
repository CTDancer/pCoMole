import torch
import torch.nn as nn
from .layers import *
from .modules import _as_bool_mask, _zero_out_padded


class RepeatedModule(nn.Module):
    def __init__(self, n_layers, d_model, d_hidden, n_head, d_k, d_v, d_inner, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(768, d_model)   # binder encoder hidden size
        self.linear2 = nn.Linear(1280, d_model)  # protein encoder hidden size
        self.d_model = d_model

        self.reciprocal_layer_stack = nn.ModuleList([
            ReciprocalLayerwithCNN(d_model, d_inner, d_hidden, n_head, d_k, d_v)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, peptide_sequence, protein_sequence, peptide_mask=None, protein_mask=None):
        """
        peptide_sequence: [B, Ls, 768]
        protein_sequence: [B, Lp, 1280]
        peptide_mask:     [B, Ls] 1/0 or bool
        protein_mask:     [B, Lp] 1/0 or bool
        """
        s_mask = _as_bool_mask(peptide_mask)
        p_mask = _as_bool_mask(protein_mask)

        sequence_attention_list = []
        prot_attention_list = []
        prot_seq_attention_list = []
        seq_prot_attention_list = []

        # project to common d_model
        sequence_enc = self.dropout(self.linear1(peptide_sequence))
        prot_enc = self.dropout_2(self.linear2(protein_sequence))

        # IMPORTANT: zero padded positions before any downstream layers
        sequence_enc = _zero_out_padded(sequence_enc, s_mask)
        prot_enc = _zero_out_padded(prot_enc, p_mask)

        for reciprocal_layer in self.reciprocal_layer_stack:
            prot_enc, sequence_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention = \
                reciprocal_layer(sequence_enc, prot_enc, sequence_mask=s_mask, protein_mask=p_mask)

            sequence_attention_list.append(sequence_attention)
            prot_attention_list.append(prot_attention)
            prot_seq_attention_list.append(prot_seq_attention)
            seq_prot_attention_list.append(seq_prot_attention)

        return (
            prot_enc,
            sequence_enc,
            sequence_attention_list,
            prot_attention_list,
            prot_seq_attention_list,
            seq_prot_attention_list,
        )
