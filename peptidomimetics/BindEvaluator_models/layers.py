from torch import nn
from .modules import MultiHeadAttentionSequence, MultiHeadAttentionReciprocal, FFN, _as_bool_mask, _zero_out_padded


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class DilatedCNN(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.first_ = nn.ModuleList()
        self.second_ = nn.ModuleList()
        self.third_ = nn.ModuleList()

        dilation_tuple = (1, 2, 3)
        dim_in_tuple = (d_model, d_hidden, d_hidden)
        dim_out_tuple = (d_hidden, d_hidden, d_hidden)

        for i, dilation_rate in enumerate(dilation_tuple):
            self.first_.append(
                ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
            )
        for i, dilation_rate in enumerate(dilation_tuple):
            self.second_.append(
                ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=5, padding=2 * dilation_rate, dilation=dilation_rate)
            )
        for i, dilation_rate in enumerate(dilation_tuple):
            self.third_.append(
                ConvLayer(dim_in_tuple[i], dim_out_tuple[i], kernel_size=7, padding=3 * dilation_rate, dilation=dilation_rate)
            )

    def forward(self, protein_seq_enc, padding_mask=None):
        """
        protein_seq_enc: [B, L, D]
        padding_mask:    [B, L] (1/0 or bool)
        """
        mask = _as_bool_mask(padding_mask)
        protein_seq_enc = _zero_out_padded(protein_seq_enc, mask)

        x = protein_seq_enc.transpose(1, 2)  # [B, D, L]

        first_embedding = x
        second_embedding = x
        third_embedding = x

        for layer in self.first_:
            first_embedding = layer(first_embedding)
        for layer in self.second_:
            second_embedding = layer(second_embedding)
        for layer in self.third_:
            third_embedding = layer(third_embedding)

        x = first_embedding + second_embedding + third_embedding
        x = x.transpose(1, 2)  # [B, L, D]

        x = _zero_out_padded(x, mask)
        return x


class ReciprocalLayerwithCNN(nn.Module):
    def __init__(self, d_model, d_inner, d_hidden, n_head, d_k, d_v):
        super().__init__()

        self.cnn = DilatedCNN(d_model, d_hidden)

        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_hidden, d_k, d_v)
        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_hidden, d_k, d_v)

        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_hidden, d_k, d_v)

        self.ffn_seq = FFN(d_hidden, d_inner)
        self.ffn_protein = FFN(d_hidden, d_inner)

    def forward(self, sequence_enc, protein_seq_enc, sequence_mask=None, protein_mask=None):
        """
        sequence_enc:     [B, Ls, D]
        protein_seq_enc:  [B, Lp, D]
        sequence_mask:    [B, Ls] 1/0 or bool
        protein_mask:     [B, Lp] 1/0 or bool
        """
        s_mask = _as_bool_mask(sequence_mask)
        p_mask = _as_bool_mask(protein_mask)

        sequence_enc = _zero_out_padded(sequence_enc, s_mask)
        protein_seq_enc = _zero_out_padded(protein_seq_enc, p_mask)

        protein_seq_enc = self.cnn(protein_seq_enc, padding_mask=p_mask)

        prot_enc, prot_attention = self.protein_attention_layer(
            protein_seq_enc, protein_seq_enc, protein_seq_enc,
            key_padding_mask=p_mask, query_padding_mask=p_mask
        )

        seq_enc, sequence_attention = self.sequence_attention_layer(
            sequence_enc, sequence_enc, sequence_enc,
            key_padding_mask=s_mask, query_padding_mask=s_mask
        )

        # prot attends to seq (keys=seq), and seq attends to prot (keys=prot)
        prot_enc, seq_enc, prot_seq_attention, seq_prot_attention = self.reciprocal_attention_layer(
            prot_enc, seq_enc, v=seq_enc, v_2=prot_enc,
            q_padding_mask=p_mask, k_padding_mask=s_mask
        )

        prot_enc = self.ffn_protein(prot_enc, padding_mask=p_mask)
        seq_enc = self.ffn_seq(seq_enc, padding_mask=s_mask)

        return prot_enc, seq_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention


class ReciprocalLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v):
        super().__init__()

        self.sequence_attention_layer = MultiHeadAttentionSequence(n_head, d_model, d_k, d_v)
        self.protein_attention_layer = MultiHeadAttentionSequence(n_head, d_model, d_k, d_v)
        self.reciprocal_attention_layer = MultiHeadAttentionReciprocal(n_head, d_model, d_k, d_v)

        self.ffn_seq = FFN(d_model, d_inner)
        self.ffn_protein = FFN(d_model, d_inner)

    def forward(self, sequence_enc, protein_seq_enc, sequence_mask=None, protein_mask=None):
        s_mask = _as_bool_mask(sequence_mask)
        p_mask = _as_bool_mask(protein_mask)

        sequence_enc = _zero_out_padded(sequence_enc, s_mask)
        protein_seq_enc = _zero_out_padded(protein_seq_enc, p_mask)

        prot_enc, prot_attention = self.protein_attention_layer(
            protein_seq_enc, protein_seq_enc, protein_seq_enc,
            key_padding_mask=p_mask, query_padding_mask=p_mask
        )

        seq_enc, sequence_attention = self.sequence_attention_layer(
            sequence_enc, sequence_enc, sequence_enc,
            key_padding_mask=s_mask, query_padding_mask=s_mask
        )

        prot_enc, seq_enc, prot_seq_attention, seq_prot_attention = self.reciprocal_attention_layer(
            prot_enc, seq_enc, v=seq_enc, v_2=prot_enc,
            q_padding_mask=p_mask, k_padding_mask=s_mask
        )

        prot_enc = self.ffn_protein(prot_enc, padding_mask=p_mask)
        seq_enc = self.ffn_seq(seq_enc, padding_mask=s_mask)

        return prot_enc, seq_enc, prot_attention, sequence_attention, prot_seq_attention, seq_prot_attention
