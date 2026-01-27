from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


def _as_bool_mask(mask):
    """
    Accepts attention_mask style (1=valid, 0=pad) or bool mask.
    Returns bool mask with True=valid, False=pad.
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    # assume 1/0
    return mask != 0


def _mask_attn_logits(attn_logits, key_mask):
    """
    attn_logits: [B, H, Lq, Lk]
    key_mask:    [B, Lk]  (True=valid, False=pad)
    """
    if key_mask is None:
        return attn_logits
    # mask pads => set to very negative before softmax
    # shape: [B, 1, 1, Lk]
    km = key_mask[:, None, None, :]
    return attn_logits.masked_fill(~km, torch.finfo(attn_logits.dtype).min)


def _zero_out_padded(x, mask) -> torch.Tensor:
    """
    x: [B, L, D]
    mask: [B, L] True=valid
    """
    if mask is None:
        return x
    return x * mask.unsqueeze(-1).type_as(x)


class MultiHeadAttentionSequence(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_head * d_k)
        self.W_K = nn.Linear(d_model, n_head * d_k)
        self.W_V = nn.Linear(d_model, n_head * d_v)
        self.W_O = nn.Linear(n_head * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        v,
        key_padding_mask,   # [B, Lk] 1/0 or bool
        query_padding_mask, # [B, Lq] 1/0 or bool
    ):
        """
        q: [B, Lq, D], k/v: [B, Lk, D]
        Returns:
            output: [B, Lq, D]
            attention: [B, H, Lq, Lk]  (softmaxed)
        """
        key_mask = _as_bool_mask(key_padding_mask)
        q_mask = _as_bool_mask(query_padding_mask)

        batch, len_q, _ = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()
        assert len_k == len_v, "k and v must have same length"

        Q = self.W_Q(q).view(batch, len_q, self.n_head, self.d_k)
        K = self.W_K(k).view(batch, len_k, self.n_head, self.d_k)
        V = self.W_V(v).view(batch, len_v, self.n_head, self.d_v)

        # [B, H, Lq, Dk]
        Q = Q.transpose(1, 2)
        # [B, H, Dk, Lk]
        K = K.transpose(1, 2).transpose(2, 3)
        # [B, H, Lk, Dv]
        V = V.transpose(1, 2)

        attn_logits = torch.matmul(Q, K) / np.sqrt(self.d_k)  # [B, H, Lq, Lk]
        attn_logits = _mask_attn_logits(attn_logits, key_mask)

        attention = F.softmax(attn_logits, dim=-1)
        # If some rows are all-masked (shouldn't happen for valid queries), softmax can produce NaNs.
        # We defensively zero them.
        attention = torch.nan_to_num(attention, nan=0.0)

        output = torch.matmul(attention, V)  # [B, H, Lq, Dv]
        output = output.transpose(1, 2).reshape(batch, len_q, self.d_v * self.n_head)
        output = self.W_O(output)
        output = self.dropout(output)

        output = self.layer_norm(output + q)

        # ensure padded queries don't carry signal downstream
        output = _zero_out_padded(output, q_mask)

        return output, attention


class MultiHeadAttentionReciprocal(nn.Module):
    """
    Reciprocal cross-attention:
      - output  uses q attending to k (keys= k)
      - output_2 uses k attending to q (keys= q)  (via transposed logits)
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, n_head * d_k)
        self.W_K = nn.Linear(d_model, n_head * d_k)

        self.W_V = nn.Linear(d_model, n_head * d_v)
        self.W_O = nn.Linear(n_head * d_v, d_model)

        self.W_V_2 = nn.Linear(d_model, n_head * d_v)
        self.W_O_2 = nn.Linear(n_head * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(
        self,
        q,    # [B, Lq, D]
        k,    # [B, Lk, D]
        v,    # [B, Lk, D] values for output (q->k)
        v_2,  # [B, Lq, D] values for output_2 (k->q)
        q_padding_mask,  # [B, Lq]
        k_padding_mask,  # [B, Lk]
    ):
        q_mask = _as_bool_mask(q_padding_mask)
        k_mask = _as_bool_mask(k_padding_mask)

        batch, len_q, _ = q.size()
        _, len_k, _ = k.size()

        Q = self.W_Q(q).view(batch, len_q, self.n_head, self.d_k)
        K = self.W_K(k).view(batch, len_k, self.n_head, self.d_k)

        V = self.W_V(v).view(batch, len_k, self.n_head, self.d_v)
        V_2 = self.W_V_2(v_2).view(batch, len_q, self.n_head, self.d_v)

        Q = Q.transpose(1, 2)                       # [B, H, Lq, Dk]
        Kt = K.transpose(1, 2).transpose(2, 3)      # [B, H, Dk, Lk]
        V = V.transpose(1, 2)                       # [B, H, Lk, Dv]
        V_2 = V_2.transpose(1, 2)                   # [B, H, Lq, Dv]

        attn_logits = torch.matmul(Q, Kt) / np.sqrt(self.d_k)  # [B, H, Lq, Lk]
        attn_logits = _mask_attn_logits(attn_logits, k_mask)

        attn_logits_2 = attn_logits.transpose(-2, -1)          # [B, H, Lk, Lq]
        attn_logits_2 = _mask_attn_logits(attn_logits_2, q_mask)

        attention = F.softmax(attn_logits, dim=-1)
        attention_2 = F.softmax(attn_logits_2, dim=-1)

        attention = torch.nan_to_num(attention, nan=0.0)
        attention_2 = torch.nan_to_num(attention_2, nan=0.0)

        out = torch.matmul(attention, V)      # [B, H, Lq, Dv]
        out2 = torch.matmul(attention_2, V_2) # [B, H, Lk, Dv]

        out = out.transpose(1, 2).reshape(batch, len_q, self.d_v * self.n_head)
        out2 = out2.transpose(1, 2).reshape(batch, len_k, self.d_v * self.n_head)

        out = self.W_O(out)
        out2 = self.W_O_2(out2)

        out = self.dropout(out)
        out = self.layer_norm(out + q)

        out2 = self.dropout_2(out2)
        out2 = self.layer_norm_2(out2 + k)

        out = _zero_out_padded(out, q_mask)
        out2 = _zero_out_padded(out2, k_mask)

        return out, out2, attention, attention_2


class FFN(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.layer_1 = nn.Conv1d(d_in, d_hid, 1)
        self.layer_2 = nn.Conv1d(d_hid, d_in, 1)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        """
        x: [B, L, D]
        padding_mask: [B, L] (1/0 or bool). Pads will be zeroed before/after FFN.
        """
        mask = _as_bool_mask(padding_mask)
        x = _zero_out_padded(x, mask)

        residual = x
        out = self.layer_1(x.transpose(1, 2))
        out = self.relu(out)
        out = self.layer_2(out)
        out = self.dropout(out)
        out = self.layer_norm(out.transpose(1, 2) + residual)

        out = _zero_out_padded(out, mask)
        return out
