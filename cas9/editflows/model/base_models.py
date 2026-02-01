from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel
import pytorch_lightning as pl

from .utils import build_z0_z1_with_alignment, remove_eps
import pdb

# ---------- Utilities ----------

def exists(x): return x is not None

def default(val, d):
    return val if exists(val) else d

# ---------- Timestep embedding (sinusoidal -> MLP) ----------

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding followed by a small MLP.
    Accepts t of shape (B,) or scalar; outputs (B, d_model) and broadcasts over L.
    """
    def __init__(self, d_model: int, hidden: Optional[int] = None, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        hidden = default(hidden, d_model * 4)
        # Use even dim for sin/cos pairing
        pe_dim = d_model if d_model % 2 == 0 else d_model - 1
        self.pe_dim = pe_dim
        self.mlp = nn.Sequential(
            nn.Linear(pe_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, t: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        t: (B,) or () in [0,1]
        returns: (B, d_model)
        """
        if t.dim() == 0:
            # scalar -> expand to batch
            if batch_size is None:
                raise ValueError("When t is scalar, provide batch_size.")
            t = t.expand(batch_size)
        B = t.shape[0]
        device = t.device
        half = self.pe_dim // 2
        # frequencies
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=t.dtype) * (-math.log(self.max_period) / (half - 1 + 1e-8))
        )
        angles = t[:, None] * freqs[None, :] * math.pi  # (B, half)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, pe_dim)
        if self.pe_dim < self.d_model:  # rare case when d_model is odd
            pe = F.pad(pe, (0, 1), value=0.0)
        return self.mlp(pe)  # (B, d_model)

# ---------- RoPE (rotary position embedding) ----------

def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: (B, h, L, d_head)
    cos, sin: (L, d_head) broadcastable to (B, h, L, d_head)
    """
    # split last dim into pairs
    d = q.shape[-1]
    if d % 2 != 0:
        # pad to even
        q = F.pad(q, (0, 1), value=0.0)
        k = F.pad(k, (0, 1), value=0.0)
        d += 1

    q1, q2 = q[..., :d//2], q[..., d//2:]
    k1, k2 = k[..., :d//2], k[..., d//2:]

    # broadcast cos/sin
    while cos.dim() < q1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    rq = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    rk = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return rq[..., :q.shape[-1]], rk[..., :k.shape[-1]]

class RotaryPositionalEmbedding(nn.Module):
    """
    Precomputes cos/sin for RoPE given max_len and head_dim.
    """
    def __init__(self, head_dim: int, max_len: int = 8192, base: int = 10000):
        super().__init__()
        if head_dim % 2 != 0:
            # allow odd by padding inside apply_rotary, but prefer even
            pass
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_len = max_len
        self.head_dim = head_dim
        self._cached_len = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _update_cache(self, seq_len: int, device, dtype):
        if seq_len <= self._cached_len and self.cos_cached.device == device and self.cos_cached.dtype == dtype:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('n,d->nd', t, self.inv_freq)  # (L, head_dim/2)
        # emb = torch.cat((freqs, freqs), dim=-1)  # (L, head_dim)
        self.cos_cached = freqs.cos().to(dtype=dtype)
        self.sin_cached = freqs.sin().to(dtype=dtype)
        self._cached_len = seq_len

    def forward(self, L: int, device, dtype):
        self._update_cache(L, device, dtype)
        return self.cos_cached[:L], self.sin_cached[:L]

# ---------- Rotary MHA + Transformer block ----------

class RotaryMHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: RotaryPositionalEmbedding, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: (B, L, d_model)
        key_padding_mask: (B, L) bool, True=pad (masked)
        """
        B, L, D = x.shape
        qkv = self.qkv(x)  # (B, L, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, h, L, d)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(L, x.device, x.dtype)
        q, k = apply_rotary(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, h, L, L)

        if exists(key_padding_mask):
            # mask: True means pad -> set attention to -inf
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)  # (B,1,1,L)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B, h, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj_dropout(self.proj(out))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, attn_dropout: float = 0.0, proj_dropout: float = 0.0, rope: Optional[RotaryPositionalEmbedding] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RotaryMHA(d_model, n_heads, rope=rope, attn_dropout=attn_dropout, proj_dropout=proj_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class ProteinEditFlowModel(nn.Module):
    """
    Inputs:
      x_t: (B, L) Long
      mask: (B, L) bool, True=pad (i.e., should be ignored)
      t: (B,) or scalar in [0,1]

    Outputs:
      lam_ins:  (B, L)         >= 0
      logits_ins: (B, L, V)
      lam_del:  (B, L)         >= 0
      lam_sub:  (B, L)         >= 0
      logits_sub: (B, L, V)
    """
    def __init__(self, vocab_size, pad_id, config):
        super().__init__()
        self.d_model = getattr(config, "d_model", 768)
        self.n_layers = getattr(config, "n_layers", 12)
        self.n_heads = getattr(config, "n_heads", 12)
        self.mlp_ratio = getattr(config, "mlp_ratio", 4)
        self.max_len = getattr(config, "max_len", 2048)
        self.dropout = getattr(config, "dropout", 0.1)
        self.attn_dropout = getattr(config, "attn_dropout", 0)
        self.proj_dropout = getattr(config, "proj_dropout", 0)
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # --- Embedding ---
        esm_model_name = getattr(config, "esm_model_name", "facebook/esm2_t33_650M_UR50D")
        freeze_esm = getattr(config, "freeze_esm", True)
        self.esm_emb = EsmModel.from_pretrained(esm_model_name)
        if freeze_esm:
            for param in self.esm_emb.parameters():
                param.requires_grad = False
        self.time_emb = TimeEmbedding(d_model=self.d_model)

        self.tok_embed_to_hidden = nn.Linear(1280, self.d_model)

        # --- RoPE shared by attention blocks ---
        rope = RotaryPositionalEmbedding(head_dim=self.d_model // self.n_heads, max_len=self.max_len)

        # --- Encoder ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                proj_dropout=self.proj_dropout,
                rope=rope
            )
            for _ in range(self.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        # --- Heads ---
        # We use small MLP heads for rates; logits are linear.
        self.lam_ins_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.lam_del_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.lam_sub_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.logits_ins_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.logits_sub_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # nonnegativity via softplus (safer than exp)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(
        self,
        x_t: torch.LongTensor,
        mask: torch.BoolTensor,
        t: torch.Tensor,
    ):
        """
        x_t: (B, L) long tokens
        mask: (B, L) bool
        t: (B,) or scalar in [0,1]
        esmbed: optional (B, L, esm2_embed_dim) if use_real_esm2=True
        """
        B, L = x_t.shape
        # pdb.set_trace()

        # --- Embedding ---
        h = self.esm_emb(x_t, mask).last_hidden_state
        h = self.tok_embed_to_hidden(h)

        # print(f"mask: {mask.tolist()}")

        # --- Add time embedding (broadcast across length) ---
        t_emb = self.time_emb(t, batch_size=B)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)  # (B, L, d_model)

        # --- Encoder blocks with key padding mask ---
        for blk in self.blocks:
            h = blk(h, key_padding_mask=(~mask))

        h = self.final_norm(h)  # (B, L, d_model)

        # --- Heads ---
        lam_ins = self.softplus(self.lam_ins_head(h)).squeeze(-1)  # (B, L)
        lam_del = self.softplus(self.lam_del_head(h)).squeeze(-1)  # (B, L)
        lam_sub = self.softplus(self.lam_sub_head(h)).squeeze(-1)  # (B, L)
        logits_ins = self.logits_ins_head(h)  # (B, L, V)
        logits_sub = self.logits_sub_head(h)  # (B, L, V)

        # --- Zero-out padded positions so they contribute nothing downstream ---
        if exists(mask):
            # For lambdas: force to 0 on pads
            pad_mask_f = mask.to(h.dtype)  # True=valid -> 1.0
            lam_ins = lam_ins * pad_mask_f
            lam_del = lam_del * pad_mask_f
            lam_sub = lam_sub * pad_mask_f

            # kill logits on pads
            neg_val = torch.tensor(-1e4, device=h.device, dtype=h.dtype)
            logits_ins = logits_ins.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_sub = logits_sub.masked_fill((~mask).unsqueeze(-1), neg_val)

        return lam_ins, logits_ins, lam_del, lam_sub, logits_sub

    def get_esm_embeddings(self, x_t, mask):
        """
        Extract ESM embeddings for given tokens.
        
        This method allows external code to reuse the ESM model for other purposes
        (e.g., sharing with Cas9Classifier to save memory).
        
        Args:
            x_t: (B, L) LongTensor - tokenized protein sequences
            mask: (B, L) BoolTensor - True for valid tokens, False for padding
        
        Returns:
            embeddings: (B, L, esm_hidden_size) - ESM embeddings
        """
        return self.esm_emb(x_t, attention_mask=mask).last_hidden_state

class ReparameterizedProteinEditFlowModel(nn.Module):
    """
    Inputs:
      x_t: (B, L) Long
      mask: (B, L) bool, True=pad (i.e., should be ignored)
      t: (B,) or scalar in [0,1]

    Outputs:
      lam_ins:  (B, L)         >= 0
      logits_ins: (B, L, V)
      lam_del:  (B, L)         >= 0
      lam_sub:  (B, L)         >= 0
      logits_sub: (B, L, V)
    """
    def __init__(self, vocab_size, pad_id, config):
        super().__init__()
        self.d_model = getattr(config, "d_model", 768)
        self.n_layers = getattr(config, "n_layers", 12)
        self.n_heads = getattr(config, "n_heads", 12)
        self.mlp_ratio = getattr(config, "mlp_ratio", 4)
        self.max_len = getattr(config, "max_len", 2048)
        self.dropout = getattr(config, "dropout", 0.1)
        self.attn_dropout = getattr(config, "attn_dropout", 0)
        self.proj_dropout = getattr(config, "proj_dropout", 0)
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # --- Embedding ---
        esm_model_name = getattr(config, "esm_model_name", "facebook/esm2_t33_650M_UR50D")
        freeze_esm = getattr(config, "freeze_esm", True)
        self.esm_emb = EsmModel.from_pretrained(esm_model_name)
        if freeze_esm:
            for param in self.esm_emb.parameters():
                param.requires_grad = False
        self.time_emb = TimeEmbedding(d_model=self.d_model)

        self.tok_embed_to_hidden = nn.Linear(self.esm_emb.config.hidden_size, self.d_model)

        # --- RoPE shared by attention blocks ---
        rope = RotaryPositionalEmbedding(head_dim=self.d_model // self.n_heads, max_len=self.max_len)

        # --- Encoder ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                proj_dropout=self.proj_dropout,
                rope=rope
            )
            for _ in range(self.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        # --- Heads ---
        # We use small MLP heads for rates; logits are linear.
        self.lam_total_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.logits_type_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 3))
        self.logits_ins_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.logits_sub_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # nonnegativity via softplus (safer than exp)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(
        self,
        x_t: torch.LongTensor,
        mask: torch.BoolTensor,
        t: torch.Tensor,
    ):
        """
        x_t: (B, L) long tokens
        mask: (B, L) bool
        t: (B,) or scalar in [0,1]
        esmbed: optional (B, L, esm2_embed_dim) if use_real_esm2=True
        """
        B, L = x_t.shape
        # pdb.set_trace()

        # --- Embedding ---
        h = self.esm_emb(x_t, mask).last_hidden_state
        h = self.tok_embed_to_hidden(h)

        # print(f"mask: {mask.tolist()}")

        # --- Add time embedding (broadcast across length) ---
        t_emb = self.time_emb(t, batch_size=B)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)  # (B, L, d_model)

        # --- Encoder blocks with key padding mask ---
        for blk in self.blocks:
            h = blk(h, key_padding_mask=(~mask))

        h = self.final_norm(h)  # (B, L, d_model)

        # --- Heads ---
        lam_total = self.softplus(self.lam_total_head(h)).squeeze(-1)  # (B, L)
        logits_type = self.logits_type_head(h)  # (B, L, 3)
        logits_ins = self.logits_ins_head(h)  # (B, L, V)
        logits_sub = self.logits_sub_head(h)  # (B, L, V)

        # --- Compute type probs and per-type rates ---
        pi_type = torch.softmax(logits_type, dim=-1)  # (B, L, 3)
        lam_ins = lam_total * pi_type[..., 0]  # (B, L)
        lam_del = lam_total * pi_type[..., 1]  # (B, L)
        lam_sub = lam_total * pi_type[..., 2]  # (B, L)

        # --- Zero-out padded positions so they contribute nothing downstream ---
        if exists(mask):
            # For lambdas: force to 0 on pads
            pad_mask_f = mask.to(h.dtype)  # True=valid -> 1.0
            lam_total = lam_total * pad_mask_f
            lam_ins = lam_ins * pad_mask_f
            lam_del = lam_del * pad_mask_f
            lam_sub = lam_sub * pad_mask_f

            # kill logits on pads
            neg_val = torch.tensor(-1e4, device=h.device, dtype=h.dtype)
            logits_type = logits_type.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_ins = logits_ins.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_sub = logits_sub.masked_fill((~mask).unsqueeze(-1), neg_val)

        return lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, logits_type, pi_type

class ReparameterizedSMILESEditFlowModel(nn.Module):
    """
    Inputs:
      x_t: (B, L) Long
      mask: (B, L) bool, True=pad (i.e., should be ignored)
      t: (B,) or scalar in [0,1]

    Outputs:
      lam_total: (B, L)         >= 0
      logits_type: (B, L, 3)    over {ins, del, sub}
      logits_ins: (B, L, V)
      logits_sub: (B, L, V)
    """
    def __init__(self, vocab_size, pad_id, config):
        super().__init__()
        self.d_model = getattr(config, "d_model", 768)
        self.n_layers = getattr(config, "n_layers", 12)
        self.n_heads = getattr(config, "n_heads", 12)
        self.mlp_ratio = getattr(config, "mlp_ratio", 4)
        self.max_len = getattr(config, "max_len", 2048)
        self.dropout = getattr(config, "dropout", 0.1)
        self.attn_dropout = getattr(config, "attn_dropout", 0)
        self.proj_dropout = getattr(config, "proj_dropout", 0)
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # --- Embedding ---
        self.seq_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.time_emb = TimeEmbedding(d_model=self.d_model)

        # --- RoPE shared by attention blocks ---
        rope = RotaryPositionalEmbedding(head_dim=self.d_model // self.n_heads, max_len=self.max_len)

        # --- Encoder ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                proj_dropout=self.proj_dropout,
                rope=rope
            )
            for _ in range(self.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        # --- Heads ---
        # We use small MLP heads for rates; logits are linear.
        self.lam_total_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.logits_type_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 3))
        self.logits_ins_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.logits_sub_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # nonnegativity via softplus (safer than exp)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(
        self,
        x_t: torch.LongTensor,
        mask: torch.BoolTensor,
        t: torch.Tensor,
    ):
        """
        x_t: (B, L) long tokens
        mask: (B, L) bool
        t: (B,) or scalar in [0,1]
        """
        B, L = x_t.shape

        # --- Embedding ---
        h = self.seq_emb(x_t)

        # --- Add time embedding (broadcast across length) ---
        t_emb = self.time_emb(t, batch_size=B)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)  # (B, L, d_model)

        # --- Encoder blocks with key padding mask ---
        for blk in self.blocks:
            h = blk(h, key_padding_mask=(~mask))

        h = self.final_norm(h)  # (B, L, d_model)

        # --- Heads ---
        lam_total = self.softplus(self.lam_total_head(h)).squeeze(-1)  # (B, L)
        logits_type = self.logits_type_head(h)  # (B, L, 3)
        logits_ins = self.logits_ins_head(h)  # (B, L, V)
        logits_sub = self.logits_sub_head(h)  # (B, L, V)

        # --- Zero-out padded positions so they contribute nothing downstream ---
        if exists(mask):
            # For lambdas: force to 0 on pads
            pad_mask_f = mask.to(h.dtype)  # True=valid -> 1.0
            lam_total = lam_total * pad_mask_f

            # kill logits on pads
            neg_val = torch.tensor(-1e4, device=h.device, dtype=h.dtype)
            logits_type = logits_type.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_ins = logits_ins.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_sub = logits_sub.masked_fill((~mask).unsqueeze(-1), neg_val)

        return lam_total, logits_type, logits_ins, logits_sub

class SMILESEditFlowModel(nn.Module):
    """
    Inputs:
      x_t: (B, L) Long
      mask: (B, L) bool, True=pad (i.e., should be ignored)
      t: (B,) or scalar in [0,1]

    Outputs:
      lam_ins:  (B, L)         >= 0
      logits_ins: (B, L, V)
      lam_del:  (B, L)         >= 0
      lam_sub:  (B, L)         >= 0
      logits_sub: (B, L, V)
    """
    def __init__(self, vocab_size, pad_id, config):
        super().__init__()
        self.d_model = getattr(config, "d_model", 768)
        self.n_layers = getattr(config, "n_layers", 12)
        self.n_heads = getattr(config, "n_heads", 12)
        self.mlp_ratio = getattr(config, "mlp_ratio", 4)
        self.max_len = getattr(config, "max_len", 2048)
        self.dropout = getattr(config, "dropout", 0.1)
        self.attn_dropout = getattr(config, "attn_dropout", 0)
        self.proj_dropout = getattr(config, "proj_dropout", 0)
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # --- Embedding ---
        self.seq_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.time_emb = TimeEmbedding(d_model=self.d_model)

        # --- RoPE shared by attention blocks ---
        rope = RotaryPositionalEmbedding(head_dim=self.d_model // self.n_heads, max_len=self.max_len)

        # --- Encoder ---
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                proj_dropout=self.proj_dropout,
                rope=rope
            )
            for _ in range(self.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)

        # --- Heads ---
        # We use small MLP heads for rates; logits are linear.
        self.lam_ins_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.lam_del_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.lam_sub_head = nn.Sequential(nn.Linear(self.d_model, self.d_model//2), nn.SiLU(), nn.Linear(self.d_model//2, 1))
        self.logits_ins_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.logits_sub_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # nonnegativity via softplus (safer than exp)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(
        self,
        x_t: torch.LongTensor,
        mask: torch.BoolTensor,
        t: torch.Tensor,
    ):
        """
        x_t: (B, L) long tokens
        mask: (B, L) bool
        t: (B,) or scalar in [0,1]
        esmbed: optional (B, L, esm2_embed_dim) if use_real_esm2=True
        """
        B, L = x_t.shape

        # --- Embedding ---
        h = self.seq_emb(x_t)

        # --- Add time embedding (broadcast across length) ---
        t_emb = self.time_emb(t, batch_size=B)  # (B, d_model)
        h = h + t_emb.unsqueeze(1)  # (B, L, d_model)

        # --- Encoder blocks with key padding mask ---
        for blk in self.blocks:
            h = blk(h, key_padding_mask=(~mask))

        h = self.final_norm(h)  # (B, L, d_model)

        # --- Heads ---
        lam_ins = self.softplus(self.lam_ins_head(h)).squeeze(-1)  # (B, L)
        lam_del = self.softplus(self.lam_del_head(h)).squeeze(-1)  # (B, L)
        lam_sub = self.softplus(self.lam_sub_head(h)).squeeze(-1)  # (B, L)
        logits_ins = self.logits_ins_head(h)  # (B, L, V)
        logits_sub = self.logits_sub_head(h)  # (B, L, V)

        # --- Zero-out padded positions so they contribute nothing downstream ---
        if exists(mask):
            # For lambdas: force to 0 on pads
            pad_mask_f = mask.to(h.dtype)  # True=valid -> 1.0
            lam_ins = lam_ins * pad_mask_f
            lam_del = lam_del * pad_mask_f
            lam_sub = lam_sub * pad_mask_f

            # kill logits on pads
            neg_val = torch.tensor(-1e4, device=h.device, dtype=h.dtype)
            logits_ins = logits_ins.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_sub = logits_sub.masked_fill((~mask).unsqueeze(-1), neg_val)

        return lam_ins, logits_ins, lam_del, lam_sub, logits_sub

class EditFlow(pl.LightningModule):
    def __init__(self, 
                 model,
                 loss_fn,                 
                 path,                     
                 source_distribution,
                 pad_id,
                 bos_id,
                 eos_id,
                 config,
    ):
        super().__init__()

        self.cfg = config

        self.source_distribution = source_distribution
        self.path = path
        self.model = model
        self.loss_fn = loss_fn
        self.loc_prop_path = getattr(config.training, "loc_prop_path", False)

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.eps_id = getattr(self.path, "eps_id", -1)
        self.lam_prop = getattr(self.cfg.training, "lam_prop", 0.0)
        self.use_aux_ce = getattr(self.cfg.training, "use_aux_ce", False)
        self.aux_ce_weight = getattr(self.cfg.training, "aux_ce_weight", 0.0)
        self.reparameterize = getattr(self.cfg.training, "reparameterize", False)
        self.gamma_rate = getattr(self.cfg.training, "gamma_rate", 1)
        self.gamma_edit = getattr(self.cfg.training, "gamma_edit", 1)
        # print(f"use_aux_ce: {self.use_aux_ce}, aux_ce_weight: {self.aux_ce_weight}")

        self._total_steps = None

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.cfg.optim.lr),
            betas=(self.cfg.optim.beta1, self.cfg.optim.beta2),
            eps=float(self.cfg.optim.eps),
            weight_decay=self.cfg.optim.weight_decay,
            fused=self.cfg.optim.fused,
        )

        warmup_ratio = getattr(self.cfg.optim, "warmup_ratio", 0.1)
        min_scale = 0.1

        def lr_lambda(global_step: int):
            # until on_train_start runs we just return 1.0
            if self._total_steps is None or self._total_steps == 0:
                return 1.0

            total_steps = self._total_steps
            warmup_steps = max(1, int(warmup_ratio * total_steps))

            if global_step < warmup_steps:
                # linear warmup: 0.1 -> 1.0
                alpha = (global_step + 1) / warmup_steps
                return 0.1 + 0.9 * alpha
            else:
                # cosine from 1.0 down to min_scale
                progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1 + math.cos(math.pi * progress))   # 1 -> 0
                return min_scale + (1.0 - min_scale) * cosine

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "interval": "step",   # <- per-step
                "frequency": 1,
            },
        }

    def preparation(self, x_1):
        B = x_1.shape[0]

        with torch.no_grad():
            allowed_tokens = torch.tensor([tok for tok in self.source_distribution._allowed_tokens if tok != self.eps_id]).to(self.device)
            # print(f"allowed_tokens: {allowed_tokens}")
            
            x_0 = self.source_distribution.sample_x0_from_x1(x_1, pad_id=self.pad_id, allowed_tokens=allowed_tokens, scale_size=self.cfg.model.scale_size, bos_id = self.bos_id, eos_id = self.eos_id)
            t = torch.rand(B, device=self.device).clamp(max=0.9999)

            # Use lambda_indep if available (for convex schedulers), otherwise fallback to d_alpha_t / sigma_t
            if hasattr(self.path, 'lambda_indep'):
                weight = self.path.lambda_indep(t)  # (B,)
            else:
                sched = self.path.scheduler(t)
                weight = sched.d_alpha_t / sched.sigma_t     # (B,)

            z_0, z_1 = build_z0_z1_with_alignment(x_0, x_1, self.eps_id, self.pad_id, self.bos_id, self.eos_id, p_optimal=self.cfg.model.p_optimal)

            if self.loc_prop_path:
                z_t, M_t, m_t = self.path.sample_localized(
                    z0=z_0, z1=z_1, t=t, lambda_prop=self.lam_prop, return_M=True
                )
            else:
                z_t = self.path.sample(z_0, z_1, t=t)
                M_t = None
                m_t = None
            
            x_t, mask = remove_eps(z_t, self.eps_id, self.pad_id)

        if self.reparameterize:
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, logits_type, pi_type = self.model(x_t=x_t, mask=mask,t=t)
            
            return lam_total, logits_type, logits_ins, logits_sub, z_t, z_1, x_t, mask, weight, M_t

        else:
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub = self.model(x_t=x_t, mask=mask,t=t)

            return lam_ins, logits_ins, lam_del, lam_sub, logits_sub, z_t, z_1, x_t, mask, weight, M_t
    

    def training_step(self, batch, batch_idx):
        x_1 = torch.tensor(batch["input_ids"]).to(self.device)
        B = x_1.shape[0]
        
        if self.reparameterize:
            lam_total, logits_type, logits_ins, logits_sub, z_t, z_1, x_t, mask, weight, M_t = self.preparation(x_1)

            if self.loc_prop_path:
                loss, loss_components = self.loss_fn.reparameterized_forward_localized(lam_total, logits_type, logits_ins, logits_sub, 
                                z_t, z_1, x_t, mask, weight, M_t, self.lam_prop, self.eps_id, self.bos_id, self.eos_id, self.gamma_rate, self.gamma_edit, self.use_aux_ce, self.aux_ce_weight)        
            else:
                loss, loss_components = self.loss_fn.reparameterized_forward(lam_total, logits_type, logits_ins, logits_sub,
                                z_t, z_1, x_t, mask, weight, self.eps_id, self.bos_id, self.eos_id, self.gamma_rate, self.gamma_edit, self.use_aux_ce, self.aux_ce_weight)
            
        else:
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub, z_t, z_1, x_t, mask, weight, M_t = self.preparation(x_1)

            if self.loc_prop_path:
                loss, loss_components = self.loss_fn.forward_localized(lam_ins, logits_ins, lam_del, lam_sub, logits_sub, 
                                    z_t, z_1, x_t, mask, weight, M_t, self.lam_prop, self.eps_id, self.bos_id, self.eos_id, self.use_aux_ce, self.aux_ce_weight)
            else:
                loss, loss_components = self.loss_fn.forward(lam_ins, logits_ins, lam_del, lam_sub, logits_sub, 
                                    z_t, z_1, x_t, mask, weight, self.eps_id, self.bos_id, self.eos_id, self.use_aux_ce, self.aux_ce_weight)
        
        # Log total loss (same metric name for both reparameterized and non-reparameterized models)
        # This ensures both model types appear on the same WandB chart for comparison
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        # Log full non-aux loss
        # Log individual non-aux loss components
        self.log("train_loss_rate", loss_components["loss_rate"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        self.log("train_loss_edit", loss_components["loss_edit"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        # Log unweighted and weighted aux losses (only if use_aux_ce is True)
        if self.use_aux_ce and self.aux_ce_weight > 0.0:
            self.log("train_loss_base", loss_components["loss_base"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("train_loss_aux_unweighted", loss_components["loss_aux_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("train_loss_aux_weighted", loss_components["loss_aux_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        # Log new weighted/unweighted components (only available for reparameterized methods)
        if "loss_rate_unweighted" in loss_components:
            self.log("train_loss_rate_unweighted", loss_components["loss_rate_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("train_loss_rate_weighted", loss_components["loss_rate_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("train_loss_edit_unweighted", loss_components["loss_edit_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("train_loss_edit_weighted", loss_components["loss_edit_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("train_loss_total_weighted", loss_components["loss_total_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("train_loss_total_unweighted", loss_components["loss_total_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_1 = torch.tensor(batch["input_ids"]).to(self.device)
        B = x_1.shape[0]
        
        if self.reparameterize:
            lam_total, logits_type, logits_ins, logits_sub, z_t, z_1, x_t, mask, weight, M_t = self.preparation(x_1)
            
            if self.loc_prop_path:
                loss, loss_components = self.loss_fn.reparameterized_forward_localized(lam_total, logits_type, logits_ins, logits_sub, 
                                z_t, z_1, x_t, mask, weight, M_t, self.lam_prop, self.eps_id, self.bos_id, self.eos_id, self.gamma_rate, self.gamma_edit, self.use_aux_ce, self.aux_ce_weight)        
            else:
                loss, loss_components = self.loss_fn.reparameterized_forward(lam_total, logits_type, logits_ins, logits_sub,
                                z_t, z_1, x_t, mask, weight, self.eps_id, self.bos_id, self.eos_id, self.gamma_rate, self.gamma_edit, self.use_aux_ce, self.aux_ce_weight)
        else:
            lam_ins, logits_ins, lam_del, lam_sub, logits_sub, z_t, z_1, x_t, mask, weight, M_t = self.preparation(x_1)
            
            if self.loc_prop_path:
                loss, loss_components = self.loss_fn.forward_localized(lam_ins, logits_ins, lam_del, lam_sub, logits_sub, 
                                    z_t, z_1, x_t, mask, weight, M_t, self.lam_prop, self.eps_id, self.bos_id, self.eos_id, self.use_aux_ce, self.aux_ce_weight)
            else:
                loss, loss_components = self.loss_fn.forward(lam_ins, logits_ins, lam_del, lam_sub, logits_sub, 
                                    z_t, z_1, x_t, mask, weight, self.eps_id, self.bos_id, self.eos_id, self.use_aux_ce, self.aux_ce_weight)
        
        # Log total loss (same metric name for both reparameterized and non-reparameterized models)
        # This ensures both model types appear on the same WandB chart for comparison
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        # Log full non-aux loss
        # Log individual non-aux loss components
        self.log("val_loss_rate", loss_components["loss_rate"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        self.log("val_loss_edit", loss_components["loss_edit"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        # Log unweighted and weighted aux losses (only if use_aux_ce is True)
        if self.use_aux_ce and self.aux_ce_weight > 0.0:
            self.log("val_loss_base", loss_components["loss_base"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("val_loss_aux_unweighted", loss_components["loss_aux_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("val_loss_aux_weighted", loss_components["loss_aux_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
        # Log new weighted/unweighted components (only available for reparameterized methods)
        if "loss_rate_unweighted" in loss_components:
            self.log("val_loss_rate_unweighted", loss_components["loss_rate_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("val_loss_rate_weighted", loss_components["loss_rate_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("val_loss_edit_unweighted", loss_components["loss_edit_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("val_loss_edit_weighted", loss_components["loss_edit_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("val_loss_total_weighted", loss_components["loss_total_weighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)
            self.log("val_loss_total_unweighted", loss_components["loss_total_unweighted"], prog_bar=False, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)

        return loss
    
    def on_train_start(self):
        # how many optimizer steps we will take in this fit
        self._total_steps = self.trainer.estimated_stepping_batches

    def on_after_backward(self):
        """
        Hook called after backward() but before gradient clipping.
        Used to log gradient information before clipping occurs.
        """
        # Get gradient clipping value from trainer (default is 1.0)
        clip_val = getattr(self.trainer, 'gradient_clip_val', 1.0)
        if clip_val is None:
            clip_val = float('inf')
        
        # Collect all parameter gradients
        total_grad_norm = 0.0
        param_grad_norms = {}
        num_clipped = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                # Compute gradient norm for this parameter
                grad_norm = param.grad.data.norm(2).item()
                param_grad_norms[name] = grad_norm
                total_grad_norm += grad_norm ** 2
                total_params += 1
                
                # Check if this gradient would be clipped
                if grad_norm > clip_val:
                    num_clipped += 1
        
        # Compute total gradient norm (L2 norm across all parameters)
        total_grad_norm = total_grad_norm ** 0.5
        
        # Log total gradient norm
        self.log("train_grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log ratio of clipped gradients
        if total_params > 0:
            clip_ratio = num_clipped / total_params
            self.log("train_grad_clip_ratio", clip_ratio, on_step=True, on_epoch=False, sync_dist=True)
        else:
            self.log("train_grad_clip_ratio", 0.0, on_step=True, on_epoch=False, sync_dist=True)
        
        # Log individual parameter gradient norms (as a dictionary/table in wandb)
        # We'll log the max, min, mean, and std of parameter gradients
        if param_grad_norms:
            grad_norms_list = list(param_grad_norms.values())
            self.log("train_grad_norm_max", max(grad_norms_list), on_step=True, on_epoch=False, sync_dist=True)
            self.log("train_grad_norm_min", min(grad_norms_list), on_step=True, on_epoch=False, sync_dist=True)
            self.log("train_grad_norm_mean", sum(grad_norms_list) / len(grad_norms_list), on_step=True, on_epoch=False, sync_dist=True)
            
            # Compute std
            mean_grad = sum(grad_norms_list) / len(grad_norms_list)
            variance = sum((x - mean_grad) ** 2 for x in grad_norms_list) / len(grad_norms_list)
            std_grad = variance ** 0.5
            self.log("train_grad_norm_std", std_grad, on_step=True, on_epoch=False, sync_dist=True)
            
            # Log full parameter gradients as a wandb table (every N steps to avoid too much data)
            # For now, we'll log a summary. To log full details, we can use wandb.Table
            if hasattr(self.logger, 'experiment') and hasattr(self.logger.experiment, 'log'):
                # Log detailed gradient info every 100 steps
                if self.global_step % 100 == 0:
                    import wandb
                    # Create a table with parameter names and their gradient norms
                    table_data = [[name, norm] for name, norm in param_grad_norms.items()]
                    table = wandb.Table(columns=["parameter", "grad_norm"], data=table_data)
                    self.logger.experiment.log({"train_grad_norms_table": table}, step=self.global_step)

    # 2

                