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
        mask: (B, L) bool, True=valid (i.e., should be kept)
        t: (B,) or scalar in [0,1]
        """
        B, L = x_t.shape

        # --- Embedding ---
        h = self.esm_emb(x_t, mask).last_hidden_state  # (B, L, esm_hidden)
        h = self.tok_embed_to_hidden(h)                # (B, L, d_model)

        # --- Add time embedding (broadcast across length) ---
        t_emb = self.time_emb(t, batch_size=B)         # (B, d_model)
        h = h + t_emb.unsqueeze(1)                     # (B, L, d_model)

        # --- Encoder blocks with key padding mask ---
        for blk in self.blocks:
            h = blk(h, key_padding_mask=(~mask))

        h = self.final_norm(h)                         # (B, L, d_model)

        # --- Heads ---
        lam_total = self.softplus(self.lam_total_head(h)).squeeze(-1)  # (B, L) >= 0
        logits_type = self.logits_type_head(h)                         # (B, L, 3)
        logits_ins = self.logits_ins_head(h)                           # (B, L, V)
        logits_sub = self.logits_sub_head(h)                           # (B, L, V)

        # --- Compute type probs and per-type rates ---
        pi_type = torch.softmax(logits_type, dim=-1)                   # (B, L, 3)
        lam_ins = lam_total * pi_type[..., 0]                          # (B, L)
        lam_del = lam_total * pi_type[..., 1]                          # (B, L)
        lam_sub = lam_total * pi_type[..., 2]                          # (B, L)

        # --- Zero-out padded positions so they contribute nothing downstream ---
        if exists(mask):
            pad_mask_f = mask.to(h.dtype)  # True=valid -> 1.0
            lam_total = lam_total * pad_mask_f
            lam_ins   = lam_ins   * pad_mask_f
            lam_del   = lam_del   * pad_mask_f
            lam_sub   = lam_sub   * pad_mask_f

            neg_val = torch.tensor(-1e4, device=h.device, dtype=h.dtype)
            logits_type = logits_type.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_ins  = logits_ins.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_sub  = logits_sub.masked_fill((~mask).unsqueeze(-1), neg_val)

        return lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type



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

        # --- Heads (REPARAM) ---
        # total rate λ_total and type distribution π_type over {ins, del, sub}
        self.lam_total_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 1),
        )
        self.type_logits_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, 3),   # 3 edit types
        )

        # token logits (unchanged)
        self.logits_ins_head = nn.Linear(self.d_model, vocab_size, bias=False)
        self.logits_sub_head = nn.Linear(self.d_model, vocab_size, bias=False)

        self.softplus = nn.Softplus(beta=1.0)

    def forward(
        self,
        x_t: torch.LongTensor,
        mask: torch.BoolTensor,
        t: torch.Tensor,
    ):
        """
        x_t: (B, L) long tokens
        mask: (B, L) bool, True = PAD (ignored)
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

        # --- Heads (REPARAM) ---
        lam_total = self.softplus(self.lam_total_head(h)).squeeze(-1)  # (B, L), >=0
        type_logits = self.type_logits_head(h)                         # (B, L, 3)
        pi_type = torch.softmax(type_logits, dim=-1)                   # (B, L, 3), sums to 1

        lam_ins = lam_total * pi_type[..., 0]                          # (B, L)
        lam_del = lam_total * pi_type[..., 1]
        lam_sub = lam_total * pi_type[..., 2]

        logits_ins = self.logits_ins_head(h)                           # (B, L, V)
        logits_sub = self.logits_sub_head(h)                           # (B, L, V)

        # --- Masking (keep your existing behavior) ---
        if exists(mask):
            pad_mask_f = mask.to(h.dtype)
            lam_total = lam_total * pad_mask_f
            lam_ins = lam_ins * pad_mask_f
            lam_del = lam_del * pad_mask_f
            lam_sub = lam_sub * pad_mask_f

            neg_val = torch.tensor(-1e4, device=h.device, dtype=h.dtype)
            logits_ins = logits_ins.masked_fill((~mask).unsqueeze(-1), neg_val)
            logits_sub = logits_sub.masked_fill((~mask).unsqueeze(-1), neg_val)

        return lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type

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

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.eps_id = getattr(self.path, "eps_id", -1)

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
            
            x_0 = self.source_distribution.sample_x0_from_x1(x_1, pad_id=self.pad_id, allowed_tokens=allowed_tokens, scale_size=self.cfg.model.scale_size, bos_id = self.bos_id, eos_id = self.eos_id)
            t = torch.rand(B, device=self.device)

            sched = self.path.scheduler(t)
            weight = sched.d_alpha_t / sched.sigma_t     # (B,)

            z_0, z_1 = build_z0_z1_with_alignment(x_0, x_1, self.eps_id, self.pad_id, self.bos_id, self.eos_id, p_optimal=self.cfg.model.p_optimal)

            z_t = self.path.sample(z_0, z_1, t=t)
            x_t, mask = remove_eps(z_t, self.eps_id, self.pad_id)

        lam_ins, logits_ins, lam_del, lam_sub, logits_sub, lam_total, pi_type = self.model(x_t=x_t, mask=mask,t=t)

        return lam_total, pi_type, logits_ins, logits_sub, z_t, z_1, x_t, mask, weight
    

    def training_step(self, batch, batch_idx):
        x_1 = torch.tensor(batch["input_ids"]).to(self.device)
        B = x_1.shape[0]
        
        lam_total, pi_type, logits_ins, logits_sub, z_t, z_1, x_t, mask, weight = self.preparation(x_1)
        
        loss = self.loss_fn(lam_total, pi_type, logits_ins, logits_sub, 
                            z_t, z_1, x_t, mask, weight, self.eps_id, self.bos_id, self.eos_id)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_1 = torch.tensor(batch["input_ids"]).to(self.device)
        B = x_1.shape[0]
        
        lam_total, pi_type, logits_ins, logits_sub, z_t, z_1, x_t, mask, weight = self.preparation(x_1)
        
        loss = self.loss_fn(lam_total, pi_type, logits_ins, logits_sub, 
                            z_t, z_1, x_t, mask, weight, self.eps_id, self.bos_id, self.eos_id)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x_1 = torch.tensor(batch["input_ids"]).to(self.device)
        B = x_1.shape[0]
        
        lam_total, pi_type, logits_ins, logits_sub, z_t, z_1, x_t, mask, weight = self.preparation(x_1)
        
        loss = self.loss_fn(lam_total, pi_type, logits_ins, logits_sub, 
                            z_t, z_1, x_t, mask, weight, self.eps_id, self.bos_id, self.eos_id)
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B, sync_dist=True)

        return loss
    def on_train_start(self):
        # how many optimizer steps we will take in this fit
        self._total_steps = self.trainer.estimated_stepping_batches