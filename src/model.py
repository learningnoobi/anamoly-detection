

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al. 2017).
    Handles both even and odd d_model correctly.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()                # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                                   # (d_model//2,)

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])                 # slice for odd d_model

        self.register_buffer('pe', pe.unsqueeze(0))                         # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────

class NetworkTransformer(nn.Module):

    def __init__(self,
                 num_numeric: int,            # count of numeric + boolean columns
                 cat_indices: list,           # positional indices of categorical columns in the input tensor
                 cat_vocab_sizes: list,       # number of unique values per categorical (same order as cat_indices)
                 num_classes: int  = 10,
                 d_model:     int  = 128,
                 nhead:       int  = 8,
                 num_layers:  int  = 3,
                 dim_feedforward: int = 512,
                 cat_embed_dim:   int = 16,   # embedding dimension per categorical feature
                 dropout:     float = 0.1):
        super().__init__()

        self.cat_indices = cat_indices        # saved so forward() knows which columns to embed

        # ── categorical embeddings (one per categorical feature) ──
        # +1 on vocab_size as a safety margin in case factorize produces an
        # unseen code at test time (shouldn't happen, but costs nothing).
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs + 1, cat_embed_dim)
            for vs in cat_vocab_sizes
        ])

        # ── input projection ──
        # Input to the linear layer = num_numeric raw values
        #                            + (num_categoricals × cat_embed_dim) embedded values
        proj_input_dim = num_numeric + cat_embed_dim * len(cat_indices)
        self.input_proj = nn.Linear(proj_input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # ── CLS token (one learnable vector, broadcast across batch) ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── positional encoding ──
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # ── transformer encoder ──
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-LN: stable training from scratch
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        # ── classification head ──
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    # ── forward ───────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, total_features)
            Mixed tensor from the DataLoader.  Categorical columns sit at
            self.cat_indices; everything else is numeric / boolean.

        Returns
        -------
        logits : (batch, num_classes)
            Raw class scores.  Do NOT softmax before passing to
            nn.CrossEntropyLoss — it applies log-softmax internally.
        """
        B, S, F = x.shape

        # 1. split numeric vs categorical
        num_idx = [i for i in range(F) if i not in self.cat_indices]
        x_num   = x[:, :, num_idx]                           # (B, S, num_numeric)

        # 2. embed each categorical column separately, then concatenate
        embeds = []
        for i, col_idx in enumerate(self.cat_indices):
            embeds.append(
                self.embeddings[i](x[:, :, col_idx].long())  # (B, S, cat_embed_dim)
            )
        x_cat = torch.cat(embeds, dim=-1)                    # (B, S, num_cat * embed_dim)

        # 3. project combined input to d_model
        h = self.input_norm(
            self.input_proj(torch.cat([x_num, x_cat], dim=-1))
        )                                                    # (B, S, d_model)

        # 4. prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, d_model)
        h   = torch.cat([cls, h], dim=1)                     # (B, S+1, d_model)

        # 5. add positional encoding
        h = self.pos_enc(h)

        # 6. transformer encoder
        h = self.encoder(h)                                  # (B, S+1, d_model)

        # 7. extract CLS token → classify
        return self.head(h[:, 0])                            # (B, num_classes)


# ─────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate the feature layout produced by load_data + preprocess:
    #   16 numeric | 3 categorical (indices 16,17,18) | 7 boolean
    #   total = 26 features

    B, S, F        = 8, 10, 26
    cat_indices     = [16, 17, 18]
    cat_vocab_sizes = [5, 12, 11]          # example vocab sizes
    num_numeric     = F - len(cat_indices) # 23

    model = NetworkTransformer(
        num_numeric=num_numeric,
        cat_indices=cat_indices,
        cat_vocab_sizes=cat_vocab_sizes,
        num_classes=10,
        d_model=128,
        nhead=8,
        num_layers=3,
    )

    # Build dummy input: numeric columns are random floats,
    # categorical columns are random integers within vocab range.
    x = torch.randn(B, S, F)
    for i, ci in enumerate(cat_indices):
        x[:, :, ci] = torch.randint(0, cat_vocab_sizes[i], (B, S)).float()

    logits = model(x)

    print(f"Input  shape : {tuple(x.shape)}")
    print(f"Logits shape : {tuple(logits.shape)}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Logits[0]    : {logits[0].detach().round(decimals=3)}")