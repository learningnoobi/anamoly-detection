"""
Transformer-based anomaly detection model with Conditional VAE.

Architecture:
1. Transformer Encoder: Learn temporal patterns in network traffic sequences
2. Conditional VAE (C-VAE): Learn latent representations conditioned on sequence
3. Reconstruction: Detect anomalies via reconstruction error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling."""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.d_model = d_model
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            encoded: (batch_size, seq_len, d_model)
        """
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        return encoded


class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder for latent representation learning."""
    
    def __init__(self, input_dim: int, latent_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            recon: Reconstruction
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class TransformerCVAE(nn.Module):
    """
    Complete model: Transformer Encoder + Conditional VAE for anomaly detection.
    
    The model processes sequences through:
    1. Transformer encoder to capture temporal dependencies
    2. Aggregation of sequence representation (mean pooling)
    3. C-VAE for reconstruction-based anomaly detection
    """
    
    def __init__(self, input_dim: int, seq_len: int,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 3,
                 latent_dim: int = 64, vae_hidden_dim: int = 256):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        # Transformer encoder for sequence modeling
        self.transformer = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )
        
        # C-VAE for latent representation and reconstruction
        self.cvae = ConditionalVAE(
            input_dim=d_model,
            latent_dim=latent_dim,
            hidden_dim=vae_hidden_dim
        )
        
        # Projection back to original space for reconstruction
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            recon_seq: Reconstructed sequence
            mu: Latent mean
            logvar: Latent log variance
        """
        batch_size = x.size(0)
        
        # Encode sequences with transformer
        encoded = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Aggregate sequence representation (mean pooling)
        seq_repr = encoded.mean(dim=1)  # (batch, d_model)
        
        # VAE reconstruction
        recon_repr, mu, logvar = self.cvae(seq_repr)
        
        # Project back to original dimension
        recon_seq = self.output_projection(recon_repr)  # (batch, input_dim)
        
        # Expand to sequence for comparison
        recon_seq = recon_seq.unsqueeze(1).expand(-1, self.seq_len, -1)
        
        return recon_seq, mu, logvar
    
    def get_reconstruction_error(self, x):
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            x: Input sequences (batch_size, seq_len, input_dim)
        Returns:
            errors: Reconstruction error per sample (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            recon_seq, _, _ = self.forward(x)
            # MSE per sequence
            errors = F.mse_loss(recon_seq, x, reduction='none').mean(dim=(1, 2))
        return errors


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction loss + KL divergence.
    
    Args:
        recon_x: Reconstructed sequences
        x: Original sequences
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence (beta-VAE)
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kld, recon_loss, kld


if __name__ == "__main__":
    # Test model
    batch_size = 16
    seq_len = 10
    input_dim = 21
    
    # Create dummy data
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Initialize model
    model = TransformerCVAE(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=128,
        nhead=8,
        num_layers=3,
        latent_dim=64
    )
    
    print(f"Model initialized")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    recon, mu, logvar = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Compute loss
    loss, recon_loss, kld = vae_loss(recon, x, mu, logvar)
    print(f"\nTotal loss: {loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL divergence: {kld.item():.4f}")
    
    # Test reconstruction error
    errors = model.get_reconstruction_error(x)
    print(f"\nReconstruction errors shape: {errors.shape}")
    print(f"Mean error: {errors.mean().item():.4f}")