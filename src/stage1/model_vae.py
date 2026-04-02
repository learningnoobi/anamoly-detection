"""
model_vae.py — Vanilla MLP VAE
================================
Adds probabilistic bottleneck over the plain AE.
Still MLP-based, still no conditioning, no Transformer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import (
    load_data,
    RowDataset,
    evaluate_model,
    BATCH_SIZE,
    SEQ_LEN,
)
from config import *

prep_file = "vae_prep.pkl"
model_file = "vae_model.pt"


# MODEL
class VanillaVAE(nn.Module):
    def __init__(self, n_features, d_hidden=D_HIDDEN,
                 d_z=D_Z, dropout=DROPOUT):
        super().__init__()
        self.n_features = n_features
        # flat_dim = n_features only, no seq_len
        
        self.encoder_body = nn.Sequential(
            nn.Linear(n_features, d_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fc_mu     = nn.Linear(d_hidden // 2, d_z)
        self.fc_logvar = nn.Linear(d_hidden // 2, d_z)

        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_hidden // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, d_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_features)
        )

    def encode(self, x):
        # x: (batch, n_features) — single row
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        return self.decoder(z)   # (batch, n_features)

    def forward(self, x, device_id=None):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    def anomaly_score(self, x, device_id=None):
        self.eval()
        with torch.no_grad():
            x_hat, _, _ = self.forward(x)
            score = ((x - x_hat) ** 2).mean(dim=1)  # mean over features
        return score.cpu().numpy()

# LOSS
def elbo_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl, recon.item(), kl.item()


# TRAINING
def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss, patience_ctr, best_weights = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        beta = min(1.0, epoch / BETA_WARMUP)

        model.train()
        total_loss = total_recon = total_kl = 0
        for x in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = elbo_loss(x, x_hat, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon
            total_kl += kl
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(DEVICE)
                x_hat, mu, logvar = model(x)
                loss, _, _ = elbo_loss(x, x_hat, mu, logvar, beta=1.0)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        n = len(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Loss: {total_loss/n:.4f} "
                f"(Recon: {total_recon/n:.4f}, KL: {total_kl/n:.4f}) | "
                f"Val: {val_loss:.4f} | beta: {beta:.2f}"
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return model


def evaluate():
    with open(prep_file, "rb") as f:
        prep = pickle.load(f)
    model = VanillaVAE(n_features=27).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    evaluate_model(
        model=model,
        prep=prep,
        model_name="VanillaVAE",
        test_path="../data/train_test_network.csv",
    )


# MAIN
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    X_train, X_val, n_features, prep = load_data()

    train_loader = DataLoader(RowDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RowDataset(X_val), batch_size=BATCH_SIZE)

    model = VanillaVAE(n_features).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    print("--- Training Vanilla VAE ---")
    model = train(model, train_loader, val_loader)

    torch.save(model.state_dict(), model_file)
    import pickle

    with open(prep_file, "wb") as f:
        pickle.dump(prep, f)

    print("\nSaved → " + model_file + ", " + prep_file)

    # Evaluate using shared function from utils
    evaluate_model(model, prep, model_name="VAE")
