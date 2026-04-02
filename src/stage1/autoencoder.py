"""
model_ae.py — Vanilla MLP Autoencoder (AE)
===========================================
Simplest baseline: MLP encoder → bottleneck → MLP decoder.
No VAE, no conditioning, no Transformer.
Anomaly score = MSE reconstruction error.
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_data, RowDataset, BATCH_SIZE, evaluate_model
from load_data import ATTACK_TYPES
from config import *

NORMAL_IDX = ATTACK_TYPES.index("normal")  # 0
prep_file = "ae_prep.pkl"
model_file = "ae_model.pt"


# MODEL
class VanillaAE(nn.Module):
    def __init__(self, n_features, d_hidden=D_HIDDEN, d_z=D_Z, dropout=DROPOUT):
        super().__init__()
        self.n_features = n_features

        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, d_z),
        )

        self.decoder = nn.Sequential(
            nn.Linear(d_z, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def anomaly_score(self, x):
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(x)
            # MSE per sample across features
            score = ((x - x_hat) ** 2).mean(dim=1)
        return score.cpu().numpy()


# TRAINING
def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss, patience_ctr, best_weights = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        for x in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x in val_loader:
                val_loss += F.mse_loss(model(x.to(DEVICE)), x.to(DEVICE)).item()
        val_loss /= len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Train: {total/len(train_loader):.4f} | "
                f"Val: {val_loss:.4f}"
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
    model = VanillaAE(n_features=27).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    evaluate_model(
        model=model,
        prep=prep,
        model_name="VanillaAE",
        test_path="../data/train_test_network.csv",
    )


# MAIN
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    X_train, X_val, n_features, prep = load_data()

    train_loader = DataLoader(RowDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RowDataset(X_val), batch_size=BATCH_SIZE)

    model = VanillaAE(n_features).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    print("--- Training Vanilla AE ---")
    model = train(model, train_loader, val_loader)

    # Save model and preprocessor together
    torch.save(model.state_dict(), model_file)
    import pickle

    with open(prep_file, "wb") as f:
        pickle.dump(prep, f)

    print("\nSaved → " + model_file + ", " + prep_file)
    print("\n running evaluate with test data to get paper metrics")
    evaluate()
