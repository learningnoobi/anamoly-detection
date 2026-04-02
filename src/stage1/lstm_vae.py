"""
model_lstm_vae.py — LSTM Variational Autoencoder (LSTM-VAE)
============================================================
Adds temporal sequence modeling over vanilla VAE.
Bidirectional LSTM encoder + LSTM decoder.
Input: (batch, seq_len, n_features) — sequences of 10 consecutive flows.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import (load_data_sequences, SequenceDataset, evaluate_model_sequences,
                   BATCH_SIZE, SEQ_LEN)

# CONFIG
D_HIDDEN    = 128
D_Z         = 64
N_LAYERS    = 2
DROPOUT     = 0.1
EPOCHS      = 100
LR          = 1e-4
PATIENCE    = 10
BETA_WARMUP = 20
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prep_file = "lstm_vae_prep.pkl"
model_file = "lstm_vae_model.pt"

# MODEL
class LSTMVAEModel(nn.Module):
    def __init__(self, n_features, d_hidden=D_HIDDEN,
                 d_z=D_Z, n_layers=N_LAYERS,
                 dropout=DROPOUT, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.d_hidden   = d_hidden
        self.n_layers   = n_layers

        # Bidirectional LSTM encoder
        # output dim = d_hidden * 2 (forward + backward)
        self.encoder = nn.LSTM(
            input_size  = n_features,
            hidden_size = d_hidden // 2,
            num_layers  = n_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout if n_layers > 1 else 0.0
        )

        # VAE bottleneck
        self.fc_mu     = nn.Linear(d_hidden, d_z)
        self.fc_logvar = nn.Linear(d_hidden, d_z)

        # Decoder — unidirectional LSTM
        self.z_up = nn.Linear(d_z, d_hidden)
        self.decoder = nn.LSTM(
            input_size  = d_hidden,
            hidden_size = d_hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout = dropout if n_layers > 1 else 0.0
        )
        self.output_proj = nn.Linear(d_hidden, n_features)

    def encode(self, x):
        # x: (B, T, n_features)
        _, (h_n, _) = self.encoder(x)
        # h_n: (n_layers*2, B, d_hidden//2)
        # take last layer forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, d_hidden)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z):
        B  = z.shape[0]
        h  = self.z_up(z)                                    # (B, d_hidden)
        # repeat latent across time steps as decoder input
        h_seq   = h.unsqueeze(1).expand(-1, self.seq_len, -1) # (B, T, d_hidden)
        out, _  = self.decoder(h_seq)                         # (B, T, d_hidden)
        return self.output_proj(out)                          # (B, T, n_features)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    def anomaly_score(self, x, beta=1.0):
        self.eval()
        with torch.no_grad():
            x_hat, mu, logvar = self.forward(x)

            # Reconstruction error
            recon = ((x - x_hat) ** 2).mean(dim=[1, 2])

            # KL divergence
            kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=[1, 2]
            )

            score = recon + beta * kl

        return score.cpu().numpy()


# LOSS
def elbo_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_hat, x, reduction='mean')
    kl    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon + beta * kl, recon.item(), kl.item()


# TRAINING
def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS)

    best_loss, patience_ctr, best_weights = float('inf'), 0, None

    for epoch in range(1, EPOCHS + 1):
        beta = min(1.0, epoch / BETA_WARMUP)

        model.train()
        t_loss = t_recon = t_kl = 0
        for (x,) in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            x_hat, mu, logvar    = model(x)
            loss, recon, kl      = elbo_loss(x, x_hat, mu, logvar, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss  += loss.item()
            t_recon += recon
            t_kl    += kl
        scheduler.step()

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for (x,) in val_loader:
                x = x.to(DEVICE)
                x_hat, mu, logvar = model(x)
                loss, _, _ = elbo_loss(x, x_hat, mu, logvar, beta=1.0)
                v_loss += loss.item()
        v_loss /= len(val_loader)

        n = len(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"Loss {t_loss/n:.4f} "
                  f"(Recon {t_recon/n:.4f}  KL {t_kl/n:.4f}) | "
                  f"Val {v_loss:.4f} | beta {beta:.2f}")

        if v_loss < best_loss:
            best_loss    = v_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return model

import pickle
def evaluate(n_features=27):
    with open(prep_file, "rb") as f:
        prep = pickle.load(f)
    model = LSTMVAEModel(n_features=n_features).to(DEVICE)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    evaluate_model_sequences(
        model=model,
        prep=prep,
        model_name="LSTMVAEModel",
        test_path="../data/all_network_data.csv",
    )


# MAIN
if __name__ == '__main__':
    # print(f"Device: {DEVICE}")

    # X_train, X_val, n_features, prep = load_data_sequences()

    # train_loader = DataLoader(
    #     TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
    #     batch_size=BATCH_SIZE, shuffle=True
    # )
    # val_loader = DataLoader(
    #     TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
    #     batch_size=BATCH_SIZE
    # )

    # model = LSTMVAEModel(n_features).to(DEVICE)
    # print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # print("--- Training LSTM-VAE ---")
    # model = train(model, train_loader, val_loader)

    # torch.save(model.state_dict(), 'lstm_vae_model.pt')
    # import pickle
    # with open('lstm_vae_prep.pkl', 'wb') as f:
    #     pickle.dump(prep, f)
    # print("\nSaved → lstm_vae_model.pt, lstm_vae_prep.pkl")

    evaluate()