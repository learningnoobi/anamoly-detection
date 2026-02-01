import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from .model import NetworkTransformer
from .load_data import ATTACK_TYPES, NUM_CLASSES


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience    = patience
        self.min_delta   = min_delta
        self.counter     = 0
        self.best_loss   = None
        self.should_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            self.should_stop = (self.counter >= self.patience)
        else:
            self.best_loss = val_loss
            self.counter   = 0


class Trainer:
    """
    Training loop, validation, test evaluation, and plotting.

    Constructor takes class_weights (the tensor that prepare_data() returns)
    and bakes it directly into CrossEntropyLoss.  This is the single place
    the MITM imbalance is corrected — no changes needed anywhere else.
    """

    def __init__(self,
                 model: NetworkTransformer,
                 class_weights: torch.Tensor,  
                 device: str,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4):
        self.model     = model.to(device)
        self.device    = device
        self.lr        = lr

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        # Scheduler is created inside train() once we know steps_per_epoch
        self.scheduler = None

        self.history: Dict[str, list] = {
            'train_loss': [], 'train_acc': [],
            'val_loss':   [], 'val_acc':   [],
        }

    def _forward(self, bx, by):
        """One forward pass → (loss, logits)."""
        logits = self.model(bx)
        loss   = self.criterion(logits, by)
        return loss, logits

    @staticmethod
    def _acc(logits: torch.Tensor, y: torch.Tensor) -> float:
        return (logits.argmax(dim=1) == y).float().mean().item()

    # ── train / val ───────────────────────────────────────────

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        tot_loss, tot_acc, n = 0., 0., 0

        for bx, by in loader:
            bx, by = bx.to(self.device), by.to(self.device)

            loss, logits = self._forward(bx, by)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()       # OneCycleLR steps per *batch*

            bs = bx.size(0)
            tot_loss += loss.item() * bs
            tot_acc  += self._acc(logits, by) * bs
            n        += bs

        return {'loss': tot_loss / n, 'acc': tot_acc / n}

    @torch.no_grad()
    def validate(self, loader) -> Dict[str, float]:
        self.model.eval()
        tot_loss, tot_acc, n = 0., 0., 0

        for bx, by in loader:
            bx, by = bx.to(self.device), by.to(self.device)
            loss, logits = self._forward(bx, by)

            bs = bx.size(0)
            tot_loss += loss.item() * bs
            tot_acc  += self._acc(logits, by) * bs
            n        += bs

        return {'loss': tot_loss / n, 'acc': tot_acc / n}

    # ── full test evaluation ──────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader):
        """
        Run on the test split after loading the best checkpoint.

        Returns
        -------
        report : str   — sklearn classification_report (precision / recall / F1 per class)
        cm     : ndarray — confusion matrix (NUM_CLASSES × NUM_CLASSES)
        """
        self.model.eval()
        all_preds, all_labels = [], []

        for bx, by in loader:
            logits = self.model(bx.to(self.device))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(by.numpy())

        preds  = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        report = classification_report(
            labels, preds, target_names=ATTACK_TYPES, zero_division=0
        )
        cm = confusion_matrix(labels, preds, labels=range(NUM_CLASSES))

        return report, cm

    # ── main training loop ────────────────────────────────────

    def train(self, train_loader, val_loader,
              num_epochs: int = 100,
              early_stop_patience: int = 15,
              save_dir: str = './checkpoints'):
        """
        Full training loop.

        Scheduler: OneCycleLR
            Warmup for the first 10 % of total steps, then cosine decay to
            eta_min=1e-5.  Transformers are sensitive to the initial LR — the
            warmup phase lets the attention weights settle before the full LR
            kicks in.  This replaces the old ReduceLROnPlateau, which has no
            warmup and reacts too slowly to training dynamics.

        Checkpointing: saves best model by *validation accuracy* (not loss).
            For an imbalanced 10-class problem, accuracy is a more direct
            objective than loss, because CrossEntropyLoss with class weights
            can decrease even when per-class recall for rare classes worsens.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # OneCycleLR needs to know total steps up front
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.1,              # 10 % warmup
            anneal_strategy='cos',
        )

        es           = EarlyStopping(patience=early_stop_patience)
        best_val_acc = 0.0

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n{'='*58}")
        print(f"  Training  |  max_epochs={num_epochs}  |  device={self.device}")
        print(f"  parameters={n_params:,}  |  batches/epoch={len(train_loader)}")
        print(f"{'='*58}\n")

        for epoch in range(1, num_epochs + 1):
            tr = self.train_epoch(train_loader)
            va = self.validate(val_loader)

            self.history['train_loss'].append(tr['loss'])
            self.history['train_acc'].append(tr['acc'])
            self.history['val_loss'].append(va['loss'])
            self.history['val_acc'].append(va['acc'])

            marker = ''
            if va['acc'] > best_val_acc:
                best_val_acc = va['acc']
                torch.save(self.model.state_dict(), save_path / 'best_model.pt')
                marker = '  ← best'

            print(f"  Ep {epoch:>3d} | "
                  f"train  loss={tr['loss']:.4f}  acc={tr['acc']:.4f} | "
                  f"val    loss={va['loss']:.4f}  acc={va['acc']:.4f}{marker}")

            es(va['loss'])
            if es.should_stop:
                print(f"\n  Early stopping triggered at epoch {epoch}")
                break

        # persist history for later plotting
        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  Best val accuracy: {best_val_acc:.4f}")
        return self.history

    # ── plotting ──────────────────────────────────────────────

    def plot_history(self, save_path: str = './checkpoints/training_curves.png'):
        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(epochs, self.history['train_loss'], label='Train')
        ax1.plot(epochs, self.history['val_loss'],   label='Val')
        ax1.set(xlabel='Epoch', ylabel='Cross-Entropy Loss', title='Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs, self.history['train_acc'], label='Train')
        ax2.plot(epochs, self.history['val_acc'],   label='Val')
        ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy', ylim=(0, 1.05))
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved training curves → {save_path}")
        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray,
                              save_path: str = './checkpoints/confusion_matrix.png'):
        """Row-normalised heatmap (each row sums to 1 = true class)."""
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)   # avoid /0 for empty classes
        cm_norm  = cm.astype(float) / row_sums

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(ATTACK_TYPES, rotation=45, ha='right')
        ax.set_yticklabels(ATTACK_TYPES)
        ax.set(xlabel='Predicted', ylabel='True',
               title='Confusion Matrix (row-normalised)')

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = 'white' if cm_norm[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                        ha='center', va='center', color=color, fontsize=8)

        fig.colorbar(im, ax=ax, label='Fraction')
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved confusion matrix → {save_path}")
        plt.close()



