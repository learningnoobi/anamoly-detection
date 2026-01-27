"""
Training script for Transformer-CVAE anomaly detection model.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List
import matplotlib.pyplot as plt

from .model import TransformerCVAE, vae_loss


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class Trainer:
    """Training manager for anomaly detection model."""
    
    def __init__(self, model: TransformerCVAE, device: str = 'cpu',
                 learning_rate: float = 1e-4, beta: float = 1.0):
        """
        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate
            beta: KL divergence weight (beta-VAE)
        """
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kld': [],
            'val_loss': [],
            'val_recon': [],
            'val_kld': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_x, _ in pbar:
            batch_x = batch_x.to(self.device)
            
            # Forward pass
            recon, mu, logvar = self.model(batch_x)
            loss, recon_loss, kld = vae_loss(recon, batch_x, mu, logvar, self.beta)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            batch_size = batch_x.size(0)
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
            n_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item() / batch_size,
                'recon': recon_loss.item() / batch_size,
                'kld': kld.item() / batch_size
            })
        
        # Average over batches
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches
        
        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kld': avg_kld
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(self.device)
                
                recon, mu, logvar = self.model(batch_x)
                loss, recon_loss, kld = vae_loss(recon, batch_x, mu, logvar, self.beta)
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kld += kld.item()
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kld = total_kld / n_batches
        
        return {
            'loss': avg_loss,
            'recon': avg_recon,
            'kld': avg_kld
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 50, early_stopping_patience: int = 10,
              save_dir: str = './checkpoints'):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_loss = float('inf')
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Beta (KL weight): {self.beta}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kld'].append(train_metrics['kld'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kld'].append(val_metrics['kld'])
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Recon: {train_metrics['recon']:.4f}, "
                  f"KLD: {train_metrics['kld']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Recon: {val_metrics['recon']:.4f}, "
                  f"KLD: {val_metrics['kld']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'history': self.history
                }, save_path / 'best_model.pt')
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")
            
            # Early stopping
            early_stopping(val_metrics['loss'])
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, save_path / 'final_model.pt')
        
        # Save training history
        with open(save_path / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.history
    
    def plot_history(self, save_path: str = None):
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total loss
        axes[0].plot(self.history['train_loss'], label='Train')
        axes[0].plot(self.history['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(self.history['train_recon'], label='Train')
        axes[1].plot(self.history['val_recon'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].set_title('Reconstruction Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # KL divergence
        axes[2].plot(self.history['train_kld'], label='Train')
        axes[2].plot(self.history['val_kld'], label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL Divergence')
        axes[2].set_title('KL Divergence')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Create directory if it doesn't exist
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Example training script
    from load_data import ToNIoTDataLoader
    from preprocess import NetworkTrafficPreprocessor, prepare_dataloaders
    
    print("Loading data...")
    loader = ToNIoTDataLoader()
    X, y = loader.load_data(use_synthetic=True, n_samples=20000)
    
    print("\nPreprocessing...")
    preprocessor = NetworkTrafficPreprocessor(sequence_length=10, stride=5)
    X_seq, y_seq = preprocessor.fit_transform(X, y)
    
    print("\nPreparing dataloaders...")
    dataloaders = prepare_dataloaders(X_seq, y_seq, batch_size=64)
    
    print("\nInitializing model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerCVAE(
        input_dim=X.shape[1],
        seq_len=10,
        d_model=128,
        nhead=8,
        num_layers=3,
        latent_dim=64
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(model, device=device, learning_rate=1e-3, beta=0.5)
    history = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=30,
        early_stopping_patience=10
    )
    
    # Plot history
    trainer.plot_history(save_path='./checkpoints/training_history.png')