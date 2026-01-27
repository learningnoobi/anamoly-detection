"""
Data preprocessing module for transforming network traffic into sequences.
Includes normalization, sequence creation, and train/test splitting.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import pickle


class NetworkTrafficPreprocessor:
    """Preprocess network traffic for Transformer-based detection."""
    
    def __init__(self, sequence_length: int = 10, stride: int = 5):
        self.sequence_length = sequence_length
        self.stride = stride
        # RobustScaler is better than StandardScaler if data has extreme outliers (like DDoS)
        self.scaler = RobustScaler()
        self.feature_names = None
        
    def _apply_log_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to skewed network metrics like bytes and duration."""
        skewed_cols = [col for col in X.columns if 'bytes' in col or 'duration' in col or 'pkts' in col]
        for col in skewed_cols:
            # log1p handles zero values safely: log(1 + x)
            X[col] = np.log1p(X[col].values)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        self.feature_names = X.columns.tolist()
        
        # 1. Log Transform skewed features
        X = self._apply_log_transform(X)
        
        # 2. Fit and Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # 3. Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)
        
        print(f"Created {len(X_seq)} sequences. Input features: {len(self.feature_names)}")
        return X_seq, y_seq
    
    def transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        X = self._apply_log_transform(X)
        X_scaled = self.scaler.transform(X)
        return self._create_sequences(X_scaled, y.values)
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding window for Transformer input.
        
        """
        sequences = []
        labels = []
        
        n_samples = len(X)
        # Convert to float32 for PyTorch efficiency
        X = X.astype(np.float32)
        
        for i in range(0, n_samples - self.sequence_length + 1, self.stride):
            seq = X[i:i + self.sequence_length]
            # Sequence Label: 1 if ANY packet in the window is an attack
            label = 1 if np.any(y[i:i + self.sequence_length] == 1) else 0
            
            sequences.append(seq)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)

    def save(self, filepath: str):
        state = {
            'params': {'sequence_length': self.sequence_length, 'stride': self.stride},
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

def prepare_dataloaders(X_seq: np.ndarray, y_seq: np.ndarray, batch_size: int = 128):
    """Prepares stratified splits for training."""
    # Stratify ensures train/val/test all have the same % of anomalies
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Create Tensors
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    return {
        'train': torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': torch.utils.data.DataLoader(val_ds, batch_size=batch_size),
        'test': torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    }


if __name__ == "__main__":
    # Test preprocessing
    from load_data import ToNIoTDataLoader
    
    # Load data
    loader = ToNIoTDataLoader()
    X, y = loader.load_data(use_synthetic=True, n_samples=5000)
    
    # Preprocess
    preprocessor = NetworkTrafficPreprocessor(sequence_length=10, stride=5)
    X_seq, y_seq = preprocessor.fit_transform(X, y)
    
    print(f"\nSequence statistics:")
    print(f"Total sequences: {len(X_seq)}")
    print(f"Normal sequences: {(y_seq==0).sum()}")
    print(f"Anomaly sequences: {(y_seq==1).sum()}")
    
    # Create dataloaders
    dataloaders = prepare_dataloaders(X_seq, y_seq, batch_size=32)
    
    # Test batch
    for batch_X, batch_y in dataloaders['train']:
        print(f"\nSample batch shape: {batch_X.shape}")
        print(f"Sample labels shape: {batch_y.shape}")
        break