import torch
import numpy as np
import pandas as pd
from pathlib import Path
from preprocess import NetworkPreprocessor
from load_data import  NUM_CLASSES, ATTACK_TYPES
from model import NetworkTransformer

# ─────────────────────────────────────────────────────────────
# Load trained model + preprocessor
# ─────────────────────────────────────────────────────────────
def load_trained_model(
    checkpoint_path: str = './checkpoints/best_model.pt',
    preprocessor_path: str = './preprocessor.pkl',
    device: str = 'cpu',
    # Model hyperparameters (must match training)
    d_model: int = 128,
    nhead: int = 8,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    cat_embed_dim: int = 16,
    dropout: float = 0.1,
):
    # Check files exist
    for p in (checkpoint_path, preprocessor_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"'{p}' not found. Train model first.")

    # Load preprocessor
    preprocessor = NetworkPreprocessor.load(preprocessor_path)
    col_names = preprocessor._col_names

    # Identify categorical and numeric features
    cat_col_names = ['proto_enc', 'service_enc', 'conn_state_enc']
    cat_indices = [col_names.index(c) for c in cat_col_names]
    num_numeric = len(col_names) - len(cat_indices)

    # Load checkpoint and extract embedding sizes directly
    state = torch.load(checkpoint_path, map_location=device)
    cat_vocab_sizes = [
        state[f'embeddings.{i}.weight'].shape[0]
        for i in range(len(cat_indices))
    ]

    # Reconstruct the model exactly
    model = NetworkTransformer(
        num_numeric=num_numeric,
        cat_indices=cat_indices,
        cat_vocab_sizes=cat_vocab_sizes,
        num_classes=NUM_CLASSES,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_feedforward, cat_embed_dim=cat_embed_dim,
        dropout=dropout,
    )
    model.load_state_dict(state)
    model.to(device).eval()

    print(f"Model loaded → {checkpoint_path}")
    print(f"Preprocessor loaded → {preprocessor_path}")
    return model, preprocessor

# ─────────────────────────────────────────────────────────────
# Prediction function (accepts NumPy array or pandas DataFrame)
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(model, preprocessor, X_input, device='cpu'):
    """
    Classify a stream of flows.
    X_input: pandas DataFrame or NumPy array with shape (N, num_features)
    Returns:
        preds: np.array of class indices
        probs: np.array of softmax probabilities
        names: list of human-readable attack types
    """
    if isinstance(X_input, np.ndarray):
        X = pd.DataFrame(X_input, columns=preprocessor._col_names)
    else:
        X = X_input.copy()

    seq_len = preprocessor.seq_len
    stride = preprocessor.stride

    # Apply preprocessing (log + scaling)
    Xn = preprocessor._apply_log(X.values.astype(np.float64))
    Xn[:, preprocessor._numeric_idx] = preprocessor.scaler.transform(
        Xn[:, preprocessor._numeric_idx]
    )

    # Build sliding windows
    seqs = [Xn[i:i+seq_len] for i in range(0, len(Xn)-seq_len+1, stride)]
    if not seqs:
        raise ValueError(f"Need at least seq_len={seq_len} flows. Got {len(Xn)}.")

    X_tensor = torch.from_numpy(np.array(seqs, dtype=np.float32)).to(device)

    # Forward pass in chunks
    chunk_size = 2048
    all_probs = []
    for start in range(0, len(X_tensor), chunk_size):
        logits = model(X_tensor[start:start+chunk_size])
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=1)
    names = [ATTACK_TYPES[p] for p in preds]

    return preds, probs, names

# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model and preprocessor
    model, preprocessor = load_trained_model(device=device)

    from load_data import load_data 
    X, y = load_data()  # X: pd.DataFrame
    preds, probs, names = predict(model, preprocessor, X, device)

    # Print summary
    print(f"\nPredicted {len(preds)} sequences from {len(X)} flows")
    unique, counts = np.unique(names, return_counts=True)
    print("Predicted distribution:")
    for nm, ct in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"  {nm:>12s}: {ct:>5}  ({ct/len(preds)*100:5.1f}%)")
    
    # First 15 predictions with top-2 confidences
    print(f"\n{'Seq':>5} {'Predicted':>12} {'Conf':>7} {'Runner-up'}")
    print("─"*55)
    for i in range(min(15, len(preds))):
        top2 = probs[i].argsort()[-2:][::-1]
        print(f"{i:>5} {names[i]:>12} {probs[i].max()*100:6.1f}% "
              f"{ATTACK_TYPES[top2[1]]} ({probs[i, top2[1]]*100:.1f}%)")
