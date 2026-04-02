"""
utils.py — Shared preprocessing and evaluation utilities
Uses the existing load_data.py for feature engineering.
"""

import sys, os

_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from load_data import (
    load_data as _load_raw,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
    ATTACK_TYPES,
    NUM_CLASSES,
)
from torch.utils.data import DataLoader, TensorDataset
from preprocess import NetworkPreprocessor

# CONFIG
SEQ_LEN = 10
BATCH_SIZE = 256
RANDOM_STATE = 42
# Column names after encoding (created by load_data.py)
CAT_ENC_COLS = [f"{c}_enc" for c in CATEGORICAL_FEATURES]
BOOL_ENC_COLS = [f"{c}_enc" for c in BOOLEAN_FEATURES]

# Normal class index — load_data.py defines ATTACK_TYPES[0] = 'normal'
NORMAL_IDX = ATTACK_TYPES.index("normal")  # 0


# Columns that need log1p — heavy right-tailed distributions
LOG_COLS = {
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "dst_pkts",
    "src_ip_bytes",
    "dst_ip_bytes",
    "http_request_body_len",
    "http_response_body_len",
}


class AnomalyPreprocessor:

    def __init__(self):
        self.scaler = RobustScaler()
        self._log_idx = []
        self._num_idx = []
        self._col_names = []

    def _setup(self, columns: list):
        self._col_names = columns
        self._log_idx = [i for i, c in enumerate(columns) if c in LOG_COLS]
        self._num_idx = [i for i, c in enumerate(columns) if c in NUMERIC_FEATURES]

        # # Debug — verify indices found
        # print(f"  Log columns:     {[columns[i] for i in self._log_idx]}")
        # print(f"  Numeric columns: {[columns[i] for i in self._num_idx]}")
        # print(f"  Total features:  {len(columns)}")

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        self._setup(X_df.columns.tolist())
        X = X_df.values.astype(np.float64)
        X = self._log_transform(X)
        X[:, self._num_idx] = self.scaler.fit_transform(X[:, self._num_idx])
        return X.astype(np.float32)

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        X = X_df.values.astype(np.float64)
        X = self._log_transform(X)
        X[:, self._num_idx] = self.scaler.transform(X[:, self._num_idx])
        return X.astype(np.float32)

    def _log_transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        for i in self._log_idx:
            X[:, i] = np.log1p(np.maximum(X[:, i], 0.0))
        return X


# LOAD AND SPLIT DATA
def load_data(normal_path="normal_data.csv"):

    print("Loading normal data ...")
    X_df, _ = _load_raw(normal_path)

    # Split FIRST before fitting scaler — avoids leakage
    X_train_df, X_val_df = train_test_split(X_df, test_size=0.1, random_state=42)

    print(f"Raw split — Train: {len(X_train_df):,} | Val: {len(X_val_df):,}")

    # Fit preprocessor on train only, transform both
    prep = AnomalyPreprocessor()
    X_train = prep.fit_transform(X_train_df)
    X_train = prep.fit_transform(X_train_df)

    # Check per-column max to find the culprit
    col_names = X_train_df.columns.tolist()
    for i, col in enumerate(col_names):
        if abs(X_train[:, i]).max() > 100:
            print(f"  PROBLEM: {col:30s} max={X_train[:, i].max():.1f}")

    X_val = prep.transform(X_val_df)

    n_features = X_train.shape[1]

    # Sanity check — values should be roughly -5 to 5
    print(f"\nFeature stats after preprocessing:")
    print(f"  Mean: {X_train.mean():.4f}  (should be ~0)")
    print(f"  Std:  {X_train.std():.4f}   (should be ~1)")
    print(f"  Min:  {X_train.min():.4f}   (should be > -20)")
    print(f"  Max:  {X_train.max():.4f}   (should be < 20)")
    print(f"  Features: {n_features}")

    return X_train, X_val, n_features, prep


# def load_data_sequences(normal_path='normal_data.csv'):
#     from load_data import load_data as _load_raw
#     from preprocess import NetworkPreprocessor

#     print("Loading normal data for sequence models ...")
#     X_df, y = _load_raw(normal_path, need_group_for_sequence=True)

#     # Split first
#     X_train_df, X_val_df = train_test_split(
#         X_df, test_size=0.1, random_state=42
#     )

#     # Use preprocessor for log + scale only — not sequencing
#     prep = NetworkPreprocessor(seq_len=SEQ_LEN, stride=1)
#     prep._setup_indices(X_train_df)

#     # Log transform
#     X_train_np = prep._apply_log(X_train_df.values.astype(np.float64))
#     X_val_np   = prep._apply_log(X_val_df.values.astype(np.float64))

#     # Scale — fit on train only
#     X_train_np[:, prep._numeric_idx] = prep.scaler.fit_transform(
#         X_train_np[:, prep._numeric_idx])
#     X_val_np[:, prep._numeric_idx] = prep.scaler.transform(
#         X_val_np[:, prep._numeric_idx])

#     # Clip outliers
#     X_train_np = np.clip(X_train_np, -10, 10).astype(np.float32)
#     X_val_np   = np.clip(X_val_np,   -10, 10).astype(np.float32)

#     # Sliding window sequences — chronological order preserved
#     def make_sequences(X, seq_len, stride):
#         seqs = []
#         for i in range(0, len(X) - seq_len + 1, stride):
#             seqs.append(X[i:i + seq_len])
#         return np.array(seqs, dtype=np.float32)

#     def make_sequences_grouped(df, seq_len, stride, feature_cols):
#         sequences = []

#         grouped = df.groupby(["src_ip", "dst_ip"])

#         for _, group in grouped:
#             group = group.sort_values("ts")

#             X = group[feature_cols].values

#             if len(X) < seq_len:
#                 continue

#             for i in range(0, len(X) - seq_len + 1, stride):
#                 sequences.append(X[i:i + seq_len])

#         return np.array(sequences, dtype=np.float32)

#     X_train_seq = make_sequences_grouped(X_train_np, SEQ_LEN, stride=5)
#     X_val_seq   = make_sequences_grouped(X_val_np,   SEQ_LEN, stride=5)

#     n_features = X_train_seq.shape[2]

#     print(f"Train sequences: {len(X_train_seq):,} | "
#           f"Val sequences: {len(X_val_seq):,} | "
#           f"Shape: {X_train_seq.shape}")


#     return X_train_seq, X_val_seq, n_features, prep
def load_data_sequences(normal_path="normal_data.csv"):
    import numpy as np
    import pandas as pd
    from load_data import load_data as _load_raw
    from preprocess import NetworkPreprocessor

    SEQ_LEN = 10
    STRIDE = 1

    print("Loading normal data for sequence models ...")

    # 🔥 Load with grouping columns retained
    X_df, y = _load_raw(normal_path, need_group_for_sequence=True)

    # ─────────────────────────────────────────────
    # 1. Sort by time FIRST (critical)
    # ─────────────────────────────────────────────
    X_df = X_df.sort_values("ts").reset_index(drop=True)

    # ─────────────────────────────────────────────
    # 2. Time-based split (NOT random)
    # ─────────────────────────────────────────────
    split_idx = int(len(X_df) * 0.9)

    X_train_df = X_df.iloc[:split_idx].copy()
    X_val_df = X_df.iloc[split_idx:].copy()

    # ─────────────────────────────────────────────
    # 3. Identify feature columns (exclude grouping)
    # ─────────────────────────────────────────────
    GROUP_COLS = ["src_ip", "dst_ip", "proto", "ts"]

    feature_cols = [col for col in X_df.columns if col not in GROUP_COLS]

    # ─────────────────────────────────────────────
    # 4. Preprocessing (log + scale)
    # ─────────────────────────────────────────────
    prep = NetworkPreprocessor(seq_len=SEQ_LEN, stride=STRIDE)

    # setup numeric indices using ONLY feature columns
    prep._setup_indices(X_train_df[feature_cols])

    # Convert to numpy
    X_train_np = X_train_df[feature_cols].values.astype(np.float64)
    X_val_np = X_val_df[feature_cols].values.astype(np.float64)

    # Log transform
    X_train_np = prep._apply_log(X_train_np)
    X_val_np = prep._apply_log(X_val_np)

    # Scale (fit ONLY on train)
    X_train_np[:, prep._numeric_idx] = prep.scaler.fit_transform(
        X_train_np[:, prep._numeric_idx]
    )
    X_val_np[:, prep._numeric_idx] = prep.scaler.transform(
        X_val_np[:, prep._numeric_idx]
    )

    # Clip
    X_train_np = np.clip(X_train_np, -10, 10).astype(np.float32)
    X_val_np = np.clip(X_val_np, -10, 10).astype(np.float32)

    # ─────────────────────────────────────────────
    # 5. Put back into DataFrame (for grouping)
    # ─────────────────────────────────────────────
    X_train_proc = X_train_df[GROUP_COLS].copy()
    X_val_proc = X_val_df[GROUP_COLS].copy()

    for i, col in enumerate(feature_cols):
        X_train_proc[col] = X_train_np[:, i]
        X_val_proc[col] = X_val_np[:, i]

    # ─────────────────────────────────────────────
    # 6. Grouped sequence creation
    # ─────────────────────────────────────────────
    def make_sequences_grouped(df, seq_len, stride):
        sequences = []

        grouped = df.groupby(["src_ip", "dst_ip", "proto"])

        for _, group in grouped:
            group = group.sort_values("ts")

            X = group[feature_cols].values

            if len(X) < seq_len:
                continue

            for i in range(0, len(X) - seq_len + 1, stride):
                sequences.append(X[i : i + seq_len])

        return np.array(sequences, dtype=np.float32)

    X_train_seq = make_sequences_grouped(X_train_proc, SEQ_LEN, STRIDE)
    X_val_seq = make_sequences_grouped(X_val_proc, SEQ_LEN, STRIDE)

    n_features = X_train_seq.shape[2]

    print(f"\nTrain sequences: {len(X_train_seq):,}")
    print(f"Val sequences:   {len(X_val_seq):,}")
    print(f"Sequence shape:  {X_train_seq.shape}")

    return X_train_seq, X_val_seq, n_features, prep


class RowDataset(Dataset):
    """Single row per sample — for AE and VAE (no temporal modeling)."""

    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i]


class SequenceDataset(Dataset):
    """Sliding window sequences — for LSTM-VAE and T-CVAE."""

    def __init__(self, X, seq_len=SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq = seq_len
        self.n = len(X) - seq_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i : i + self.seq]


# DATASETS
class NormalDataset(Dataset):
    """Sliding window sequences over normal-only data for Stage 1 training."""

    def __init__(self, X, seq_len=SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.seq = seq_len
        self.n = len(X) - seq_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i : i + self.seq]


class TestDataset(Dataset):
    """Sliding window over mixed normal+attack data for Stage 1 evaluation."""

    def __init__(self, X, labels, seq_len=SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.lbl = torch.tensor(labels, dtype=torch.long)
        self.seq = seq_len
        self.n = len(X) - seq_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # Sequence is anomalous if ANY record in the window is an attack
        label = self.lbl[i : i + self.seq].max().item()
        return self.X[i : i + self.seq], label


class FullDataset(Dataset):
    """Full labeled dataset for Stage 2 classification."""

    def __init__(self, X, y, seq_len=SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq = seq_len
        self.n = len(X) - seq_len + 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        label = self.y[i + self.seq - 1].item()  # label of last record
        return self.X[i : i + self.seq], label


# EVALUATION HELPERS
def find_threshold(scores, labels):
    """Grid search over percentiles to find threshold that maximises F1."""
    thresholds = np.percentile(scores, np.arange(50, 99, 1))
    best_f1, best_tau = 0, thresholds[0]
    for tau in thresholds:
        preds = (scores > tau).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, tau
    return best_tau


def evaluate_anomaly(model_name, scores, labels):
    """Evaluate Stage 1 anomaly detection."""
    auroc = roc_auc_score(labels, scores)
    tau = find_threshold(scores, labels)
    preds = (scores > tau).astype(int)

    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  FPR:       {fpr:.4f}")

    return dict(
        model=model_name, auroc=auroc, f1=f1, precision=prec, recall=rec, fpr=fpr
    )


def evaluate_model(
    model,
    prep,
    model_name="Model",
    test_path="../data/train_test_network.csv",
    batch_size=BATCH_SIZE,
    device=None,
):
    """
    Generic evaluation function for all Stage 1 anomaly detection models.
    Works for AE, VAE,  with anomaly_score().

    Parameters
    ----------
    model      : trained PyTorch model with anomaly_score(x) method
    prep       : fitted AnomalyPreprocessor
    model_name : string label for results output
    test_path  : path to test CSV with both normal and attack rows
    """
    import pickle
    from load_data import load_data as _load_raw, ATTACK_TYPES

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NORMAL_IDX = ATTACK_TYPES.index("normal")

    # ── Load and preprocess test data ──────────────────────
    print(f"Loading test data for {model_name} ...")
    X_test_df, y_test_raw = _load_raw(test_path)
    X_test = prep.transform(X_test_df)
    y_test = (y_test_raw.values != NORMAL_IDX).astype(np.int64)

    print(
        f"Test: {len(X_test):,} rows "
        f"({y_test.sum():,} attacks, "
        f"{(y_test==0).sum():,} normal)"
    )

    # ── Score ──────────────────────────────────────────────
    from torch.utils.data import TensorDataset

    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=batch_size,
    )

    model.eval()
    all_scores, all_labels = [], []
    for x, lbl in test_loader:
        all_scores.extend(model.anomaly_score(x.to(device)))
        all_labels.extend(lbl.numpy())

    result = evaluate_anomaly(model_name, np.array(all_scores), np.array(all_labels))

    print(f"\nLaTeX row:")
    print(
        f"{model_name} & {result['auroc']:.4f} & {result['f1']:.4f} & "
        f"{result['precision']:.4f} & {result['recall']:.4f} & "
        f"{result['fpr']:.4f} \\\\"
    )

    return result

def evaluate_anomaly_S(model_name, scores, labels, threshold=None):
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score,
        recall_score, confusion_matrix
    )
    import numpy as np

    # AUROC (threshold-free)
    auroc = roc_auc_score(labels, scores)

    # Threshold
    if threshold is None:
        # fallback (NOT ideal for papers)
        threshold = np.percentile(scores, 95)

    preds = (scores > threshold).astype(int)

    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  FPR:       {fpr:.4f}")
    print(f"  Threshold: {threshold:.4f}")

    return dict(
        model=model_name,
        auroc=auroc,
        f1=f1,
        precision=prec,
        recall=rec,
        fpr=fpr,
        threshold=threshold
    )


def evaluate_model_sequences(
    model,
    prep,
    model_name='Model',
    test_path='../data/train_test_network.csv',
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from load_data import load_data as _load_raw, ATTACK_TYPES
    NORMAL_IDX = ATTACK_TYPES.index('normal')

    print(f"Loading test data for {model_name} ...")
    X_test_df, y_test_raw = _load_raw(
        test_path, need_group_for_sequence=True)

    GROUP_COLS   = ['src_ip', 'dst_ip', 'proto', 'ts']
    feature_cols = [c for c in X_test_df.columns if c not in GROUP_COLS]

    # Apply same preprocessor — log + scale only
    X_np = prep._apply_log(
        X_test_df[feature_cols].values.astype(np.float64))
    X_np[:, prep._numeric_idx] = prep.scaler.transform(
        X_np[:, prep._numeric_idx])
    X_np = np.clip(X_np, -10, 10).astype(np.float32)

    # Binary labels per row
    y_vals = (y_test_raw.values != NORMAL_IDX).astype(np.int64)

    # Put processed features back with group cols for grouping
    X_test_df = X_test_df[GROUP_COLS].copy()
    for i, col in enumerate(feature_cols):
        X_test_df[col] = X_np[:, i]
    X_test_df['_label'] = y_vals

    # Grouped sequences — same logic as training
    seqs, labels = [], []
    for _, group in X_test_df.groupby(['src_ip', 'dst_ip', 'proto']):
        group = group.sort_values('ts')
        X_g = group[feature_cols].values.astype(np.float32)
        y_g = group['_label'].values

        if len(X_g) < seq_len:
            continue

        for i in range(0, len(X_g) - seq_len + 1):
            seqs.append(X_g[i:i + seq_len])
            # anomalous if any row in window is attack
            labels.append(int(y_g[i:i + seq_len].max()))

    X_test_seq = np.array(seqs, dtype=np.float32)
    y_test     = np.array(labels, dtype=np.int64)

    print(f"Test sequences: {len(X_test_seq):,} "
          f"({y_test.sum():,} attacks, "
          f"{(y_test==0).sum():,} normal)")

    # Score
    from torch.utils.data import TensorDataset
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test_seq, dtype=torch.float32),
            torch.tensor(y_test,     dtype=torch.long)
        ),
        batch_size=batch_size
    )

    model.eval()
    all_scores, all_labels = [], []
    for x, lbl in test_loader:
        all_scores.extend(model.anomaly_score(x.to(device)))
        all_labels.extend(lbl.numpy())


    print(f"Scores shape: {len(all_scores)}")
    print(f"Labels shape: {len(all_labels)}")
    print(f"Score stats: mean={np.mean(all_scores):.4f} "
        f"std={np.std(all_scores):.4f} "
        f"min={np.min(all_scores):.4f} "
        f"max={np.max(all_scores):.4f}")
    print(f"Label distribution: "
        f"normal={sum(l==0 for l in all_labels)} "
        f"attack={sum(l==1 for l in all_labels)}")
    
    scores_arr = np.array(all_scores)
    labels_arr = np.array(all_labels)
    print(f"\nMean score for NORMAL:  {scores_arr[labels_arr==0].mean():.4f}")
    print(f"Mean score for ATTACK:  {scores_arr[labels_arr==1].mean():.4f}")
    result = evaluate_anomaly_S(model_name,
                              scores_arr,
                              labels_arr)

    print(f"\nLaTeX row:")
    print(f"{model_name} & {result['auroc']:.4f} & {result['f1']:.4f} & "
          f"{result['precision']:.4f} & {result['recall']:.4f} & "
          f"{result['fpr']:.4f} \\\\")

    return result

SEQ_LEN    = 10
BATCH_SIZE = 256
 
 
# ─────────────────────────────────────────────
# THRESHOLD SEARCH
# ─────────────────────────────────────────────
# def find_threshold(scores, labels):
#     """Find threshold that maximises F1."""
#     thresholds = np.percentile(scores, np.arange(1, 99, 1))
#     best_f1, best_tau = 0, thresholds[0]
#     for tau in thresholds:
#         preds = (scores > tau).astype(int)
#         f1    = f1_score(labels, preds, zero_division=0)
#         if f1 > best_f1:
#             best_f1, best_tau = f1, tau
#     return best_tau
 
 
# ─────────────────────────────────────────────
# PRINT METRICS
# # ─────────────────────────────────────────────
# def print_metrics(model_name, scores, labels):
#     auroc = roc_auc_score(labels, scores)
#     tau   = find_threshold(scores, labels)
#     preds = (scores > tau).astype(int)
 
#     f1   = f1_score(labels,    preds, zero_division=0)
#     prec = precision_score(labels, preds, zero_division=0)
#     rec  = recall_score(labels,    preds, zero_division=0)
#     tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
#     fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
 
#     print(f"\n{'='*45}")
#     print(f"  {model_name}")
#     print(f"{'='*45}")
#     print(f"  AUROC:     {auroc:.4f}")
#     print(f"  F1:        {f1:.4f}")
#     print(f"  Precision: {prec:.4f}")
#     print(f"  Recall:    {rec:.4f}")
#     print(f"  FPR:       {fpr:.4f}")
 
#     # Score separation diagnostic
#     print(f"\n  Score diagnostics:")
#     print(f"    Mean score NORMAL: {scores[labels==0].mean():.4f}")
#     print(f"    Mean score ATTACK: {scores[labels==1].mean():.4f}")
#     print(f"    Threshold used:    {tau:.4f}")
 
#     print(f"\n  LaTeX row:")
#     print(f"  {model_name} & {auroc:.4f} & {f1:.4f} & "
#           f"{prec:.4f} & {rec:.4f} & {fpr:.4f} \\\\")
 
#     return dict(model=model_name, auroc=auroc, f1=f1,
#                 precision=prec, recall=rec, fpr=fpr)
 
 

# def evaluate_model_sequences(
#     model,
#     prep,
#     model_name='Model',
#     test_path='../data/train_test_network.csv',
#     seq_len=SEQ_LEN,
#     batch_size=BATCH_SIZE,
#     balance_test=True,
#     device=None
# ):
#     """
#     Evaluate a sequence-based anomaly detection model.
 
#     Parameters
#     ----------
#     model       : trained model with anomaly_score(x) method
#     prep        : fitted preprocessor with _apply_log, scaler, _numeric_idx
#     model_name  : label for output
#     test_path   : path to test CSV with normal + attack rows
#     seq_len     : sequence length (must match training)
#     balance_test: if True, balance normal/attack sequences 50/50
#     device      : torch device
#     """
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
#     from load_data import load_data as _load_raw, ATTACK_TYPES
#     NORMAL_IDX = ATTACK_TYPES.index('normal')
 
#     # ── Load test data ──────────────────────────────────────
#     print(f"\nLoading test data for {model_name} ...")
#     X_test_df, y_test_raw = _load_raw(
#         test_path, need_group_for_sequence=True)
 
#     GROUP_COLS   = ['src_ip', 'dst_ip', 'proto', 'ts']
#     feature_cols = [c for c in X_test_df.columns if c not in GROUP_COLS]
 
#     print(f"  Raw test rows: {len(X_test_df):,}")
#     print(f"  Features: {len(feature_cols)}")
 
#     # ── Preprocess — log + scale ────────────────────────────
#     X_np = prep._apply_log(
#         X_test_df[feature_cols].values.astype(np.float64))
#     X_np[:, prep._numeric_idx] = prep.scaler.transform(
#         X_np[:, prep._numeric_idx])
#     X_np = np.clip(X_np, -10, 10).astype(np.float32)
 
#     # Binary labels per row
#     y_vals = (y_test_raw.values != NORMAL_IDX).astype(np.int64)
 
#     # ── Put processed features back with group cols ─────────
#     import pandas as pd
#     X_proc = X_test_df[GROUP_COLS].copy()
#     for i, col in enumerate(feature_cols):
#         X_proc[col] = X_np[:, i]
#     X_proc['_label'] = y_vals
 
#     # ── Group by connection, create sequences ───────────────
#     print("\n  Creating sequences grouped by connection ...")
#     seqs, labels = [], []
#     groups_total    = 0
#     groups_skipped  = 0
 
#     # Check if ts column exists for sorting
#     has_ts = 'ts' in X_proc.columns and X_proc['ts'].notna().any()
 
#     for _, group in X_proc.groupby(['src_ip', 'dst_ip', 'proto']):
#         groups_total += 1
 
#         # Sort by timestamp if available
#         if has_ts:
#             group = group.sort_values('ts')
 
#         X_g = group[feature_cols].values.astype(np.float32)
#         y_g = group['_label'].values
 
#         # Skip groups too small for a sequence
#         if len(X_g) < seq_len:
#             groups_skipped += 1
#             continue
 
#         for i in range(0, len(X_g) - seq_len + 1):
#             seqs.append(X_g[i:i + seq_len])
#             # Sequence is anomalous if ANY row in window is attack
#             labels.append(int(y_g[i:i + seq_len].max()))
 
#     print(f"  Connection groups: {groups_total:,} total, "
#           f"{groups_skipped:,} skipped (< {seq_len} flows)")
 
#     X_test_seq = np.array(seqs,   dtype=np.float32)
#     y_test_seq = np.array(labels, dtype=np.int64)
 
#     n_normal = (y_test_seq == 0).sum()
#     n_attack = (y_test_seq == 1).sum()
#     print(f"\n  Before balancing:")
#     print(f"    Normal sequences:  {n_normal:,} ({n_normal/len(y_test_seq)*100:.1f}%)")
#     print(f"    Attack sequences:  {n_attack:,} ({n_attack/len(y_test_seq)*100:.1f}%)")
 
#     # ── Balance test set ────────────────────────────────────
#     if balance_test and n_normal > 0 and n_attack > 0:
#         normal_idx = np.where(y_test_seq == 0)[0]
#         attack_idx = np.where(y_test_seq == 1)[0]
 
#         # Take min of both classes
#         n_each = min(len(normal_idx), len(attack_idx))
 
#         normal_sampled = np.random.RandomState(42).choice(
#             normal_idx, size=n_each, replace=False)
#         attack_sampled = np.random.RandomState(42).choice(
#             attack_idx, size=n_each, replace=False)
 
#         balanced_idx = np.concatenate([normal_sampled, attack_sampled])
#         np.random.RandomState(42).shuffle(balanced_idx)
 
#         X_test_seq = X_test_seq[balanced_idx]
#         y_test_seq = y_test_seq[balanced_idx]
 
#         print(f"\n  After balancing:")
#         print(f"    Total sequences: {len(X_test_seq):,} "
#               f"(50% normal, 50% attack)")
 
#     # ── Score with model ────────────────────────────────────
#     print(f"\n  Scoring sequences ...")
#     test_loader = DataLoader(
#         TensorDataset(
#             torch.tensor(X_test_seq, dtype=torch.float32),
#             torch.tensor(y_test_seq, dtype=torch.long)
#         ),
#         batch_size=batch_size
#     )
 
#     model.eval()
#     all_scores, all_labels = [], []
#     for x, lbl in test_loader:
#         all_scores.extend(model.anomaly_score(x.to(device)))
#         all_labels.extend(lbl.numpy())
 
#     scores = np.array(all_scores)
#     labels = np.array(all_labels)
 
#     # ── Compute and print metrics ───────────────────────────
#     result = print_metrics(model_name, scores, labels)
#     return result