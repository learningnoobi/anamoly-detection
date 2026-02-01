

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import pickle

from load_data import NUMERIC_FEATURES, ATTACK_TYPES, NUM_CLASSES

# Columns to log1p — heavy right-tailed counts / durations.
# Ports, status codes, and small integer codes (dns_qclass etc.) are left raw.
LOG_COLS = {
    'duration', 'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'dst_pkts', 'src_ip_bytes', 'dst_ip_bytes',
    'http_request_body_len', 'http_response_body_len',
}


class NetworkPreprocessor:
    """Log-transform → scale (numeric only) → grouped sequences."""

    def __init__(self, seq_len: int = 10, stride: int = 5):
        self.seq_len = seq_len
        self.stride  = stride
        self.scaler  = RobustScaler()
        # Populated on first fit — indices into the numpy array
        self._col_names:   list = []
        self._numeric_idx: list = []   # columns to scale
        self._log_idx:     list = []   # columns to log-transform

    # ── internal helpers ──────────────────────────────────────

    def _setup_indices(self, X: pd.DataFrame):
        """Map column names to positional indices once."""
        self._col_names   = X.columns.tolist()
        self._numeric_idx = [i for i, c in enumerate(self._col_names) if c in NUMERIC_FEATURES]
        self._log_idx     = [i for i, c in enumerate(self._col_names) if c in LOG_COLS]

    def _apply_log(self, X: np.ndarray) -> np.ndarray:
        """log1p only on skewed columns; clips negatives to 0 first."""
        X = X.copy()
        for i in self._log_idx:
            X[:, i] = np.log1p(np.maximum(X[:, i], 0.0))
        return X

    # ── public API ────────────────────────────────────────────

    def fit_transform(self, X: pd.DataFrame, y: pd.Series
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Fit scaler on training flows, then transform + create sequences."""
        self._setup_indices(X)
        Xn = self._apply_log(X.values.astype(np.float64))
        Xn[:, self._numeric_idx] = self.scaler.fit_transform(Xn[:, self._numeric_idx])
        return self._sequence(Xn, y.values.astype(np.int64))

    def transform(self, X: pd.DataFrame, y: pd.Series
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform with already-fitted scaler (for val / test)."""
        Xn = self._apply_log(X.values.astype(np.float64))
        Xn[:, self._numeric_idx] = self.scaler.transform(Xn[:, self._numeric_idx])
        return self._sequence(Xn, y.values.astype(np.int64))

    def _sequence(self, X: np.ndarray, y: np.ndarray,
                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sliding window WITHIN each class, then shuffle at the sequence level.

        Why within-class?
            Flows belonging to the same attack type share temporal context.
            E.g. a scanning attack = many rapid short-lived connections fired
            in rapid succession.  A window that captures 10 consecutive scanning
            flows encodes that pattern.  A window that mixes scanning + normal +
            xss flows would be noise.

        Why shuffle sequences (not rows)?
            Each sequence is a self-contained sample with a single clean label.
            Shuffling sequences lets each training batch see a natural mix of
            all 10 attack types without breaking any sequence's internal order.
        """
        seqs, labels = [], []

        for cls in range(NUM_CLASSES):
            X_cls = X[y == cls]
            n     = len(X_cls)

            if n < self.seq_len:
                # Edge case: fewer flows than window size.
                # Tile (repeat) the available flows to fill one sequence.
                X_cls = np.tile(X_cls, (self.seq_len // n + 1, 1))[:self.seq_len]
                seqs.append(X_cls)        # shape (seq_len, features)
                labels.append(cls)
                n_seq = 1
            else:
                n_seq = 0
                for i in range(0, n - self.seq_len + 1, self.stride):
                    seqs.append(X_cls[i:i + self.seq_len])
                    labels.append(cls)
                    n_seq += 1

            print(f"    {cls}: {ATTACK_TYPES[cls]:12s} | "
                  f"{n:>6,} flows -> {n_seq:>5,} sequences")

        X_seq = np.array(seqs, dtype=np.float32)   # (N, seq_len, features)
        y_seq = np.array(labels, dtype=np.int64)   # (N,)

        # Shuffle sequences
        perm = np.random.RandomState(seed).permutation(len(X_seq))
        return X_seq[perm], y_seq[perm]

    # ── persistence ───────────────────────────────────────────

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str) -> 'NetworkPreprocessor':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        return obj


# ── stand-alone helpers ───────────────────────────────────────


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Inverse-frequency weights for nn.CrossEntropyLoss.

    MITM has ~1 043 flows (vs 50 000 normal).  After splitting and
    sequencing it ends up with roughly 200 train sequences.  Without
    weighting the model will simply never predict MITM.

    Usage in your training loop:
        weights   = compute_class_weights(y_train_seq)
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1)                           # avoid div-by-zero
    w      = len(y) / (NUM_CLASSES * counts)
    w      = w / w.sum() * NUM_CLASSES                       # normalise: sum == NUM_CLASSES

    print("  Class weights:")
    for i, name in enumerate(ATTACK_TYPES):
        print(f"    {i}: {name:12s}  weight={w[i]:6.2f}  (n={int(counts[i]):>6,})")

    return torch.tensor(w, dtype=torch.float32)


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    seq_len:    int   = 10,
    stride:     int   = 5,
    batch_size: int   = 128,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[Dict[str, DataLoader], NetworkPreprocessor, torch.Tensor]:
    """
    Single entry-point that runs the full pipeline.

    Returns
    -------
    loaders       - dict  {'train', 'val', 'test'}  -> DataLoader
    preprocessor  - fitted NetworkPreprocessor  (save it for inference later)
    class_weights - float32 tensor for CrossEntropyLoss  (computed on train only)
    """
    # ── 1. Flow-level stratified split ──────────────────────
    print("── 1. Flow-level stratified split ──")
    temp = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_ratio / temp,          # e.g. 0.15/0.30 = 0.5 of temp
        random_state=42, stratify=y_temp
    )
    print(f"  Flows — train: {len(X_train):,}  "
          f"val: {len(X_val):,}  test: {len(X_test):,}\n")

    # ── 2. Fit on train, transform all three splits ─────────
    print("── 2. Train  (fit_transform) ──")
    preprocessor              = NetworkPreprocessor(seq_len=seq_len, stride=stride)
    X_train_s, y_train_s      = preprocessor.fit_transform(X_train, y_train)

    print("\n── 3. Val    (transform) ──")
    X_val_s, y_val_s          = preprocessor.transform(X_val,  y_val)

    print("\n── 4. Test   (transform) ──")
    X_test_s, y_test_s        = preprocessor.transform(X_test, y_test)

    # ── 3. Class weights from training sequences only ──────
    print("\n── 5. Class weights ──")
    class_weights = compute_class_weights(y_train_s)

    # ── 4. Wrap in DataLoaders ──────────────────────────────
    def _loader(Xs, ys, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=shuffle, pin_memory=True)

    loaders = {
        'train': _loader(X_train_s, y_train_s, shuffle=True),
        'val':   _loader(X_val_s,   y_val_s,   shuffle=False),
        'test':  _loader(X_test_s,  y_test_s,  shuffle=False),
    }

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  Sequences — train: {len(X_train_s):,}  "
          f"val: {len(X_val_s):,}  test: {len(X_test_s):,}")
    print(f"  Each sequence shape: ({seq_len}, {X_train_s.shape[2]})  "
          f"[seq_len x features]")
    print(f"  Batch X dtype: float32   Batch y dtype: int64")
    print(f"{'='*58}\n")

    return loaders, preprocessor, class_weights


# ─── quick sanity-check ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    from load_data import load_data

    path = sys.argv[1] if len(sys.argv) > 1 else "train_test_network.csv"
    X, y = load_data(path)

    loaders, preprocessor, class_weights = prepare_data(
        X, y, seq_len=10, stride=5, batch_size=64
    )

    # Spot-check one batch
    bX, by = next(iter(loaders['train']))
    print(f"  Batch X : shape={tuple(bX.shape)}  dtype={bX.dtype}")
    print(f"  Batch y : shape={tuple(by.shape)}  dtype={by.dtype}")
    print(f"  Classes in this batch: {sorted(by.unique().tolist())}")

    print(f"\n  In  training loop:")
    print(f"    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))")

    preprocessor.save("preprocessor.pkl")
    print(f"  Preprocessor saved -> preprocessor.pkl")