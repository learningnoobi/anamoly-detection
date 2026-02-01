
import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path


# ─────────────────────────────────────────────────────────
# FEATURE DEFINITIONS  (from TON_IOT feature description PDF)
# ─────────────────────────────────────────────────────────

# Numeric features → log-transformed (if skewed) + RobustScaled in preprocess.py
#
# *** IMPORTANT ***
# src_ip_bytes and dst_ip_bytes are BYTE COUNTS (total length of the IP header
# field), NOT raw IP addresses.  They carry real traffic-volume signal and must
# be kept.  The original code dropped them by mistake because their names
# contain "ip".
NUMERIC_FEATURES = [
    # Connection activity
    'duration', 'src_bytes', 'dst_bytes', 'missed_bytes',
    # Statistical activity
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
    # Ports (numeric, will be scaled)
    'src_port', 'dst_port',
    # DNS numeric  (0 when DNS is not active — that zero IS informative)
    'dns_qclass', 'dns_qtype', 'dns_rcode',
    # HTTP numeric (0 when HTTP is not active — also informative)
    'http_trans_depth', 'http_request_body_len',
    'http_response_body_len', 'http_status_code',
]

# Categorical → label-encoded integer → will be embedded (nn.Embedding) in model
CATEGORICAL_FEATURES = ['proto', 'service', 'conn_state']

# Boolean features stored as "T" / "F" / "-" in the CSV.
# "-" means the protocol was NOT active for that flow.
# That is real signal (e.g. "-" on all SSL fields tells you it wasn't an SSL flow).
# We encode three states:  T → 1,  F → 0,  - (N/A) → 2
BOOLEAN_FEATURES = [
    'dns_aa', 'dns_rd', 'dns_ra', 'dns_rejected',
    'ssl_resumed', 'ssl_established',
    'weird_notice',
]

# ── What is DROPPED and why ──────────────────────────────
#   src_ip, dst_ip            → sparse string IPs; overfit to the lab network
#   dns_query                 → high-cardinality string, ~95 % are "-"
#   ssl_version / cipher /
#     subject / issuer        → high-cardinality strings, ~95 % "-"
#   http_method / uri /
#     version                 → high-cardinality strings, ~90 % "-"
#   http_user_agent           → PDF says "Number" but it is actually a string; sparse
#   http_orig_mime_types /
#     http_resp_mime_types    → high-cardinality strings, ~90 % "-"
#   weird_name / weird_addl   → high-cardinality strings, ~99 % "-"
#   label                     → binary (0/1) version of 'type'; redundant

TARGET  = 'type'
ATTACK_TYPES = [
    'normal', 'backdoor', 'ddos', 'dos', 'injection',
    'password', 'ransomware', 'scanning', 'xss', 'mitm',
]
NUM_CLASSES  = len(ATTACK_TYPES)   # 10


def load_data(filepath: str ='train_test_network.csv') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load TON_IOT CSV, select and encode features.

    Returns
    -------
    X : pd.DataFrame
        Columns: NUMERIC_FEATURES  +  *_enc (categorical)  +  *_enc (boolean).
        Numeric columns are still raw here — log + scale happens in preprocess.py.
    y : pd.Series  (int64)
        Class index 0-9, ordered by ATTACK_TYPES.
    """
    filepath =  f'../data/{filepath}'
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()   # normalise casing
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns\n")

    # ── 1. Multiclass target (uses 'type', ignores 'label') ──
    df[TARGET] = df[TARGET].str.strip().str.lower()
    type_to_idx = {t: i for i, t in enumerate(ATTACK_TYPES)}
    y = df[TARGET].map(type_to_idx)

    valid = y.notna()
    if (~valid).any():
        print(f"  Dropping {(~valid).sum()} rows with unknown types: "
              f"{df.loc[~valid, TARGET].unique()}")
    #   Drop invalid rows
    df = df.loc[valid].copy().reset_index(drop=True)
    y  = y[valid].astype(int).reset_index(drop=True)

    # ── 2. Numeric: coerce types, replace '-' with 0 ──
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace('-', np.nan), errors='coerce'
            ).fillna(0)
        else:
            # Column missing entirely — fill zeros (shouldn't happen with real TON_IOT)
            print(f"  WARNING: '{col}' not found in CSV, filling with 0")
            df[col] = 0

    # ── 3. Categorical: label-encode ──
    cat_enc_cols = []
    for col in CATEGORICAL_FEATURES:
        raw = df[col].astype(str).replace('-', 'none').fillna('none')
        codes, uniques = pd.factorize(raw, sort=True)   # sort=True → deterministic
        enc = f'{col}_enc'
        df[enc] = codes
        cat_enc_cols.append(enc)
        print(f"  {col} -> {len(uniques)} categories: "
              f"{dict(zip(uniques, range(len(uniques))))}")

    # ── 4. Boolean: 3-state encode  (T=1, F=0, -=2) ──
    bool_enc_cols = []
    for col in BOOLEAN_FEATURES:
        enc = f'{col}_enc'
        df[enc] = (
            df[col].astype(str).str.lower()
            .map({'t': 1, 'f': 0, '-': 2})
            .fillna(2)
            .astype(int)
        )
        bool_enc_cols.append(enc)

    # ── 5. Assemble final feature matrix ──
    feature_cols = NUMERIC_FEATURES + cat_enc_cols + bool_enc_cols
    X = df[feature_cols].copy()

    # ── Summary ──
    print(f"\n{'='*58}")
    print(f"  {X.shape[0]:,} samples x {X.shape[1]} features  "
          f"(numeric={len(NUMERIC_FEATURES)}, "
          f"cat={len(cat_enc_cols)}, bool={len(bool_enc_cols)})")
    print(f"\n  Class distribution:")
    for name, idx in type_to_idx.items():
        n = (y == idx).sum()
        print(f"    {idx}: {name:12s}  {n:>6,}  ({n/len(y)*100:5.1f} %)")
    print(f"{'='*58}\n")

    return X, y


# ─── quick standalone test ────────────────────────────────
if __name__ == "__main__":
    path =  "train_test_network.csv"
    X, y = load_data(path)
    # print(X.dtypes)
    # print(X.head())
    # print(f"\ny.value_counts():\n{y.value_counts().sort_index()}")