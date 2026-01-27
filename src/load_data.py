"""
Data loading module for ToN-IoT network traffic dataset.
Includes synthetic data generation for testing when real dataset is unavailable.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ToNIoTDataLoader:
    """Loader for ToN-IoT network traffic dataset."""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = Path(data_dir)
        
        # Real ToN-IoT feature columns (numeric features from Zeek logs)
        # These are the actual columns we'll use from the real dataset
        self.feature_columns = [
            'duration', 'src_bytes', 'dst_bytes', 'missed_bytes',
            'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes',
            'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA',
            'http_trans_depth', 'http_request_body_len', 'http_response_body_len',
            'http_status_code'
        ]
    
    def load_data(self, filename: Optional[str] = None, 
                  use_synthetic: bool = False,
                  n_samples: int = 50000) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load ToN-IoT data or generate synthetic data for testing.
        
        Args:
            filename: CSV file to load (e.g., 'Train_Test_Network.csv')
            use_synthetic: Generate synthetic data instead
            n_samples: Number of samples to load
            
        Returns:
            X: Feature dataframe
            y: Labels (0=normal, 1=anomaly)
        """
        if use_synthetic or filename is None:
            print(f"Generating {n_samples} synthetic samples for testing...")
            return self._generate_synthetic_data(n_samples)
        
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"File {filepath} not found. Generating synthetic data...")
            return self._generate_synthetic_data(n_samples)
        
        print(f"Loading data from {filepath}...")
        print("="*60)
        
        # CRITICAL FIX #1: Load ALL data first (to avoid alphabetical sorting trap)
        # Then shuffle and sample
        try:
            # Load entire dataset to get proper random sample
            print("Loading full dataset to ensure proper sampling...")
            df = pd.read_csv(filepath, low_memory=False)
            print(f"Full dataset size: {len(df):,} rows")
            
            # SHUFFLE before sampling to avoid alphabetical bias
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"Shuffled dataset")
            
            # Now take sample
            if n_samples < len(df):
                df = df.iloc[:n_samples].copy()
                print(f"Sampled {n_samples:,} rows")
            
        except MemoryError:
            print("Dataset too large, using chunk sampling...")
            # Fallback: sample chunks throughout file
            total_rows = sum(1 for _ in open(filepath)) - 1
            skip = sorted(np.random.choice(range(1, total_rows), 
                                          size=total_rows - n_samples, 
                                          replace=False))
            df = pd.read_csv(filepath, skiprows=skip, low_memory=False)
        
        print(f"\nProcessing {len(df):,} samples...")
        
        # Process labels - FIXED for binary classification
        if 'type' in df.columns:
            # Binary: normal=0, any attack=1
            y = (df['type'].str.lower() != 'normal').astype(int)
            print(f"Using 'type' column for binary labels")
            
            # Show attack distribution
            print(f"\nAttack type distribution:")
            type_dist = df['type'].value_counts()
            for attack_type, count in type_dist.items():
                print(f"  {attack_type:15s}: {count:6,} ({count/len(df)*100:5.1f}%)")
                
        elif 'label' in df.columns:
            y = (df['label'] != 0).astype(int)
            print(f"Using 'label' column")
        else:
            raise ValueError(f"No label column found. Available: {df.columns.tolist()}")
        
        print(f"\nLabel distribution:")
        print(f"  Normal:  {(y==0).sum():6,} ({(y==0).mean()*100:5.1f}%)")
        print(f"  Anomaly: {(y==1).sum():6,} ({(y==1).mean()*100:5.1f}%)")
        
        # CRITICAL FIX #2: Encode categorical features (proto, service, conn_state)
        categorical_features = []
        
        # Protocol (tcp, udp, icmp, etc.)
        if 'proto' in df.columns:
            print(f"\nEncoding 'proto' column...")
            proto_mapping = {proto: idx for idx, proto in enumerate(df['proto'].unique())}
            df['proto_encoded'] = df['proto'].map(proto_mapping)
            categorical_features.append('proto_encoded')
            print(f"  {len(proto_mapping)} unique protocols")
        
        # Service (http, dns, ssl, etc.)
        if 'service' in df.columns:
            print(f"Encoding 'service' column...")
            # Replace '-' with 'none'
            df['service'] = df['service'].fillna('none').replace('-', 'none')
            service_mapping = {svc: idx for idx, svc in enumerate(df['service'].unique())}
            df['service_encoded'] = df['service'].map(service_mapping)
            categorical_features.append('service_encoded')
            print(f"  {len(service_mapping)} unique services")
        
        # Connection state (SF, S0, REJ, etc.) - VERY IMPORTANT!
        if 'conn_state' in df.columns:
            print(f"Encoding 'conn_state' column...")
            df['conn_state'] = df['conn_state'].fillna('unknown').replace('-', 'unknown')
            state_mapping = {state: idx for idx, state in enumerate(df['conn_state'].unique())}
            df['conn_state_encoded'] = df['conn_state'].map(state_mapping)
            categorical_features.append('conn_state_encoded')
            print(f"  {len(state_mapping)} unique connection states")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # CRITICAL FIX #3: Drop IP addresses (cause overfitting)
        ip_cols = ['src_ip', 'dst_ip', 'src_ip_bytes', 'dst_ip_bytes']
        exclude_cols = ['label', 'type', 'Label', 'Type'] + ip_cols
        
        # Keep only valid numeric columns
        numeric_features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Combine numeric and encoded categorical
        all_features = numeric_features + categorical_features
        
        print(f"\nSelected {len(all_features)} features:")
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  Categorical (encoded): {len(categorical_features)}")
        
        # Create feature dataframe
        X = df[all_features].copy()
        
        # CRITICAL FIX #4: Clean '-' and non-numeric values
        print(f"\nCleaning data...")
        for col in X.columns:
            # Ensure we are looking at a Series and force conversion
            X[col] = pd.to_numeric(df[col].astype(str).replace('-', np.nan), errors='coerce')
        
        # Handle missing values
        n_missing = X.isnull().sum().sum()
        if n_missing > 0:
            print(f"  Filling {n_missing:,} missing values with median...")
            X = X.fillna(X.median())
            X = X.fillna(0)  # If entire column was NaN
        
        # Handle infinite values
        n_inf = np.isinf(X.values).sum()
        if n_inf > 0:
            print(f"  Replacing {n_inf:,} infinite values...")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            X = X.fillna(0)
        
        # Remove constant features (no variance)
        variances = X.var()
        constant_features = variances[variances == 0].index.tolist()
        if constant_features:
            print(f"  Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        if X.shape[1] == 0:
            raise ValueError("No valid features remaining after preprocessing!")
        
        print(f"\n{'='*60}")
        print(f"FINAL DATASET:")
        print(f"{'='*60}")
        print(f"Samples: {len(X):,}")
        print(f"Features: {X.shape[1]}")
        print(f"Normal: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
        print(f"Anomaly: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
        print(f"\nFeatures ({X.shape[1]}):")
        print(f"  First 10: {list(X.columns[:10])}")
        if X.shape[1] > 10:
            print(f"  Last 5:   {list(X.columns[-5:])}")
        print(f"{'='*60}\n")
        
        return X, y
    
    def _generate_synthetic_data(self, n_samples: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic network traffic data matching ToN-IoT Zeek format."""
        np.random.seed(42)
        
        # Generate normal traffic (80%)
        n_normal = int(n_samples * 0.8)
        n_anomaly = n_samples - n_normal
        
        # Normal traffic patterns (based on Zeek/ToN-IoT structure)
        normal_data = {
            # Connection duration (seconds)
            'duration': np.random.exponential(5.0, n_normal),
            
            # Bytes transferred
            'src_bytes': np.random.exponential(5000, n_normal),
            'dst_bytes': np.random.exponential(4000, n_normal),
            'missed_bytes': np.random.poisson(10, n_normal),
            
            # Packet counts
            'src_pkts': np.random.poisson(15, n_normal),
            'dst_pkts': np.random.poisson(12, n_normal),
            
            # IP-level bytes
            'src_ip_bytes': np.random.exponential(6000, n_normal),
            'dst_ip_bytes': np.random.exponential(5000, n_normal),
            
            # DNS features (many zeros for non-DNS traffic)
            'dns_qclass': np.random.choice([0, 1], n_normal, p=[0.95, 0.05]),
            'dns_qtype': np.random.choice([0, 1, 5, 15, 28], n_normal, p=[0.9, 0.04, 0.03, 0.02, 0.01]),
            'dns_rcode': np.random.choice([0, 3], n_normal, p=[0.98, 0.02]),
            'dns_AA': np.random.choice([0, 1], n_normal, p=[0.7, 0.3]),
            'dns_RD': np.random.choice([0, 1], n_normal, p=[0.3, 0.7]),
            'dns_RA': np.random.choice([0, 1], n_normal, p=[0.4, 0.6]),
            
            # HTTP features (many zeros for non-HTTP traffic)
            'http_trans_depth': np.random.choice([0, 1, 2, 3], n_normal, p=[0.7, 0.2, 0.07, 0.03]),
            'http_request_body_len': np.random.exponential(100, n_normal) * (np.random.random(n_normal) < 0.3),
            'http_response_body_len': np.random.exponential(2000, n_normal) * (np.random.random(n_normal) < 0.3),
            'http_status_code': np.random.choice([0, 200, 301, 404], n_normal, p=[0.7, 0.25, 0.03, 0.02]),
        }
        
        # Anomaly traffic patterns (different distributions for attacks)
        anomaly_data = {
            # Longer durations for some attacks, very short for others (port scans)
            'duration': np.concatenate([
                np.random.exponential(0.1, n_anomaly // 2),  # Quick scans
                np.random.exponential(30.0, n_anomaly - n_anomaly // 2)  # Long connections
            ]),
            
            # Much higher bytes for DDoS/DoS
            'src_bytes': np.random.exponential(50000, n_anomaly),
            'dst_bytes': np.random.exponential(500, n_anomaly),  # Low response
            'missed_bytes': np.random.poisson(100, n_anomaly),  # More packet loss
            
            # High packet rates for attacks
            'src_pkts': np.random.poisson(100, n_anomaly),
            'dst_pkts': np.random.poisson(5, n_anomaly),  # Few responses
            
            # Unbalanced IP bytes
            'src_ip_bytes': np.random.exponential(60000, n_anomaly),
            'dst_ip_bytes': np.random.exponential(1000, n_anomaly),
            
            # Unusual DNS patterns
            'dns_qclass': np.random.choice([0, 1, 255], n_anomaly, p=[0.5, 0.3, 0.2]),
            'dns_qtype': np.random.choice([0, 1, 5, 15, 28, 255], n_anomaly, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
            'dns_rcode': np.random.choice([0, 2, 3, 5], n_anomaly, p=[0.4, 0.3, 0.2, 0.1]),
            'dns_AA': np.random.choice([0, 1], n_anomaly, p=[0.5, 0.5]),
            'dns_RD': np.random.choice([0, 1], n_anomaly, p=[0.5, 0.5]),
            'dns_RA': np.random.choice([0, 1], n_anomaly, p=[0.6, 0.4]),
            
            # Suspicious HTTP patterns
            'http_trans_depth': np.random.choice([0, 1, 5, 10], n_anomaly, p=[0.3, 0.3, 0.3, 0.1]),
            'http_request_body_len': np.random.exponential(5000, n_anomaly) * (np.random.random(n_anomaly) < 0.7),
            'http_response_body_len': np.random.exponential(100, n_anomaly) * (np.random.random(n_anomaly) < 0.2),
            'http_status_code': np.random.choice([0, 200, 400, 404, 500, 503], n_anomaly, p=[0.3, 0.1, 0.2, 0.2, 0.1, 0.1]),
        }
        
        # Shuffle the anomaly duration to mix patterns
        np.random.shuffle(anomaly_data['duration'])
        
        # Combine normal and anomaly data
        X_normal = pd.DataFrame(normal_data)
        X_anomaly = pd.DataFrame(anomaly_data)
        X = pd.concat([X_normal, X_anomaly], ignore_index=True)
        
        # Create labels
        y = pd.Series([0] * n_normal + [1] * n_anomaly)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X.iloc[shuffle_idx].reset_index(drop=True)
        y = y.iloc[shuffle_idx].reset_index(drop=True)
        
        print(f"Generated {n_samples} synthetic samples (ToN-IoT Zeek format)")
        print(f"Normal samples: {n_normal}, Anomaly samples: {n_anomaly}")
        print(f"Features: {list(X.columns)}")
        
        return X, y
    



if __name__ == "__main__":
    # Test data loading
    loader = ToNIoTDataLoader()
    X, y = loader.load_data(use_synthetic=True, n_samples=10000)
    
    print("\nData shape:", X.shape)
    print("Label distribution:\n", y.value_counts())
    print("\nSample features:\n", X.head())
    print("\nFeature statistics:\n", X.describe())