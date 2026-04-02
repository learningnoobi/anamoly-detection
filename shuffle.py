import pandas as pd
import numpy as np

df = pd.read_csv('./data/train_test_network.csv')
print(f"Before: First 10 classes: {df['type'].head(10).tolist()}")

df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"After: First 10 classes: {df_shuffled['type'].head(10).tolist()}")

df_shuffled.to_csv('train_test_network_SHUFFLED.csv', index=False)
print("Saved shuffled data")