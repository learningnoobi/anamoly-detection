import time
import torch
from src.load_data  import load_data, NUM_CLASSES
from src.preprocess import prepare_data
from src.model import NetworkTransformer
from src.train import Trainer

# csv_path = sys.argv[1] if len(sys.argv) > 1 else "train_test_network.csv"
start_time = time.time()

# ── 1. Load raw data ──────────────────────────────────
X, y = load_data()

# ── 2. Extract categorical metadata for the model ──
#   load_data() appends columns in this order:
#       NUMERIC_FEATURES (16)  |  proto_enc, service_enc, conn_state_enc  |  bool_enc (7)
#   The model needs to know which columns are categorical so it can route
#   them through nn.Embedding instead of the linear projection.
cat_col_names   = ['proto_enc', 'service_enc', 'conn_state_enc']
col_list        = X.columns.tolist()
cat_indices     = [col_list.index(c) for c in cat_col_names]
cat_vocab_sizes = [int(X[c].max()) + 1 for c in cat_col_names]
num_numeric     = len(col_list) - len(cat_indices)   # numeric + boolean columns

print(f"\n  Feature layout:")
print(f"    total columns      = {len(col_list)}")
print(f"    numeric + boolean  = {num_numeric}  (fed to linear projection)")
print(f"    categorical        = {len(cat_indices)}  (fed to embeddings)")
print(f"    cat_indices        = {cat_indices}")
print(f"    cat_vocab_sizes    = {cat_vocab_sizes}")

# ── 3. Preprocess → DataLoaders + class weights ──
loaders, preprocessor, class_weights = prepare_data(
    X, y, seq_len=10, stride=5, batch_size=128
)

# ── 4. Build model ────────────────────────────────────
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nBuilding model using {device} .......")
model = NetworkTransformer(
    num_numeric=num_numeric,
    cat_indices=cat_indices,
    cat_vocab_sizes=cat_vocab_sizes,
    num_classes=NUM_CLASSES,
    d_model=128,
    nhead=8,
    num_layers=3,
    dim_feedforward=512,
    cat_embed_dim=16,
    dropout=0.1,
)

# ── 5. Train ──────────────────────────────────────────
trainer = Trainer(model, class_weights, device, lr=1e-3)
trainer.train(
    loaders['train'],
    loaders['val'],
    num_epochs=100,
    early_stop_patience=15,
)
trainer.plot_history()

# ── 6. Evaluate on test set ───────────────────────────
#   Load the best checkpoint (saved during training by val accuracy).
model.load_state_dict(
    torch.load('./checkpoints/best_model.pt', map_location=device)
)
report, cm = trainer.evaluate(loaders['test'])
print("\n" + report)
trainer.plot_confusion_matrix(cm)

end_time = time.time()
# Calculate elapsed time
elapsed = end_time - start_time
print(f"\n{'='*58}")
print(f"Time taken: {elapsed:.4f} seconds")
print(f"{'='*58}")