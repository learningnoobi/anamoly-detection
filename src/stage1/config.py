import torch

D_HIDDEN = 256
D_Z      = 64
DROPOUT  = 0.1
EPOCHS   = 100
LR       = 1e-4
PATIENCE = 10
BETA_WARMUP = 20
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
