# Transformer-based Network Anomaly Detection Prototype (learning and work in progress)

A simple prototype demonstrating transformer encoder for reconstruction-based anomaly detection on ToN-IoT network dataset .

## Goal
Use Ton_IOT dataset  (WIP)
Prototype sequential modeling of network traffic using PyTorch TransformerEncoder.  
Train autoencoder-style reconstruction only on benign traffic → detect anomalies via high reconstruction error.

## Dataset
ToN_IoT (network flows)  
- Official: https://research.unsw.edu.au/projects/toniot-datasets  


## Setup
```bash
pip install -r requirements.txt
```
## Future Work
- Extend to C-VAE / WAE-MMD
- Additional datasets (CIC-IoMT 2024)
- Latent space analysis
