# Transformer-based Network Anomaly Detection Prototype

A work-in-progress prototype demonstrating transformer-based modeling for network anomaly detection using the ToN-IoT dataset. This project includes a complete end-to-end pipeline for data preprocessing, model training, inference, and evaluation.

---

## Current Prototype

- **Data Pipeline:** Loads and preprocesses ToN-IoT network flows, including numeric, categorical, and boolean features.  
- **Model:** Transformer Encoder classifier for 10 network traffic types (normal + 9 attacks).  
- **Evaluation Metrics:**  
  - ROC Curves  
  - Response Time  
  - Feature Importance (Mutual Information)  
  - PCA (Feature Reduction)  
  - Classifier Fairness (Equalized Odds)  
  - K-Fold Confusion Matrices  
- **Inference:** Complete pipeline from raw flows to predicted classes and probabilities.

> ⚠️ Note: The current high accuracy is mainly due to distinct feature signatures in the dataset. The Transformer is learning class-level centroids rather than deep sequential patterns. This prototype is a stepping stone toward a fully research-grade anomaly detection framework.

---

## Dataset

- **ToN-IoT (network flows)** – [Official link](https://research.unsw.edu.au/projects/toniot-datasets)  

---

## Setup

```bash
pip install -r requirements.txt
python demo.py
```
##### The long-term vision of this project goes beyond the current prototype. Planned directions include:
- Implement Conditional Variational Autoencoder (C-VAE) or Wasserstein Autoencoder with MMD (WAE-MMD) for sequence reconstruction.
- Move from class-wise sequences to chronological, mixed-class sliding windows to capture real temporal dependencies.
- Extend research to other IoT/Network datasets like CIC-IoMT 2024.
- Refine the model into a publication-quality study.