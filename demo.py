"""
Quick demonstration for running with synthetic data.
Runs a complete pipeline from data generation to evaluation.
"""

import torch
import numpy as np
from pathlib import Path

# Import modules
from src.load_data import ToNIoTDataLoader
from src.preprocess import NetworkTrafficPreprocessor, prepare_dataloaders
from src.model import TransformerCVAE
from src.train import Trainer
from src.evaluate import AnomalyDetectionEvaluator


def run_demo(n_samples=15000, num_epochs=20, save_results=True):
    """
    Run complete anomaly detection pipeline.
    
    Args:
        n_samples: Number of samples to generate
        num_epochs: Number of training epochs
        save_results: Whether to save results
    """
    print("="*70)
    print("TRANSFORMER-CVAE NETWORK ANOMALY DETECTION - DEMO")
    print("="*70)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 1. Load/Generate Data
    print("\n" + "="*70)
    print("STEP 1: Data Loading")
    print("="*70)
    
    loader = ToNIoTDataLoader()
    X, y = loader.load_data(use_synthetic=True, n_samples=n_samples)
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(X):,}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Normal samples: {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"  Anomaly samples: {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    
    # 2. Preprocess Data
    print("\n" + "="*70)
    print("STEP 2: Data Preprocessing")
    print("="*70)
    
    preprocessor = NetworkTrafficPreprocessor(sequence_length=10, stride=5)
    X_seq, y_seq = preprocessor.fit_transform(X, y)
    
    print(f"\nSequence Statistics:")
    print(f"  Total sequences: {len(X_seq):,}")
    print(f"  Sequence shape: {X_seq.shape}")
    print(f"  Normal sequences: {(y_seq==0).sum():,}")
    print(f"  Anomaly sequences: {(y_seq==1).sum():,}")
    
    # 3. Create DataLoaders
    print("\nCreating train/val/test splits...")
    dataloaders = prepare_dataloaders(X_seq, y_seq, batch_size=64)
    
    # 4. Initialize Model
    print("\n" + "="*70)
    print("STEP 3: Model Initialization")
    print("="*70)
    
    model = TransformerCVAE(
        input_dim=X.shape[1],
        seq_len=10,
        d_model=128,
        nhead=8,
        num_layers=3,
        latent_dim=64,
        vae_hidden_dim=256
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"\n  Components:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"    - {name}: {params:,} parameters")
    
    # 5. Train Model
    print("\n" + "="*70)
    print("STEP 4: Model Training")
    print("="*70)
    
    trainer = Trainer(model, device=device, learning_rate=1e-3, beta=0.5)
    
    history = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=num_epochs,
        early_stopping_patience=7,
        save_dir='./checkpoints'
    )
    
    # Plot training history
    if save_results:
        print("\nSaving training history plot...")
        Path('./results').mkdir(exist_ok=True, parents=True)
        trainer.plot_history(save_path='./results/training_history.png')
    
    # 6. Evaluate Model
    print("\n" + "="*70)
    print("STEP 5: Model Evaluation")
    print("="*70)
    
    evaluator = AnomalyDetectionEvaluator(model, device=device)
    
    print("\nEvaluating on test set...")
    metrics, errors, labels, predictions = evaluator.evaluate(dataloaders['test'])
    
    # Print detailed report
    evaluator.print_evaluation_report(metrics, labels, predictions)
    
    # Save results
    if save_results:
        print("\nSaving evaluation results...")
        
        evaluator.plot_evaluation_results(
            errors, labels, predictions,
            metrics['threshold'],
            save_dir='./results'
        )
        
        evaluator.save_results(metrics, './results/metrics.json')
    
    # 7. Summary
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nKey Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if save_results:
        print(f"\nResults saved to:")
        print(f"  - Model checkpoint: ./checkpoints/best_model.pt")
        print(f"  - Training history: ./results/training_history.png")
        print(f"  - Evaluation plots: ./results/evaluation_results.png")
        print(f"  - Metrics JSON: ./results/metrics.json")
    
    print("\n" + "="*70)
    
    return model, metrics, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer-CVAE Anomaly Detection Demo')
    parser.add_argument('--samples', type=int, default=15000,
                       help='Number of samples to generate (default: 15000)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    # Run demo
    model, metrics, history = run_demo(
        n_samples=args.samples,
        num_epochs=args.epochs,
        save_results=not args.no_save
    )
    
    print("\nDemo completed! We can now:")
    print("  1. Check the results in ./results/ directory")
    print("  2. Load the trained model from ./checkpoints/")
    print("  3. Explore interactively with notebooks/exploration.ipynb")
    print("  4. Adapt the code for your own datasets")