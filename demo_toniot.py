"""
Demo script specifically for running with real ToN-IoT dataset.
Use this after downloading the ToN-IoT dataset.
"""

import sys
from pathlib import Path

# Check if data file exists
data_file = Path('./data/Train_Test_Network.csv')
if not data_file.exists():
    print("=" * 70)
    print("ToN-IoT DATASET NOT FOUND")
    print("=" * 70)
    print("\nThe ToN-IoT dataset file is not in the data/ directory.")
    print("\nTo use real data:")
    print("1. Download from: https://cloudstor.aarnet.edu.au/plus/s/ds5zW91vdgjEj9i")
    print("2. Place 'Train_Test_Network.csv' in the data/ directory")
    print("3. Run this script again")
    print("\nAlternatively, run the regular demo with synthetic data:")
    print("  python demo.py")
    print("=" * 70)
    sys.exit(1)

# If we get here, data file exists - run demo with real data
print("=" * 70)
print("RUNNING DEMO WITH REAL TON-IoT DATASET")
print("=" * 70)
print()

from src.load_data import ToNIoTDataLoader
from src.preprocess import NetworkTrafficPreprocessor, prepare_dataloaders
from src.model import TransformerCVAE
from src.train import Trainer
from src.evaluate import AnomalyDetectionEvaluator
import torch

def run_toniot_demo(filename='Network_dataset_1.csv', n_samples=30000, num_epochs=25):
    """Run demo with real ToN-IoT data."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # 1. Load REAL ToN-IoT data
    print("=" * 70)
    print("STEP 1: Loading Real ToN-IoT Dataset")
    print("=" * 70)
    
    loader = ToNIoTDataLoader(data_dir='./data')
    X, y = loader.load_data(
        filename=filename,
        use_synthetic=False,  # Use REAL data
        n_samples=n_samples
    )
    
    # 2. Preprocess
    print("\n" + "=" * 70)
    print("STEP 2: Preprocessing Real Data")
    print("=" * 70)
    
    preprocessor = NetworkTrafficPreprocessor(sequence_length=10, stride=5)
    X_seq, y_seq = preprocessor.fit_transform(X, y)
    
    dataloaders = prepare_dataloaders(X_seq, y_seq, batch_size=64)
    
    # 3. Initialize Model
    print("\n" + "=" * 70)
    print("STEP 3: Model Initialization")
    print("=" * 70)
    
    model = TransformerCVAE(
        input_dim=X.shape[1],
        seq_len=10,
        d_model=128,
        nhead=8,
        num_layers=3,
        latent_dim=64
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Train
    print("\n" + "=" * 70)
    print("STEP 4: Training on Real ToN-IoT Data")
    print("=" * 70)
    
    trainer = Trainer(model, device=device, learning_rate=1e-3, beta=0.5)
    
    Path('./results').mkdir(exist_ok=True, parents=True)
    
    history = trainer.train(
        dataloaders['train'],
        dataloaders['val'],
        num_epochs=num_epochs,
        early_stopping_patience=10,
        save_dir='./checkpoints'
    )
    
    # Save training plot
    trainer.plot_history(save_path='./results/toniot_training_history.png')
    
    # 5. Evaluate
    print("\n" + "=" * 70)
    print("STEP 5: Evaluation on Real Data")
    print("=" * 70)
    
    evaluator = AnomalyDetectionEvaluator(model, device=device)
    metrics, errors, labels, predictions = evaluator.evaluate(dataloaders['test'])
    
    evaluator.print_evaluation_report(metrics, labels, predictions)
    
    # Save results
    evaluator.plot_evaluation_results(
        errors, labels, predictions,
        metrics['threshold'],
        save_dir='./results'
    )
    
    evaluator.save_results(metrics, './results/toniot_metrics.json')
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("REAL DATA DEMO COMPLETED!")
    print("=" * 70)
    
    print(f"\nPerformance on Real ToN-IoT Data:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\nResults saved:")
    print(f"  - Model: ./checkpoints/best_model.pt")
    print(f"  - Training plot: ./results/toniot_training_history.png")
    print(f"  - Evaluation plots: ./results/evaluation_results.png")
    print(f"  - Metrics: ./results/toniot_metrics.json")
    
    print("\n" + "=" * 70)
    
    return model, metrics, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run demo with single ToN-IoT dataset file')
    parser.add_argument('--filename', type=str, default='Train_Test_Network.csv',
                       help='ToN-IoT CSV filename in data/ directory (default: Train_Test_Network.csv)')
    parser.add_argument('--samples', type=int, default=50000,
                       help='Number of samples to load (default: 50000)')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of training epochs (default: 25)')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  File: {args.filename}")
    print(f"  Samples: {args.samples:,}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Dataset: Single ToN-IoT file\n")
    
    model, metrics, history = run_toniot_demo(
        filename=args.filename,
        n_samples=args.samples,
        num_epochs=args.epochs
    )
    
    print("\nSuccessfully trained on real ToN-IoT dataset!")
