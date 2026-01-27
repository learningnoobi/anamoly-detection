"""
Evaluation script for anomaly detection model.
Includes metrics calculation, threshold optimization, and visualization.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import json

from .model import TransformerCVAE


class AnomalyDetectionEvaluator:
    """Evaluate anomaly detection performance."""
    
    def __init__(self, model: TransformerCVAE, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.threshold = None
        
    def compute_reconstruction_errors(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reconstruction errors for all samples.
        
        Returns:
            errors: Reconstruction error per sample
            labels: True labels
        """
        self.model.eval()
        
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                
                errors = self.model.get_reconstruction_error(batch_x)
                
                all_errors.append(errors.cpu().numpy())
                all_labels.append(batch_y.numpy())
        
        errors = np.concatenate(all_errors)
        labels = np.concatenate(all_labels)
        
        return errors, labels
    
    def find_optimal_threshold(self, errors: np.ndarray, labels: np.ndarray,
                               metric: str = 'f1') -> float:
        """
        Find optimal threshold for anomaly detection.
        
        Args:
            errors: Reconstruction errors
            labels: True labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
        
        Returns:
            Optimal threshold
        """
        # Try different percentiles as thresholds
        thresholds = np.percentile(errors, np.arange(50, 100, 1))
        
        best_score = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            predictions = (errors > threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(labels, predictions, zero_division=0)
            elif metric == 'precision':
                score = precision_score(labels, predictions, zero_division=0)
            elif metric == 'recall':
                score = recall_score(labels, predictions, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Optimal threshold ({metric}): {best_threshold:.6f} (score: {best_score:.4f})")
        
        return best_threshold
    
    def evaluate(self, dataloader, threshold: float = None) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            dataloader: Data to evaluate on
            threshold: Detection threshold (if None, uses stored threshold)
            
        Returns:
            Dictionary of metrics
        """
        errors, labels = self.compute_reconstruction_errors(dataloader)
        
        if threshold is None:
            if self.threshold is None:
                threshold = self.find_optimal_threshold(errors, labels)
            else:
                threshold = self.threshold
        
        predictions = (errors > threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'roc_auc': roc_auc_score(labels, errors),
            'threshold': threshold,
            'mean_error_normal': errors[labels == 0].mean(),
            'mean_error_anomaly': errors[labels == 1].mean(),
            'std_error_normal': errors[labels == 0].std(),
            'std_error_anomaly': errors[labels == 1].std()
        }
        
        return metrics, errors, labels, predictions
    
    def print_evaluation_report(self, metrics: Dict[str, float], 
                                labels: np.ndarray, predictions: np.ndarray):
        """Print detailed evaluation report."""
        print("\n" + "="*60)
        print("ANOMALY DETECTION EVALUATION REPORT")
        print("="*60)
        
        print(f"\nThreshold: {metrics['threshold']:.6f}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        print(f"\nReconstruction Error Statistics:")
        print(f"  Normal Traffic:    {metrics['mean_error_normal']:.6f} ± {metrics['std_error_normal']:.6f}")
        print(f"  Anomaly Traffic:   {metrics['mean_error_anomaly']:.6f} ± {metrics['std_error_anomaly']:.6f}")
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(labels, predictions)
        print(f"                Predicted")
        print(f"              Normal  Anomaly")
        print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:6d}")
        print(f"       Anomaly  {cm[1,0]:6d}  {cm[1,1]:6d}")
        
        print("\n" + "="*60)
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(labels, predictions, 
                                   target_names=['Normal', 'Anomaly'],
                                   digits=4))
    
    def plot_evaluation_results(self, errors: np.ndarray, labels: np.ndarray,
                                predictions: np.ndarray, threshold: float,
                                save_dir: str = None):
        """Create comprehensive evaluation plots."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Reconstruction error distribution
        ax1 = fig.add_subplot(gs[0, :2])
        normal_errors = errors[labels == 0]
        anomaly_errors = errors[labels == 1]
        
        bins = np.linspace(min(errors), max(errors), 50)
        ax1.hist(normal_errors, bins=bins, alpha=0.6, label='Normal', color='blue', density=True)
        ax1.hist(anomaly_errors, bins=bins, alpha=0.6, label='Anomaly', color='red', density=True)
        ax1.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Density')
        ax1.set_title('Reconstruction Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 2])
        fpr, tpr, _ = roc_curve(labels, errors)
        roc_auc = roc_auc_score(labels, errors)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        ax3 = fig.add_subplot(gs[1, 0])
        precision, recall, _ = precision_recall_curve(labels, errors)
        
        ax3.plot(recall, precision, color='purple', lw=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.grid(True, alpha=0.3)
        
        # 4. Confusion Matrix
        ax4 = fig.add_subplot(gs[1, 1])
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        ax4.set_ylabel('True Label')
        ax4.set_xlabel('Predicted Label')
        ax4.set_title('Confusion Matrix')
        
        # 5. Error scatter plot
        ax5 = fig.add_subplot(gs[1, 2])
        normal_idx = np.where(labels == 0)[0][:500]  # Sample for visibility
        anomaly_idx = np.where(labels == 1)[0][:500]
        
        ax5.scatter(normal_idx, errors[normal_idx], alpha=0.5, s=20, label='Normal', color='blue')
        ax5.scatter(anomaly_idx, errors[anomaly_idx], alpha=0.5, s=20, label='Anomaly', color='red')
        ax5.axhline(threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
        ax5.set_xlabel('Sample Index')
        ax5.set_ylabel('Reconstruction Error')
        ax5.set_title('Reconstruction Errors (Sample)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Box plot
        ax6 = fig.add_subplot(gs[2, :])
        data_to_plot = [normal_errors, anomaly_errors]
        bp = ax6.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], 
                        patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax6.axhline(threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
        ax6.set_ylabel('Reconstruction Error')
        ax6.set_title('Reconstruction Error by Class')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Anomaly Detection Evaluation Results', fontsize=16, y=0.995)
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / 'evaluation_results.png', dpi=300, bbox_inches='tight')
            print(f"Evaluation plots saved to {save_path / 'evaluation_results.png'}")
        else:
            plt.show()
    
    def save_results(self, metrics: Dict[str, float], save_path: str):
        """Save evaluation results to JSON."""
        # Convert NumPy types to Python native types for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"Results saved to {save_path}")


if __name__ == "__main__":
    # Example evaluation
    from load_data import ToNIoTDataLoader
    from preprocess import NetworkTrafficPreprocessor, prepare_dataloaders
    
    print("Loading data...")
    loader = ToNIoTDataLoader()
    X, y = loader.load_data(use_synthetic=True, n_samples=20000)
    
    print("\nPreprocessing...")
    preprocessor = NetworkTrafficPreprocessor(sequence_length=10, stride=5)
    X_seq, y_seq = preprocessor.fit_transform(X, y)
    
    dataloaders = prepare_dataloaders(X_seq, y_seq, batch_size=64)
    
    print("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TransformerCVAE(
        input_dim=X.shape[1],
        seq_len=10,
        d_model=128,
        nhead=8,
        num_layers=3,
        latent_dim=64
    )
    
    # Load trained weights if available
    try:
        checkpoint = torch.load('./checkpoints/best_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model weights")
    except FileNotFoundError:
        print("No trained model found, using random initialization for demo")
    
    # Evaluate
    evaluator = AnomalyDetectionEvaluator(model, device=device)
    
    print("\nEvaluating on test set...")
    metrics, errors, labels, predictions = evaluator.evaluate(dataloaders['test'])
    
    # Print report
    evaluator.print_evaluation_report(metrics, labels, predictions)
    
    # Plot results
    evaluator.plot_evaluation_results(errors, labels, predictions, 
                                     metrics['threshold'],
                                     save_dir='./results')
    
    # Save results
    evaluator.save_results(metrics, './results/metrics.json')