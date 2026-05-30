#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main parameterizable program to analyze all saved experiments.
This script loads, visualizes, and compares all model experiments saved as pickle files.

Usage:
    python main_analyze_experiments.py --model DL
    python main_analyze_experiments.py --model all
    python main_analyze_experiments.py --list
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add Article directory to path for imports
ensemble_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ensemble_path)

try:
    from src.config.Config import save_model_path
    from src.modules.save_model import load_experiment
except ImportError as e:
    print(f"Error: Could not import required modules. Make sure Article/src exists.")
    print(f"Details: {e}")
    sys.exit(1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ExperimentAnalyzer:
    """Analyze and visualize saved experiments"""
    
    def __init__(self, save_path="/home/akachat/tf_env/Ensemble_CNN_Melanoma/Models/"):
        self.save_path = save_path
        self.available_models = self._get_available_models()
    
    def _get_available_models(self):
        """Get list of available pickle files"""
        pkl_files = list(Path(self.save_path).glob("*.pkl"))
        return sorted([f.stem for f in pkl_files])
    
    def list_experiments(self):
        """List all available experiments"""
        print("=" * 80)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 80)
        for i, model in enumerate(self.available_models, 1):
            pkl_file = os.path.join(self.save_path, f"{model}.pkl")
            size_mb = os.path.getsize(pkl_file) / (1024**2)
            print(f"{i:2d}. {model:40s} ({size_mb:6.2f} MB)")
        print("=" * 80)
        return self.available_models
    
    def load_experiment(self, model_name):
        """Load a single experiment"""
        print(f"\n{'='*80}")
        print(f"Loading: {model_name}")
        print(f"{'='*80}")
        
        try:
            exp = load_experiment(model_name)
            if exp is None:
                print(f"✗ Error: Could not load {model_name}")
                return None
            
            experiment_data = exp.get("experiment_data", {})
            models = experiment_data.get("models", {})
            metrics = experiment_data.get("metrics", {})
            history = experiment_data.get("history", {})
            
            print(f"✓ Successfully loaded {model_name}")
            print(f"  Models: {len(models)} - {list(models.keys())}")
            print(f"  Metrics: {len(metrics)} classifiers")
            print(f"  History: {type(history).__name__}")
            
            return {
                'name': exp.get('name', model_name),
                'models': models,
                'metrics': metrics,
                'history': history
            }
        except Exception as e:
            print(f"✗ Error loading {model_name}: {str(e)}")
            return None
    
    def plot_confusion_matrices(self, exp_data, model_name):
        """Plot confusion matrices for all classifiers"""
        metrics = exp_data.get('metrics', {})
        
        if not metrics:
            print("  No metrics available for confusion matrices")
            return
        
        print(f"\n  Plotting {len(metrics)} confusion matrices...")
        
        for clf_name, metric_dict in metrics.items():
            if "y_true" not in metric_dict or "y_pred" not in metric_dict:
                continue
            
            y_true = metric_dict["y_true"]
            y_pred = metric_dict["y_pred"]
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix - {model_name}\n{clf_name}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.show()
    
    def plot_metrics_comparison(self, exp_data, model_name):
        """Plot metrics comparison for all classifiers"""
        metrics = exp_data.get('metrics', {})
        
        if not metrics:
            print("  No metrics available")
            return
        
        print(f"\n  Plotting metrics for {len(metrics)} classifiers...")
        
        for clf_name, metric_dict in metrics.items():
            # Filter scalar metrics
            filtered_metrics = {
                k: v for k, v in metric_dict.items()
                if np.isscalar(v) and k not in ['y_true', 'y_pred']
            }
            
            if not filtered_metrics:
                continue
            
            metric_names = list(filtered_metrics.keys())
            metric_values = list(filtered_metrics.values())
            
            plt.figure(figsize=(10, 5))
            bars = plt.bar(metric_names, metric_values, color='steelblue', alpha=0.7)
            plt.ylim(0, 1)
            plt.title(f"Performance Metrics - {model_name}\n{clf_name}", fontsize=12, fontweight='bold')
            plt.ylabel("Score")
            plt.xlabel("Metrics")
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, metric_values)):
                plt.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", 
                        ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.show()
    
    def plot_training_history(self, exp_data, model_name):
        """Plot training history"""
        history = exp_data.get('history', {})
        
        # Check if history has data
        has_history = False
        history_dict = None
        
        if history:
            if isinstance(history, dict) and "history" in history:
                has_history = True
                history_dict = history.history
            elif hasattr(history, 'history') and isinstance(history.history, dict):
                has_history = True
                history_dict = history.history
        
        if not has_history or not history_dict or "accuracy" not in history_dict:
            print("  No training history available (expected for ML-only models)")
            return
        
        print(f"\n  Plotting training history ({len(history_dict['accuracy'])} epochs)...")
        
        train_acc = history_dict.get("accuracy", [])
        val_acc = history_dict.get("val_accuracy", [])
        train_loss = history_dict.get("loss", [])
        val_loss = history_dict.get("val_loss", [])
        epochs_range = range(1, len(train_acc) + 1)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(epochs_range, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        if val_acc:
            axes[0].plot(epochs_range, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        axes[0].set_title(f'Accuracy - {model_name}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(epochs_range, train_loss, 'b-', label='Training Loss', linewidth=2)
        if val_loss:
            axes[1].plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
        axes[1].set_title(f'Loss - {model_name}', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self, exp_data, model_name):
        """Print experiment summary"""
        print(f"\n{'='*80}")
        print(f"EXPERIMENT SUMMARY: {model_name}")
        print(f"{'='*80}")
        
        models = exp_data.get('models', {})
        metrics = exp_data.get('metrics', {})
        history = exp_data.get('history', {})
        
        print(f"\n📦 MODELS ({len(models)}):")
        for i, model_key in enumerate(models.keys(), 1):
            print(f"   {i}. {model_key}")
        
        print(f"\n📊 METRICS ({len(metrics)} classifiers):")
        for clf_name, metric_dict in metrics.items():
            acc = metric_dict.get('accuracy', 0)
            auc = metric_dict.get('auc-roc', 0)
            sens = metric_dict.get('sensitivity', 0)
            spec = metric_dict.get('specificity', 0)
            print(f"   {clf_name}:")
            print(f"     Accuracy: {acc:.4f}, AUC: {auc:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
        
        print(f"\n📈 HISTORY:")
        if hasattr(history, 'history') and isinstance(history.history, dict):
            print(f"   Type: Keras History")
            print(f"   Keys: {list(history.history.keys())}")
            print(f"   Epochs: {len(history.history.get('accuracy', []))}")
        else:
            print(f"   No training history (ML-only model)")
        
        print(f"\n{'='*80}\n")
    
    def analyze(self, model_names, plot=True):
        """Analyze one or more experiments"""
        if not model_names:
            print("No models specified")
            return
        
        for model_name in model_names:
            exp_data = self.load_experiment(model_name)
            if exp_data is None:
                continue
            
            # Print summary
            self.print_summary(exp_data, model_name)
            
            # Generate visualizations
            if plot:
                print("Generating visualizations...")
                self.plot_confusion_matrices(exp_data, model_name)
                self.plot_metrics_comparison(exp_data, model_name)
                self.plot_training_history(exp_data, model_name)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Analyze saved ML/DL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_analyze_experiments.py --list
  python main_analyze_experiments.py --model DL
  python main_analyze_experiments.py --model DL_ML
  python main_analyze_experiments.py --model all
        """
    )
    
    parser.add_argument('--list', action='store_true', help='List all available experiments')
    parser.add_argument('--model', type=str, help='Model name to analyze (or "all")')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer()
    
    # List experiments
    if args.list:
        analyzer.list_experiments()
        return
    
    # Analyze specific model(s)
    if args.model:
        if args.model.lower() == 'all':
            models = analyzer.available_models
        else:
            models = [args.model]
        
        analyzer.analyze(models, plot=not args.no_plot)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
