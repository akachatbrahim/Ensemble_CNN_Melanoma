#!/usr/bin/env python3
"""
📊 EXTRACT AND GENERATE PLOTS FROM PKL FILES
==============================================

This script automatically:
1. Scans all pickle files in /Models/ directory
2. Generates visualizations (Confusion Matrices, Metrics, Training History)
3. Saves all plots to /plots directory

Plots generated:
- Confusion Matrices (heatmaps with counts and percentages)
- Performance Metrics (bar charts)
- Training History (accuracy and loss curves)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.Config import save_model_path, PLOTS_DIR,path_project
from src.utils.statistics import plot_model
from src.modules.save_model import load_experiment

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(path_project).resolve()
MODELS_DIR = PROJECT_ROOT / "Models"
PLOTS_OUTPUT_DIR = PROJECT_ROOT / "plots"

# Create output directory if it doesn't exist
PLOTS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subheader(title):
    """Print formatted subheader."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print('─' * 80)

def print_success(msg):
    """Print success message."""
    print(f"✓ {msg}")

def print_error(msg):
    """Print error message."""
    print(f"✗ {msg}")

def print_info(msg):
    """Print info message."""
    print(f"ℹ {msg}")

def print_warning(msg):
    """Print warning message."""
    print(f"⚠ {msg}")

# ============================================================================
# PLOT EXTRACTION
# ============================================================================

class PlotExtractor:
    """Extract and generate plots from PKL files."""
    
    def __init__(self, models_dir=MODELS_DIR, output_dir=PLOTS_OUTPUT_DIR):
        """Initialize the extractor."""
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.results = {
            "total_files": 0,
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
    
    def find_pkl_files(self):
        """Find all PKL files in the models directory."""
        if not self.models_dir.exists():
            print_error(f"Models directory not found: {self.models_dir}")
            return []
        
        pkl_files = sorted(self.models_dir.glob("*.pkl"))
        return pkl_files
    
    def extract_plots_from_file(self, pkl_file):
        """
        Extract and generate plots from a single PKL file.
        
        Parameters:
        -----------
        pkl_file : Path
            Path to the PKL file
        
        Returns:
        --------
        dict : Status and generated files
        """
        result = {
            "file": pkl_file.name,
            "success": False,
            "plots_generated": [],
            "error": None
        }
        
        try:
            # Get model name without extension
            model_name = pkl_file.stem
            
            print_subheader(f"Processing: {model_name}")
            print_info(f"File: {pkl_file}")
            
            # Load experiment to check if valid
            exp = load_experiment(str(pkl_file).replace(".pkl", ""))
            
            if exp is None:
                raise Exception("Could not load experiment data")
            
            # Extract experiment data
            experiment_data = exp.get("experiment_data", {})
            metrics_dict = experiment_data.get("metrics", {})
            history = experiment_data.get("history", {})
            
            exp_name = exp.get('name', model_name)
            
            print_success(f"Loaded experiment: {exp_name}")
            
            # ========== GENERATE CONFUSION MATRICES ==========
            if metrics_dict:
                import numpy as np
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.metrics import confusion_matrix
                
                for model_name_inner, metrics in metrics_dict.items():
                    if "y_true" in metrics and "y_pred" in metrics:
                        y_true = metrics["y_true"]
                        y_pred = metrics["y_pred"]
                        
                        # Compute confusion matrix
                        cm = confusion_matrix(y_true, y_pred)
                        cm_percent = cm.astype(float) / cm.sum() * 100
                        
                        # Create annotation labels
                        labels = np.empty_like(cm, dtype=object)
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                labels[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
                        
                        # Plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(
                            cm, 
                            annot=labels,
                            fmt="",
                            cmap="Blues",
                            cbar_kws={"label": "Count"},
                            xticklabels=["Benign", "Malignant"],
                            yticklabels=["Benign", "Malignant"],
                            ax=ax
                        )
                        
                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("True Label")
                        plt.title(f"Confusion Matrix - {model_name_inner}")
                        plt.tight_layout()
                        
                        # Save
                        plot_file = self.output_dir / f"{exp_name}_{model_name_inner}_ConfusionMatrix.png"
                        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print_success(f"Generated: {plot_file.name}")
                        result["plots_generated"].append(plot_file.name)
            
            # ========== GENERATE METRICS COMPARISON ==========
            if metrics_dict:
                import numpy as np
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                for model_name_inner, metrics in metrics_dict.items():
                    filtered_metrics = {
                        k: v for k, v in metrics.items()
                        if np.isscalar(v) and k not in ['y_true', 'y_pred']
                    }
                    
                    if filtered_metrics:
                        metric_names = list(filtered_metrics.keys())
                        metric_values = list(filtered_metrics.values())
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.bar(metric_names, metric_values, color='steelblue')
                        ax.set_ylim(0, 1)
                        ax.set_title(f"Performance Metrics - {model_name_inner}", fontsize=14, fontweight='bold')
                        ax.set_ylabel("Score")
                        plt.xticks(rotation=45, ha='right')
                        
                        # Display values on top of bars
                        for i, v in enumerate(metric_values):
                            ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)
                        
                        plt.tight_layout()
                        
                        # Save
                        plot_file = self.output_dir / f"{exp_name}_{model_name_inner}_Metrics.png"
                        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        print_success(f"Generated: {plot_file.name}")
                        result["plots_generated"].append(plot_file.name)
            
            # ========== GENERATE TRAINING HISTORY ==========
            has_history = False
            history_dict = {}
            
            if history:
                if isinstance(history, dict) and "history" in history:
                    has_history = True
                    history_dict = history.get("history", {})
                elif hasattr(history, 'history') and isinstance(history.history, dict):
                    has_history = True
                    history_dict = history.history
            
            if has_history and history_dict and "accuracy" in history_dict:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
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
                axes[0].set_title(f'Accuracy - {exp_name}', fontsize=12, fontweight='bold')
                axes[0].set_xlabel('Epochs')
                axes[0].set_ylabel('Accuracy')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Loss plot
                axes[1].plot(epochs_range, train_loss, 'b-', label='Training Loss', linewidth=2)
                if val_loss:
                    axes[1].plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
                axes[1].set_title(f'Loss - {exp_name}', fontsize=12, fontweight='bold')
                axes[1].set_xlabel('Epochs')
                axes[1].set_ylabel('Loss')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save
                plot_file = self.output_dir / f"{exp_name}_TrainingHistory.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print_success(f"Generated: {plot_file.name}")
                result["plots_generated"].append(plot_file.name)
            else:
                if metrics_dict:
                    print_info("No training history (ML-only model or history not saved)")
            
            result["success"] = True
            print_success(f"✓ Processed successfully")
            return result
        
        except Exception as e:
            result["error"] = str(e)
            print_error(f"Exception: {str(e)}")
            return result
    
    def extract_all(self):
        """Extract plots from all PKL files."""
        print_header("🎨 EXTRACTING PLOTS FROM PKL FILES")
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find PKL files
        pkl_files = self.find_pkl_files()
        self.results["total_files"] = len(pkl_files)
        
        if not pkl_files:
            print_warning("No PKL files found in Models directory")
            return
        
        print_info(f"Found {len(pkl_files)} PKL file(s) to process")
        print_info(f"Output directory: {self.output_dir}")
        print()
        
        # Process each file
        for idx, pkl_file in enumerate(pkl_files, 1):
            print(f"\n[{idx}/{len(pkl_files)}]")
            
            result = self.extract_plots_from_file(pkl_file)
            
            self.results["processed"] += 1
            if result["success"]:
                self.results["successful"] += 1
            else:
                self.results["failed"] += 1
                self.results["errors"].append(result["error"])
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print extraction summary."""
        print_header("📊 EXTRACTION SUMMARY")
        
        print(f"\n  Total files found: {self.results['total_files']}")
        print(f"  ✓ Successfully processed: {self.results['successful']}")
        print(f"  ✗ Failed: {self.results['failed']}")
        
        if self.results["errors"]:
            print_warning(f"\n  Errors encountered:")
            for error in self.results["errors"]:
                print(f"    - {error}")
        
        print_info(f"\nAll plots saved to: {self.output_dir}")
        
        # List generated plots
        plot_files = sorted(self.output_dir.glob("*.png"))
        if plot_files:
            print_info(f"\n📁 Generated {len(plot_files)} plot files:")
            for plot_file in plot_files:
                size_kb = plot_file.stat().st_size / 1024
                print(f"    - {plot_file.name} ({size_kb:.1f} KB)")
        else:
            print_warning("No plot files generated")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    try:
        # Create extractor
        extractor = PlotExtractor(models_dir=MODELS_DIR, output_dir=PLOTS_OUTPUT_DIR)
        
        # Extract all plots
        extractor.extract_all()
        
        print_header("✅ EXTRACTION COMPLETED")
        print_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Extraction interrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
