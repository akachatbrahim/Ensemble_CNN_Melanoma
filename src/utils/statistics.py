import numpy as np
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, "/home/akachat/tf_env/Ensemble_CNN_Melanoma")
from src.config.Config import save_model_path,PLOTS_DIR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import h5py
from src.modules.save_model import load_experiment
import os
from pathlib import Path

file_pkl = "EfficientNet_V2B2"

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def plot_model(filename):
    """
    Plot model evaluation results from pickle files.
    Supports both DL models (with training history) and ML models.
    
    Parameters:
    -----------
    filename : str
        Name of the experiment or full path to model without extension
        (will try .pkl first from Models/ directory)
    """
    import os
    
    # Handle both full paths and just experiment names
    if not filename.startswith("/") and not os.path.dirname(filename):
        # Just a name like "VGG19" - prepend Models path
        filename = os.path.join(save_model_path, filename)
    
    # Try pickle format first (new format)
    pkl_path = filename + ".pkl"
    
    if os.path.exists(pkl_path):
        # ========== LOAD PICKLE FILE ==========
        exp = load_experiment(filename)
        if exp is None:
            print(f"Error: Could not load experiment from {pkl_path}")
            return
            
        # Extract experiment data
        experiment_data = exp.get("experiment_data", {})
        models = experiment_data.get("models", {})
        metrics_dict = experiment_data.get("metrics", {})
        history = experiment_data.get("history", {})
        
        exp_name = exp.get('name', os.path.basename(filename))
        print(f"✓ Loaded experiment: {exp_name}")
        
        # ========== PLOT CONFUSION MATRICES ==========
        if metrics_dict:
            for model_name, metrics in metrics_dict.items():
                if "y_true" in metrics and "y_pred" in metrics:
                    y_true = metrics["y_true"]
                    y_pred = metrics["y_pred"]
                    sensitivity = metrics["sensitivity"]
                    specificity = metrics["specificity"]
                    accuracy = metrics["accuracy"]
                    precision = metrics["precision"]
                    auc_roc = metrics["auc-roc"]

               
                    # Compute confusion matrix
                    cm = confusion_matrix(y_true, y_pred)
            
                    # Calculate percentages
                    cm_percent = cm.astype(float) / cm.sum() * 100
                    
                    # Create annotation labels: "count\n(percent%)"
                    labels = np.empty_like(cm, dtype=object)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            labels[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"
                    
                    # Plot using seaborn heatmap for better control
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(
                        cm, 
                        annot=labels,
                        fmt="",
                        cmap="Blues",
                        cbar_kws={"label": "Count"},
                        xticklabels=labels,
                        yticklabels=labels,
                        ax=ax
                    )
                    
                    # Set labels
                    ax.set_xlabel("Predicted Label")
                    ax.set_ylabel("True Label")
                    ax.set_xticklabels(["Benign", "Malignant"])
                    ax.set_yticklabels(["Benign", "Malignant"])
                    plt.title(f"Confusion Matrix - {model_name}")
                    plt.tight_layout()
                    
                    # Save instead of show
                    plot_file = os.path.join(PLOTS_DIR, f"{exp_name}_{model_name}_cm.png")
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                    print(f"✓ Saved: {plot_file}")
                    plt.close()

        # ========== PLOT METRICS COMPARISON ==========
        if metrics_dict:
            for model_name, metrics in metrics_dict.items():
                # Keep only scalar numeric metrics (skip arrays like y_true, y_pred)
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
                    ax.set_title(f"Performance Metrics - {model_name}", fontsize=14, fontweight='bold')
                    ax.set_ylabel("Score")
                    plt.xticks(rotation=45, ha='right')
            
                    # Display values on top of bars
                    for i, v in enumerate(metric_values):
                        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)

                    plt.tight_layout()
                    
                    # Save instead of show
                    plot_file = os.path.join(PLOTS_DIR, f"{exp_name}_{model_name}_metrics.png")
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                    print(f"✓ Saved: {plot_file}")
                    plt.close()

        # ========== PLOT TRAINING HISTORY ==========
        # Check if history is available and has data
        has_history = False
        if history:
            if isinstance(history, dict) and "history" in history:
                # Keras History object
                has_history = True
                history_dict = history.history
            elif hasattr(history, 'history') and isinstance(history.history, dict):
                # Keras History object (check for attribute)
                has_history = True
                history_dict = history.history
        
        if has_history and history_dict and "accuracy" in history_dict:
            print(f"✓ Plotting training history ({len(history_dict['accuracy'])} epochs)")
            
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
            
            # Save instead of show
            plot_file = os.path.join(PLOTS_DIR, f"{exp_name}_training_history.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {plot_file}")
            plt.close()
        else:
            print(f"ℹ No training history available (ML-only models)")            
    else:
        print(f"✗ Error: No model file found!")
        print(f"   Expected: {pkl_path}")



if __name__ == "__main__":
    plot_model(save_model_path+file_pkl)