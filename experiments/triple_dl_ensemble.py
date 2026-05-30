import os
import tensorflow as tf
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import (
    patience, min_delta, epochs, learning_rate, weight_decay, learning_rate_one, weight_decay_one,path_project,
    learning_rate_two, weight_decay_two, learning_rate_three, weight_decay_three, input_shape, num_classes, train_dir, test_dir
)

# Configure environment for progress bars display
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from src.modules.Dataset import load_dataset,Balanced
from src.modules.save_model import save_experiment
from src.modules.Evaluate import evaluate_model
from src.modules.Model import build_model, DL_MODELS, load_model_and_preprocess
from src.utils.parameter_loader import load_model_parameters, get_model_parameters_for_ids
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from src.modules.memory_optimized_test import test_memory_optimized
import tensorflow.keras.applications as tfa
from pathlib import Path

# ======================================================
# Memory optimization
# ======================================================
test_memory_optimized()

# ============================================================================
# LOAD MODELS AND PARAMETERS FROM CONFIGURATION FILE
# ============================================================================

print("\n" + "="*80)
print("  🔬 TRIPLE DL ENSEMBLE - LOAD MODELS FROM CONFIGURATION")
print("="*80 + "\n")

# Try to load model selection from configuration file
PROJECT_ROOT = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma")
config_file = PROJECT_ROOT / "current_model_parameters.json"

selected_choices = []
model_names = []

if config_file.exists():
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    selected_models = config_data.get('selected_models', [])
    
    if len(selected_models) >= 3:
        # Use the first 3 selected models
        selected_choices = selected_models[:3]
        
        for choice in selected_choices:
            model_names.append(DL_MODELS[choice]['name'])
        
        print(f"✓ Loaded configuration from: {config_file}")
        for i, name in enumerate(model_names, 1):
            print(f"  Model {i}: {name}")
        print()

# If no config file or insufficient models, fallback to defaults
if not selected_choices:
    print("⚠️  No configuration file found or insufficient models")
    print("Using default models: ResNet50, EfficientNetV2B0, DenseNet169\n")
    selected_choices = ["1", "2", "3"]
    for choice in selected_choices:
        model_names.append(DL_MODELS[choice]['name'])

# ============================================================================
# USE FIXED PARAMETERS FROM CONFIG.PY (learning_rate_one, learning_rate_two, learning_rate_three, etc.)
# ============================================================================

print(f"{'─'*80}")
print(f"📊 Using FIXED parameters from Config.py:")
print(f"  Model 1 ({model_names[0]}): LR={learning_rate_one}, WD={weight_decay_one}")
print(f"  Model 2 ({model_names[1]}): LR={learning_rate_two}, WD={weight_decay_two}")
print(f"  Model 3 ({model_names[2]}): LR={learning_rate_three}, WD={weight_decay_three}")
print(f"{'─'*80}\n")

model_params = [
    (learning_rate_one, weight_decay_one),
    (learning_rate_two, weight_decay_two),
    (learning_rate_three, weight_decay_three)
]

# ======================================================
# Dataset
# ======================================================
train_ds, val_ds, test_ds = load_dataset(
    train_dir, test_dir,
    augment=True,
    deeplearning=False,
    preprocess_fn=None
)

class_weights = Balanced(train_ds)
# ======================================================
# Load and Build Selected Models
# ======================================================
models = []
model_configs = []  # Track configuration for each model

for idx, choice in enumerate(selected_choices):
    model_info = DL_MODELS[choice]
    model_name = model_info['name']
    
    # Get parameters for this position (FIXED for this model)
    lr, wd = model_params[idx]
    
    # Store configuration
    config = {
        'name': model_name,
        'learning_rate': lr,
        'weight_decay': wd,
        'position': idx + 1
    }
    model_configs.append(config)
    
    # Handle custom CNN separately
    if model_info.get('type') == 'custom':
        from src.modules.Model import build_model_cnn
        print(f"✓ {model_name} loaded successfully")
        model = build_model_cnn()
        models.append(model)
        continue
    
    try:
        deep_learning, preprocess_fn = load_model_and_preprocess(model_info)
        if deep_learning is None:
            print(f"⚠ Skipping {model_name}")
            continue
        
        print(f"✓ {model_name} loaded successfully")
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")
        continue
    
    # Build the model with its assigned parameters
    print(f"  └─ LR: {lr}, WD: {wd}")
    model = build_model(deep_learning, lr, wd)
    print(f"  ✓ Model compiled with assigned parameters\n")
    models.append(model)

# PARAMETRIZABLE MODEL NAMES (AUTO)
name_model = "_".join(model_names) + "_ENSEMBLE"

print(f"\n✓ Building ensemble: {name_model}\n")

# Initialize history tracking
histories = {}

# ======================================================
# Head Training - Each model with its own parameters
# ======================================================
print("\n" + "="*80)
print("  🏋️  TRAINING MODELS (Each with fixed LR and WD)")
print("="*80 + "\n")

for i, (model, config) in enumerate(zip(models, model_configs)):
    print(f"Training [{i+1}]: {config['name']}")
    print(f"  Parameters (FIXED throughout training):")
    print(f"    └─ Learning Rate: {config['learning_rate']}")
    print(f"    └─ Weight Decay:  {config['weight_decay']}\n")

    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode="max",
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Store Training History
    histories[f"{model_names[i]}"] = history.history
    
    print(f"✓ Training completed for {config['name']}\n")

# ======================================================
# ENSEMBLE — Average Predictions (Better than concat)
# ======================================================
# Create ensemble model with single input feeding all 3 models
# ======================================================
single_input = tf.keras.Input(shape=input_shape)
outputs = [model(single_input) for model in models]
final_output = tf.keras.layers.Average()(outputs)

combined_model = tf.keras.models.Model(
    inputs=single_input,
    outputs=final_output
)

# ======================================================
# Evaluation
# ======================================================
ml_models = {model_names[i]: models[i] for i in range(len(models))}
ml_models["Ensemble_Model"] = combined_model

metrics_per_model = {}

# Evaluate each model individually
for model_name, model in ml_models.items():
    ml_models, metrics_per_model = evaluate_model(
        model,
        model_name,
        None,
        None,
        test_ds,
        ml_models
    )

# ======================================================
# Save
# ======================================================
experiment_data = {
    'models': ml_models,
    'metrics': metrics_per_model,
    'histories': histories
}

save_experiment(combined_model, name_model, experiment_data)