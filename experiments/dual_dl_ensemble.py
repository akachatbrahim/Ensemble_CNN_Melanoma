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
from src.modules.Evaluate import evaluate_model
from src.modules.save_model import save_experiment
from src.modules.Model import build_model, build_model_cnn, DL_MODELS,load_model_and_preprocess
from src.utils.parameter_loader import load_model_parameters
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from src.modules.memory_optimized_test import test_memory_optimized
import tensorflow.keras.applications as tfa
from pathlib import Path

#optimized configuration for memory-constrained systems
test_memory_optimized()

# ============================================================================
# LOAD MODELS FROM CONFIGURATION FILE (saved by main_project.py)
# ============================================================================

print("\n" + "="*80)
print("  🔬 DUAL DL ENSEMBLE - LOAD MODELS FROM CONFIGURATION")
print("="*80 + "\n")

# Try to load model selection and parameters from configuration file
PROJECT_ROOT = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma")
config_file = PROJECT_ROOT / "current_model_parameters.json"

model1_choice = None
model2_choice = None
model1_params = None
model2_params = None

if config_file.exists():
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    selected_models = config_data.get('selected_models', [])
    parameters = config_data.get('parameters', {})
    
    if len(selected_models) >= 2:
        # Use the first 2 selected models
        model1_choice = selected_models[0]
        model2_choice = selected_models[1]
        
        if model1_choice in parameters:
            model1_params = (parameters[model1_choice]['learning_rate'], 
                           parameters[model1_choice]['weight_decay'])
        
        if model2_choice in parameters:
            model2_params = (parameters[model2_choice]['learning_rate'], 
                           parameters[model2_choice]['weight_decay'])
        
        print(f"✓ Loaded configuration from: {config_file}")
        print(f"  Model 1: {DL_MODELS[model1_choice]['name']}")
        print(f"  Model 2: {DL_MODELS[model2_choice]['name']}\n")

# If no config file or insufficient models, use defaults
if model1_choice is None or model2_choice is None:
    print("⚠️  No configuration file found or insufficient models")
    print("Please select 2 models manually:\n")
    
    # Display available models
    print("📋 Available models:\n")
    for key in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
        model_name = DL_MODELS[key]['name']
        print(f"  {key}. {model_name}")
    print()
    
    # Get user choice for Model 1
    while True:
        model1_choice = input("Select Model 1 (enter number): ").strip()
        if model1_choice in DL_MODELS:
            break
        else:
            print(f"❌ Invalid choice. Please select a valid model number.")
    
    # Get user choice for Model 2 (must be different from Model 1)
    while True:
        model2_choice = input("Select Model 2 (enter number, must be different from Model 1): ").strip()
        if model2_choice in DL_MODELS:
            if model2_choice != model1_choice:
                break
            else:
                print(f"❌ Model 2 must be different from Model 1. Please choose another model.")
        else:
            print(f"❌ Invalid choice. Please select a valid model number.")

# Get model names
model1_name = DL_MODELS[model1_choice]['name']
model2_name = DL_MODELS[model2_choice]['name']

print(f"\n✓ Selected models: {model1_name} and {model2_name}")
print("="*80 + "\n")

# Display available models
print("\n📋 Available models:\n")
selected = [model1_choice, model2_choice]
for key in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
    model_name = DL_MODELS[key]['name']
    status = " ✓" if key in selected else ""
    print(f"  {key}. {model_name}{status}")
print()

# ============================================================================
# ASK USER FOR PARAMETERS
# ============================================================================

print(f"\n📊 Enter parameters for each model:\n")

# Model 1 parametersbuild_model
print(f"Paramètres pour {model1_name}:")
while True:
    lr1_input = input(f"  Learning Rate [default: 3e-4]: ").strip() or "3e-4"
    try:
        model1_lr = float(lr1_input)
        break
    except ValueError:
        print(f"Format invalide. Entrez un nombre (ex: 1e-4 ou 0.0001)")

while True:
    wd1_input = input(f"  Weight Decay [default: 1e-5]: ").strip() or "1e-5"
    try:
        model1_wd = float(wd1_input)
        break
    except ValueError:
        print(f"Format invalide. Entrez un nombre (ex: 1e-5 ou 0.00001)")

# Model 2 parameters
print(f"\nParamètres pour {model2_name}:")
while True:
    lr2_input = input(f"  Learning Rate [default: 3e-4]: ").strip() or "3e-4"
    try:
        model2_lr = float(lr2_input)
        break
    except ValueError:
        print(f"Format invalide. Entrez un nombre (ex: 1e-4 ou 0.0001)")

while True:
    wd2_input = input(f"  Weight Decay [default: 1e-5]: ").strip() or "1e-5"
    try:
        model2_wd = float(wd2_input)
        break
    except ValueError:
        print(f"Format invalide. Entrez un nombre (ex: 1e-5 ou 0.00001)")

print(f"{'─'*80}")
print(f"📊 Configuration sélectionnée:")
print(f"  Model 1 ({model1_name}): LR={model1_lr}, WD={model1_wd}")
print(f"  Model 2 ({model2_name}): LR={model2_lr}, WD={model2_wd}")
print(f"{'─'*80}\n")

# Create parameter tuples for later use
model1_params = (model1_lr, model1_wd)
model2_params = (model2_lr, model2_wd)

'''
# Helper function to load model and preprocess function
def load_model_and_preprocess(model_info):
    """Dynamically load a model and its preprocessing function with special handling for VGG and ConvNeXt"""
    import tensorflow.keras.applications as tfa
    
    if model_info.get('type') == 'custom':
        return build_model_cnn(), None
    
    module_name = model_info.get('module')
    model_class_name = model_info.get('model_class')
    preprocess_name = model_info.get('preprocess')
    model_name = model_info.get('name')
    
    try:
        # Special handling for VGG models
        if module_name in ['vgg16', 'vgg19']:
            try:
                model_module = getattr(tfa, module_name)
                Model = getattr(model_module, model_class_name)
                preprocess_module = getattr(tfa, preprocess_name)
                preprocess_fn = preprocess_module.preprocess_input
            except AttributeError:
                # Fallback for older TensorFlow versions
                if module_name == 'vgg16':
                    from tensorflow.keras.applications.vgg16 import VGG16 as Model
                    from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_fn
                elif module_name == 'vgg19':
                    from tensorflow.keras.applications.vgg19 import VGG19 as Model
                    from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_fn
            
            base_model = Model(weights="imagenet", include_top=False, input_shape=input_shape)
            return base_model, preprocess_fn
        
        # Special handling for ConvNeXt
        elif module_name == 'convnext':
            try:
                model_module = getattr(tfa, module_name)
                Model = getattr(model_module, model_class_name)
                preprocess_module = getattr(tfa, preprocess_name)
                preprocess_fn = preprocess_module.preprocess_input
                base_model = Model(weights="imagenet", include_top=False, input_shape=input_shape)
                return base_model, preprocess_fn
            except AttributeError:
                print(f"⚠ {model_name} not available in this TensorFlow version. Skipping.")
                return None, None
        
        # Standard handling for other models
        else:
            model_module = getattr(tfa, module_name)
            Model = getattr(model_module, model_class_name)
            preprocess_module = getattr(tfa, preprocess_name)
            preprocess_fn = preprocess_module.preprocess_input
            base_model = Model(weights="imagenet", include_top=False, input_shape=input_shape)
            return base_model, preprocess_fn
    
    except AttributeError as e:
        print(f"✗ Error loading {model_name}: {e}")
        return None, None
'''
# Load selected models
model_indices = [model1_choice, model2_choice]
model_params = [model1_params, model2_params]

selected_model_infos = []
model_bases = []
model_names = []
preprocess_fns = []

for idx in model_indices:
    model_info = DL_MODELS[idx]
    model_name = model_info['name']
    model_names.append(model_name)
    selected_model_infos.append(model_info)
    
    base_model, preprocess_fn = load_model_and_preprocess(model_info)
    if base_model is None:
        print(f"\n✗ Error loading {model_name}. Exiting.")
        exit(1)
    
    model_bases.append(base_model)
    preprocess_fns.append(preprocess_fn)
    print(f"✓ {model_name} loaded successfully")

print()

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_auc', mode="max", patience=patience, restore_best_weights=True)

# Define paths
train_ds, val_ds, test_ds = load_dataset(train_dir,test_dir, augment=True, deeplearning=False,preprocess_fn=None)
class_weights = Balanced(train_ds)

# Build models with their specific parameters
print("Building models with assigned parameters...\n")
models = []
model_configs = []  # Track config for each model

for i, (base_model, preprocess_fn) in enumerate(zip(model_bases, preprocess_fns)):
    lr, wd = model_params[i]
    model_name = model_names[i]
    
    # Store configuration
    config = {
        'name': model_name,
        'learning_rate': lr,
        'weight_decay': wd,
        'index': i
    }
    model_configs.append(config)
    
    print(f"  [{i+1}] {model_name}")
    print(f"      └─ LR: {lr}, WD: {wd}")
    
    # Build model with these specific parameters
    model = build_model(base_model, lr, wd)
    models.append(model)
    print(f"      ✓ Model compiled with assigned parameters\n")

print()

# PARAMETRIZABLE MODEL NAMES (AUTO)
name_model = "_".join(model_names) + "_ENSEMBLE"

histories = {}

# ======================================================
# Head Training - Each model with its own parameters
# ======================================================
print("="*80)
print("  🏋️  TRAINING MODELS (Each with fixed LR and WD)")
print("="*80 + "\n")

for i, (model, config) in enumerate(zip(models, model_configs)):
    print(f"Training [{i+1}]: {config['name']}")
    print(f"  Parameters (FIXED):")
    print(f"    └─ Learning Rate: {config['learning_rate']}")
    print(f"    └─ Weight Decay:  {config['weight_decay']}\n")

    # EarlyStopping callback
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
    # Store history
    histories[f"{model_names[i]}"] = history.history
    
    print(f"✓ Training completed for {config['name']}\n")

# ======================================================
# ENSEMBLE — Average Predictions (Better than concat)
# ======================================================
# Create ensemble model with single input feeding all 2 models
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
ml_models = {
    model1_name: models[0],
    model2_name: models[1],
    "Ensemble_Model": combined_model
}

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

# Save model, metrics, and training history to pickle
experiment_data = {
    'models': ml_models,
    'metrics': metrics_per_model,
    'history': histories
}
save_experiment(combined_model, name_model, experiment_data)
