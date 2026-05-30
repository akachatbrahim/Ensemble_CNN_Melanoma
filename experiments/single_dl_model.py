import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Optimize memory
from src.modules.memory_optimized_test import test_memory_optimized
test_memory_optimized()
from src.config.Config import ( epochs, patience, min_delta, learning_rate_one, train_dir, test_dir, 
                                input_shape, num_classes, weight_decay_one,path_project
)

# Configure environment for progress bars display
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from src.modules.Model import build_model_cnn, DL_MODELS, build_model, load_model_and_preprocess
from src.modules.Dataset import load_dataset, Balanced
from src.modules.save_model import save_experiment
from src.modules.Evaluate import evaluate_model
import tensorflow.keras.applications as tfa


# ============================================================================
# SELECT DEEP LEARNING MODEL
# ============================================================================

print("\n" + "="*80)
print("  🔬 DEEP LEARNING MODEL SELECTION")
print("="*80 + "\n")

# Display options
print("📋 Available Deep Learning Models:\n")
for key, info in DL_MODELS.items():
    print(f"  {key:2}. {info['name']:<25} - {info['description']}")

# Get user choice
print("\n")
choice = input("Select model (1-13) [default: 1 (ResNet50)]: ").strip()

if choice not in DL_MODELS:
    choice = "1"  # Default to ResNet50

selected_model_info = DL_MODELS[choice]
model_name = selected_model_info['name']
name_model = model_name

print(f"\n✓ Selected: {model_name}")

# ============================================================================
# BUILD MODEL BASED ON TYPE
# ============================================================================

print(f"  Loading {model_name}...")

if selected_model_info['type'] == 'custom':
    # Custom CNN
    print(f"  Building custom CNN...")
    train_ds, val_ds, test_ds = load_dataset(train_dir, test_dir, augment=True, deeplearning=False, preprocess_fn=None)
    model = build_model_cnn()
    print(f"✓ Custom CNN built successfully\n")

else:
    # Standard Keras models (ResNet50, MobileNetV2, EfficientNet, DenseNet, VGG16, VGG19, ConvNeXtSmall)
    try:
        deep_learning, deep_learning_pre = load_model_and_preprocess(selected_model_info)
        print(f"✓ {model_name} loaded successfully\n")
        
        train_ds, val_ds, test_ds = load_dataset(train_dir, test_dir, augment=True, deeplearning=True, preprocess_fn=deep_learning_pre)
        model = build_model(deep_learning,learning_rate_one, weight_decay_one)
        
    except Exception as e:
        print(f"✗ Error loading {model_name}: {e}")
        print(f"  Falling back to ResNet50...")
        model_name = "ResNet50"
        name_model = model_name
        from tensorflow.keras.applications.resnet50 import ResNet50 as deep_learning
        from tensorflow.keras.applications.resnet50 import preprocess_input as deep_learning_pre
        train_ds, val_ds, test_ds = load_dataset(train_dir, test_dir, augment=True, deeplearning=True, preprocess_fn=deep_learning_pre)
        model = build_model(deep_learning,learning_rate_one, weight_decay_one)


# TRAIN MODEL 
print(f"Training {name_model}...")

class_weights = Balanced(train_ds)

# EarlyStopping callback
early_stopping_phase1 = EarlyStopping(
    monitor='val_auc', 
    mode="max", 
    patience=patience,
    min_delta=min_delta, 
    restore_best_weights=True, 
    verbose=1
)

# Phase Training
print(f"Starting training ...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping_phase1],
    verbose=1
)

# ============================================================================
# EVALUATE AND SAVE
# ============================================================================
ml_models = {}
ml_models, metrics_model = evaluate_model(model, name_model, None, None, test_ds, ml_models)

# Save model, metrics, and training history to pickle
experiment_data = {
    'models': ml_models,
    'metrics': metrics_model,
    'history': history
}
save_experiment(model, name_model, experiment_data)