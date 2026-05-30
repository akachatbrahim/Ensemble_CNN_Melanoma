# -*- coding: utf-8 -*-   modified
"""
Created on Wed Oct  8 08:17:27 2025

@author: brahim
"""

# CNN  Binary Classification (Malignant vs Benign)

import os
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import ( patience,min_delta, threshold,train_dir,test_dir, input_shape,metrics_per_model,path_project,
                                ml_models, learning_rate_one, weight_decay_one, learning_rate_two, weight_decay_two
)

# Configure environment for progress bars display
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from src.modules.Model import build_model_cnn, DL_MODELS
from src.modules.save_model import save_experiment
from src.modules.Evaluate import evaluate_model
from src.modules.Dataset import load_dataset,Balanced
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Concatenate
import tensorflow.keras.applications as tfa
from src.config.Config import patience,min_delta, threshold,train_dir,test_dir, input_shape,metrics_per_model,ml_models, learning_rate_one, weight_decay_one, learning_rate_two, weight_decay_two
from src.utils.parameter_loader import load_model_parameters
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from src.modules.memory_optimized_test import test_memory_optimized

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#optimized configuration for memory-constrained systems
test_memory_optimized()

# ============================================================================
# LOAD MODELS FROM CONFIGURATION FILE (saved by main_project.py)
# ============================================================================

print("\n" + "="*80)
print("  🔬 DUAL DL + ML - LOAD MODELS FROM CONFIGURATION")
print("="*80 + "\n")

# Try to load model selection from configuration file
from pathlib import Path
PROJECT_ROOT = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma")
config_file = PROJECT_ROOT / "current_model_parameters.json"

choice1 = None
choice2 = None

if config_file.exists():
    with open(config_file, 'r') as f:
        import json
        config_data = json.load(f)
    
    selected_models = config_data.get('selected_models', [])
    
    if len(selected_models) >= 2:
        # Use the first 2 selected models
        choice1 = selected_models[0]
        choice2 = selected_models[1]
        
        print(f"✓ Loaded configuration from: {config_file}")
        print(f"  Model 1: {DL_MODELS[choice1]['name']}")
        print(f"  Model 2: {DL_MODELS[choice2]['name']}\n")

# If no config file or insufficient models, use defaults
if choice1 is None or choice2 is None:
    print("⚠️  No configuration file found or insufficient models")
    print("Please select 2 models manually:\n")
    
    print("📋 Available models:\n")
    for key in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
        model_name = DL_MODELS[key]['name']
        print(f"  {key}. {model_name}")
    print()
    
    # Get user choice for Model 1
    while True:
        choice1 = input("Select Model 1 (enter number): ").strip()
        if choice1 in DL_MODELS:
            break
        else:
            print(f"❌ Invalid choice. Please select a valid model number.")
    
    # Get user choice for Model 2 (must be different from Model 1)
    while True:
        choice2 = input("Select Model 2 (enter number, must be different from Model 1): ").strip()
        if choice2 in DL_MODELS:
            if choice2 != choice1:
                break
            else:
                print(f"❌ Model 2 must be different from Model 1. Please choose another model.")
        else:
            print(f"❌ Invalid choice. Please select a valid model number.")

# Helper function to load model and preprocess function
def load_model_and_preprocess(model_info):
    """Dynamically load a model and its preprocessing function"""
    if model_info.get('type') == 'custom':
        return build_model_cnn(), None
    else:
        try:
            model_module = getattr(tfa, model_info['module'])
            model_class = getattr(model_module, model_info['model_class'])
            preprocess_module = getattr(tfa, model_info['preprocess'])
            preprocess_fn = preprocess_module.preprocess_input
            model = model_class(weights="imagenet", include_top=False, input_shape=input_shape)
            return model, preprocess_fn
        except AttributeError as e:
            print(f"✗ Error loading {model_info['name']}: {e}")
            return None, None

model1_info = DL_MODELS[choice1]
model2_info = DL_MODELS[choice2]
model1_name = model1_info['name']
model2_name = model2_info['name']

print(f"\n✓ Selected models: {model1_name} and {model2_name}")
print("="*80 + "\n")

# ============================================================================
# ASK USER FOR PARAMETERS
# ============================================================================

print(f"\n📊 Enter parameters for each model:\n")

# Model 1 parameters
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

# Load models and preprocessing functions
model1_base, deep_learning_pre1 = load_model_and_preprocess(model1_info)
model2_base, deep_learning_pre2 = load_model_and_preprocess(model2_info)

if model1_base is None or model2_base is None:
    print("✗ Error loading models. Exiting.")
    exit(1)

name_model = model1_name+"_"+model2_name+"_ML"

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_auc', mode="max", patience=patience,min_delta=min_delta, restore_best_weights=True, 
        verbose=1)

# Define paths
train_ds, val_ds, test_ds = load_dataset(train_dir,test_dir, augment=True, deeplearning=False,preprocess_fn=None)

# Helper function to apply preprocessing (handles None case for custom CNN)
def apply_preprocess(image, preprocess_fn):
    if preprocess_fn is None:
        return image
    return preprocess_fn(image)

# Preprocessing function for two models
def preprocess_two_models(image, label):
    image = tf.cast(image, tf.float32)
    model1_img = apply_preprocess(image, deep_learning_pre1)
    model2_img = apply_preprocess(image, deep_learning_pre2)
    return (model1_img, model2_img), label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess_two_models, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess_two_models, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# Build model (two inputs)
model1_input = layers.Input(shape=input_shape, name="model1_input")
model2_input = layers.Input(shape=input_shape, name="model2_input")

# Set trainable to False for both models
model1_base.trainable = False
model2_base.trainable = False

# Connect inputs to base models
model1_out = model1_base(model1_input)
model2_out = model2_base(model2_input)

# GAP
resnet_feat = layers.GlobalAveragePooling2D(name="model1_gap")(model1_out)
densenet_feat = layers.GlobalAveragePooling2D(name="model2_gap")(model2_out)

# Combine features
combined_features = layers.Concatenate(name="combined_features")([resnet_feat, densenet_feat])

# Model
feature_model = Model(inputs=[model1_input, model2_input], outputs=combined_features, name="Combined_Feature_Extractor")

# -------------------------------
# Feature extraction function
# -------------------------------
def extract_features_from_dataset(dataset, model):
    X_list = []
    y_list = []
    for (resnet_batch, densenet_batch), label_batch in dataset:
        feats = model.predict([resnet_batch, densenet_batch], verbose=0)
        X_list.append(feats)
        y_list.append(label_batch.numpy())
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

# -------------------------------
# Extract features
# -------------------------------
X_train, y_train = extract_features_from_dataset(train_ds, feature_model)
X_test, y_test = extract_features_from_dataset(test_ds, feature_model)

y_train = (y_train > threshold).astype(int).ravel()
y_test  = (y_test  > threshold).astype(int).ravel()

metrics_per_model = {}
ml_models = {}
history = {}

# Add feature extractors to ml_models (use parametrizable names)
ml_models[f'{model1_name}_Extractor'] = Model(inputs=model1_input, outputs=resnet_feat, name=f'{model1_name}_Extractor')
ml_models[f'{model2_name}_Extractor'] = Model(inputs=model2_input, outputs=densenet_feat, name=f'{model2_name}_Extractor')
ml_models['Combined_Feature_Extractor'] = feature_model

# Initialize and train the SVM classifier
svm_classifier_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",   # très important pour mélanome
        probability=True,
        random_state=42
    ))
])
svm_classifier_rbf.fit(X_train, y_train)
ml_models, metrics_per_model = evaluate_model(svm_classifier_rbf, 'ML_svm_classifier_rbf', X_test,y_test,test_ds,ml_models)

# Initialize and train the SVM classifier
svm_classifier_poly = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="poly",
        C=1.0,
        class_weight="balanced",   # très important pour mélanome
        probability=True,
        random_state=42
    ))
])
svm_classifier_poly.fit(X_train, y_train)
ml_models, metrics_per_model = evaluate_model(svm_classifier_poly, 'ML_svm_classifier_poly', X_test,y_test,test_ds,ml_models)

# Initialize and train the SVM classifier
svm_classifier_sigmoid = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="sigmoid",
        C=1.0,
        class_weight="balanced",   # très important pour mélanome
        probability=True,
        random_state=42
    ))
])
svm_classifier_sigmoid.fit(X_train, y_train)
ml_models, metrics_per_model = evaluate_model(svm_classifier_sigmoid, 'ML_svm_classifier_sigmoid', X_test,y_test,test_ds,ml_models)

# --- 4. Train a Random Forest Classifier on Extracted Features ---
rf_classifier = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
rf_classifier.fit(X_train, y_train)
ml_models, metrics_per_model = evaluate_model(rf_classifier, 'ML_rf_classifier', X_test,y_test,test_ds,ml_models)

# --- 5. Train a Decision tree classifier on Extracted Features ---
df_classifier=DecisionTreeClassifier(class_weight="balanced",random_state=42)
df_classifier.fit(X_train, y_train)
ml_models, metrics_per_model = evaluate_model(df_classifier, 'ML_df_classifier', X_test,y_test,test_ds,ml_models)

knn_classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,        # à ajuster
        weights="distance",   # meilleur pour petit dataset
        metric="minkowski"    # euclidienne par défaut
    ))
])
knn_classifier.fit(X_train, y_train)
ml_models, metrics_per_model = evaluate_model(knn_classifier, 'ML_knn_classifier', X_test,y_test,test_ds,ml_models)

# Save model, metrics, and training history to pickle
experiment_data = {
    'models': ml_models,
    'metrics': metrics_per_model,
    'history': history
}
save_experiment(feature_model, name_model, experiment_data)