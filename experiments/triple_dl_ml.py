# -*- coding: utf-8 -*-   modified
"""
Created on Wed Oct  8 08:17:27 2025

@author: brahim

Modified for THREE Deep Learning Models
"""

# CNN  Binary Classification (Malignant vs Benign)

import os
import tensorflow as tf
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import (
    patience, min_delta, threshold, train_dir, test_dir, input_shape, metrics_per_model, ml_models,
    learning_rate_one, weight_decay_one, learning_rate_two, weight_decay_two,
    learning_rate_three, weight_decay_three,path_project
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
from src.config.Config import (
    patience, min_delta, threshold, train_dir, test_dir, input_shape, metrics_per_model, ml_models,
    learning_rate_one, weight_decay_one, learning_rate_two, weight_decay_two,
    learning_rate_three, weight_decay_three
)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from src.modules.memory_optimized_test import test_memory_optimized
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#optimized configuration for memory-constrained systems
test_memory_optimized()

# ============================================================================
# LOAD MODELS AND PARAMETERS FROM CONFIGURATION FILE
# ============================================================================

print("\n" + "="*80)
print("  🔬 TRIPLE DL + ML - LOAD MODELS FROM CONFIGURATION")
print("="*80 + "\n")

# Try to load model selection from configuration file
PROJECT_ROOT = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma")
config_file = PROJECT_ROOT / "current_model_parameters.json"

choice1 = None
choice2 = None
choice3 = None

if config_file.exists():
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    selected_models = config_data.get('selected_models', [])
    
    if len(selected_models) >= 3:
        # Use the first 3 selected models
        choice1, choice2, choice3 = selected_models[0], selected_models[1], selected_models[2]
        
        model1_info = DL_MODELS[choice1]
        model2_info = DL_MODELS[choice2]
        model3_info = DL_MODELS[choice3]
        model1_name = model1_info['name']
        model2_name = model2_info['name']
        model3_name = model3_info['name']
        
        print(f"✓ Loaded configuration from: {config_file}")
        print(f"  Model 1: {model1_name}")
        print(f"  Model 2: {model2_name}")
        print(f"  Model 3: {model3_name}\n")

# If no config file or insufficient models, fallback to defaults
if choice1 is None or choice2 is None or choice3 is None:
    print("⚠️  No configuration file found or insufficient models")
    print("Using default models: ResNet50, DenseNet169, EfficientNetV2B0\n")
    choice1, choice2, choice3 = "1", "8", "2"
    
    model1_info = DL_MODELS[choice1]
    model2_info = DL_MODELS[choice2]
    model3_info = DL_MODELS[choice3]
    model1_name = model1_info['name']
    model2_name = model2_info['name']
    model3_name = model3_info['name']

print(f"{'─'*80}")
print(f"📊 Using FIXED parameters from Config.py:")
print(f"  Model 1 ({model1_name}): LR={learning_rate_one}, WD={weight_decay_one}")
print(f"  Model 2 ({model2_name}): LR={learning_rate_two}, WD={weight_decay_two}")
print(f"  Model 3 ({model3_name}): LR={learning_rate_three}, WD={weight_decay_three}")
print(f"{'─'*80}\n")

print(f"✓ Selected models: {model1_name}, {model2_name}, and {model3_name}")
print("="*80 + "\n")

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

# Load the three base models
print("\n📥 Loading base models...")
model1_base, deep_learning_pre1 = load_model_and_preprocess(model1_info)
model2_base, deep_learning_pre2 = load_model_and_preprocess(model2_info)
model3_base, deep_learning_pre3 = load_model_and_preprocess(model3_info)

if model1_base is None or model2_base is None or model3_base is None:
    print("✗ Error loading one or more models!")
    exit(1)

print(f"✓ All 3 base models loaded successfully\n")

# Define paths
train_ds, val_ds, test_ds = load_dataset(train_dir,test_dir, augment=True, deeplearning=False,preprocess_fn=None)

# Helper function to apply preprocessing (handles None case for custom CNN)
def apply_preprocess(image, preprocess_fn):
    if preprocess_fn is None:
        return image
    return preprocess_fn(image)

# Preprocessing function for three models
def preprocess_three_models(image, label):
    image = tf.cast(image, tf.float32)
    model1_img = apply_preprocess(image, deep_learning_pre1)
    model2_img = apply_preprocess(image, deep_learning_pre2)
    model3_img = apply_preprocess(image, deep_learning_pre3)
    return (model1_img, model2_img, model3_img), label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(preprocess_three_models, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess_three_models, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# Build model (three inputs)
model1_input = layers.Input(shape=input_shape, name="model1_input")
model2_input = layers.Input(shape=input_shape, name="model2_input")
model3_input = layers.Input(shape=input_shape, name="model3_input")

# Set trainable to False for all models
model1_base.trainable = False
model2_base.trainable = False
model3_base.trainable = False

# Connect inputs to base models
model1_out = model1_base(model1_input)
model2_out = model2_base(model2_input)
model3_out = model3_base(model3_input)

# GAP
model1_feat = layers.GlobalAveragePooling2D(name="model1_gap")(model1_out)
model2_feat = layers.GlobalAveragePooling2D(name="model2_gap")(model2_out)
model3_feat = layers.GlobalAveragePooling2D(name="model3_gap")(model3_out)

# Combine features
combined_features = layers.Concatenate(name="combined_features")([model1_feat, model2_feat, model3_feat])

# Model
feature_model = Model(inputs=[model1_input, model2_input, model3_input], outputs=combined_features, name="Combined_Feature_Extractor")

# -------------------------------
# Feature extraction function
# -------------------------------
def extract_features_from_dataset(dataset, model):
    X_list = []
    y_list = []
    for (model1_batch, model2_batch, model3_batch), label_batch in dataset:
        feats = model.predict([model1_batch, model2_batch, model3_batch], verbose=0)
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
ml_models[f'{model1_name}_Extractor'] = Model(inputs=model1_input, outputs=model1_feat, name=f'{model1_name}_Extractor')
ml_models[f'{model2_name}_Extractor'] = Model(inputs=model2_input, outputs=model2_feat, name=f'{model2_name}_Extractor')
ml_models[f'{model3_name}_Extractor'] = Model(inputs=model3_input, outputs=model3_feat, name=f'{model3_name}_Extractor')
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

# --- 6. Train a KNN classifier on Extracted Features ---
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
