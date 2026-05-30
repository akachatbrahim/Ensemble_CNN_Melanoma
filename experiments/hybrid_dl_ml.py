# -*- coding: utf-8 -*-   modified
"""
Created on Wed Oct  8 08:17:27 2025

@author: brahim
"""

# Hybrid Binary Classification (Malignant vs Benign)

import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import (
    epochs, patience,min_delta, threshold,train_dir,test_dir,
    learning_rate_one, weight_decay_one,path_project
)    

# Configure environment for progress bars display
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from src.modules.Model import build_model,build_model_cnn, DL_MODELS
from src.modules.save_model import save_experiment
from src.modules.Evaluate import evaluate_model
from src.modules.Dataset import load_dataset,Balanced
import tensorflow.keras.applications as tfa

from tensorflow.keras.callbacks import EarlyStopping
from src.modules.memory_optimized_test import test_memory_optimized

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats

#optimized configuration for memory-constrained systems
test_memory_optimized()

# ============================================================================
# SELECT DEEP LEARNING MODEL
# ============================================================================

print("\n" + "="*80)
print("  🔬 DEEP LEARNING MODEL SELECTION (DL + ML Hybrid)")
print("="*80 + "\n")

# Display options
print("📋 Available Deep Learning Models:\n")
for key in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
    info = DL_MODELS[key]
    print(f"  {key:2}. {info['name']:20} - {info['description']}")

# Get user choice
print("\n")
choice = input(f"Select model (1-{len(DL_MODELS)}) [default: 1 (ResNet50)]: ").strip()

if choice not in DL_MODELS:
    choice = "1"  # Default to ResNet50

selected_model_info = DL_MODELS[choice]
model_name = selected_model_info['name']

print(f"\n✓ Selected: {model_name}")

# Handle custom CNN separately
if selected_model_info.get('type') == 'custom':
    print(f"✓ {model_name} loaded successfully\n")
    model = build_model_cnn()
    deep_learning_pre = None
else:
    # Load selected model from Keras Applications
    try:
        model_module = getattr(tfa, selected_model_info['module'])
        deep_learning = getattr(model_module, selected_model_info['model_class'])
        preprocess_module = getattr(tfa, selected_model_info['preprocess'])
        deep_learning_pre = preprocess_module.preprocess_input
        print(f"✓ {model_name} loaded successfully\n")
    except AttributeError as e:
        print(f"✗ Error loading {model_name}: {e}")
        print(f"  Falling back to ResNet50...")
        from tensorflow.keras.applications.resnet50 import ResNet50 as deep_learning, preprocess_input as deep_learning_pre
        model_name = "ResNet50"

# PARAMETRIZABLE MODEL NAME (AUTO)
name_model = model_name+"_ML"

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_auc', mode="max", patience=patience,min_delta=min_delta, restore_best_weights=True, 
        verbose=1)

# Define paths
train_ds, val_ds, test_ds = load_dataset(train_dir,test_dir, augment=True, deeplearning=True,preprocess_fn=deep_learning_pre)
class_weights = Balanced(train_ds)

# Build model based on type
if selected_model_info.get('type') == 'custom':
    # Model already built above
    pass
else:
    model = build_model(deep_learning,learning_rate_one, weight_decay_one)

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# --- 3. Extract Features-
def extract_features(dataset, model):
    X, y = [], []
    for images, labels in dataset:
        features = model.predict(images, verbose=0)
        X.append(features)
        y.append(labels.numpy())
    return np.vstack(X), np.concatenate(y).ravel()

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-3].output
)

X_train, y_train  = extract_features(train_ds, feature_extractor)
X_test, y_test  = extract_features(test_ds, feature_extractor)

y_train = (y_train > threshold).astype(int).ravel()
y_test  = (y_test  > threshold).astype(int).ravel()

metrics_per_model = {}
ml_models = {
    f'{model_name}_Feature_Extractor': feature_extractor
}

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
save_experiment(model, name_model, experiment_data)
