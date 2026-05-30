import os
import tensorflow as tf
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import ( epochs, patience,min_delta, threshold,train_dir,test_dir,learning_rate, 
                                weight_decay,path_project
)

# Configure environment for progress bars display
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from src.modules.Model import build_model, build_model_cnn, DL_MODELS
from src.modules.Dataset import load_dataset,Balanced
from src.config.Config import epochs, patience,min_delta, threshold,train_dir,test_dir,learning_rate, weight_decay
import tensorflow.keras.applications as tfa
from src.modules.save_model import save_experiment
from src.modules.Evaluate import evaluate_model
from src.modules.memory_optimized_test import test_memory_optimized

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#optimized configuration for memory-constrained systems
test_memory_optimized()

# ============================================================================
# SELECT DEEP LEARNING MODEL
# ============================================================================

print("\n" + "="*80)
print("  🔬 DEEP LEARNING MODEL SELECTION (DL + ML ENSEMBLE)")
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
        print(f"  Falling back to EfficientNetV2B0...")
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0 as deep_learning, preprocess_input as deep_learning_pre
        model_name = "EfficientNetV2B0"

# PARAMETRIZABLE MODEL NAME (AUTO)
name_model = model_name+"_ML_BAGGING"

# EarlyStopping callback
from tensorflow.keras.callbacks import EarlyStopping
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
    model = build_model(deep_learning,learning_rate, weight_decay)

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

def extract_features(dataset, model):
    X, y = [], []
    for images, labels in dataset:
        features = model.predict(images, verbose=0)
        X.append(features)
        y.append(labels.numpy())
    return np.vstack(X), np.concatenate(y)

feature_extractor = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.layers[-3].output
)
X_train, y_train = extract_features(train_ds, feature_extractor)
X_test,  y_test  = extract_features(test_ds, feature_extractor)

y_train = (y_train > threshold).astype(int).ravel()
y_test  = (y_test  > threshold).astype(int).ravel()

metrics_per_model = {}
ml_models = {}

# Add DL model to ml_models
ml_models[f'{model_name}_Feature_Extractor'] = feature_extractor

bagging_svm = BaggingClassifier(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ]),
    n_estimators=20,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

bagging_knn = BaggingClassifier(
    estimator=Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=7,
            weights="distance"
        ))
    ]),
    n_estimators=30,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

bagging_dt = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),
    n_estimators=100,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

bagging_rf = BaggingClassifier(
    estimator=RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),
    n_estimators=10,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

ensemble = VotingClassifier(
    estimators=[
        ("bag_svm", bagging_svm),
        ("bag_knn", bagging_knn),
        ("bag_dt", bagging_dt),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ],
    voting="soft",
    weights=[3, 2, 2, 2],
    n_jobs=-1
)

ensemble.fit(X_train, y_train)

ml_models, metrics_per_model = evaluate_model(ensemble,name_model,X_test,y_test,test_ds,ml_models)

# Save model, metrics, and training history to pickle
experiment_data = {
    'models': ml_models,
    'metrics': metrics_per_model,
    'history': history
}
save_experiment(ensemble, name_model, experiment_data)