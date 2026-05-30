import os
import tensorflow as tf
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import patience,min_delta, threshold,train_dir,test_dir, input_shape ,path_project

# Configure environment for progress bars display
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from src.modules.save_model import save_experiment
from src.modules.Evaluate import evaluate_model
from src.modules.Model import build_model_cnn, DL_MODELS
from tensorflow.keras import layers, Model
from src.modules.Dataset import load_dataset
from src.config.Config import patience,min_delta, threshold,train_dir,test_dir, input_shape 
import tensorflow.keras.applications as tfa
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.callbacks import EarlyStopping
from src.modules.memory_optimized_test import test_memory_optimized

#optimized configuration for memory-constrained systems
test_memory_optimized()

# ============================================================================
# LOAD MODELS FROM CONFIGURATION FILE (saved by main_project.py)
# ============================================================================

print("\n" + "="*80)
print("  🔬 DUAL DL + ML ENSEMBLE - LOAD MODELS FROM CONFIGURATION")
print("="*80 + "\n")

# Try to load model selection from configuration file
PROJECT_ROOT = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma")
config_file = PROJECT_ROOT / "current_model_parameters.json"

choice1 = None
choice2 = None

if config_file.exists():
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    selected_models = config_data.get('selected_models', [])
    
    if len(selected_models) >= 2:
        # Use the first 2 selected models
        choice1 = selected_models[0]
        choice2 = selected_models[1]
        
        print(f"✓ Loaded configuration from: {config_file}")
        print(f"  Model 1: {DL_MODELS[choice1]['name']}")
        print(f"  Model 2: {DL_MODELS[choice2]['name']}\n")

# If no config file or insufficient models, ask user to select manually
if choice1 is None or choice2 is None:
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



selected_model_info1 = DL_MODELS[choice1]
model1_name = selected_model_info1['name']
print(f"\n✓ Selected Model 1: {model1_name}")

selected_model_info2 = DL_MODELS[choice2]
model2_name = selected_model_info2['name']
print(f"✓ Selected Model 2: {model2_name}")

# Ask user for parameters
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

print(f"\n{'─'*80}")
print(f"📊 Configuration sélectionnée:")
print(f"  Model 1 ({model1_name}): LR={model1_lr}, WD={model1_wd}")
print(f"  Model 2 ({model2_name}): LR={model2_lr}, WD={model2_wd}")
print(f"{'─'*80}\n")

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

# Load models and preprocessing functions
model1_base, deep_learning_pre1 = load_model_and_preprocess(selected_model_info1)
model2_base, deep_learning_pre2 = load_model_and_preprocess(selected_model_info2)

if model1_base is None or model2_base is None:
    print("\n✗ Error loading models. Exiting.")
    exit(1)

print(f"✓ {model1_name} loaded successfully")
print(f"✓ {model2_name} loaded successfully\n")

# PARAMETRIZABLE MODEL NAMES (AUTO)
name_model = model1_name+"_"+model2_name+"_ML_BAGGING"


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

resnet_model = Model(inputs=model1_input, outputs=resnet_feat)
densenet_model = Model(inputs=model2_input, outputs=densenet_feat)

    # Combine features
combined_features = layers.Concatenate(name="combined_features")([resnet_feat, densenet_feat])

    # Model
feature_model = Model(inputs=[model1_input, model2_input], outputs=combined_features, name="Combined_Feature_Extractor")
#feature_model.summary()

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

# Add individual and combined feature extractors to ml_models
ml_models[f'{model1_name}_Extractor'] = resnet_model
ml_models[f'{model2_name}_Extractor'] = densenet_model
ml_models['Combined_Feature_Extractor'] = feature_model

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
    n_jobs=1,
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
    n_jobs=1,
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
    n_jobs=1,
    random_state=42
)

bagging_rf = BaggingClassifier(
    estimator=RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        n_jobs=1,
        random_state=42
    ),
    n_estimators=10,
    max_samples=0.8,
    bootstrap=True,
    n_jobs=1,
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
            n_jobs=1
        ))
    ],
    voting="soft",
    weights=[3, 2, 2, 2],
    n_jobs=2
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