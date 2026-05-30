import os
import tensorflow as tf
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config.Config import (
    patience, min_delta, threshold, train_dir, test_dir, input_shape,
    learning_rate_one, weight_decay_one, learning_rate_two, weight_decay_two,
    learning_rate_three, weight_decay_three, path_project
)

# Configure environment for progress bars display
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['KERAS_PROGRESS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from src.modules.save_model import save_experiment
from src.modules.Evaluate import evaluate_model
from src.modules.Model import build_model, build_model_cnn, DL_MODELS
from tensorflow.keras import layers, Model
from src.modules.Dataset import load_dataset
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
# LOAD MODELS AND PARAMETERS FROM CONFIGURATION FILE
# ============================================================================

print("\n" + "="*80)
print("  🔬 TRIPLE DL + ML ENSEMBLE - LOAD MODELS FROM CONFIGURATION")
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
    print("Using default models: EfficientNetV2B0, DenseNet169, ResNet50\n")
    choice1, choice2, choice3 = "2", "8", "5"
    
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

# Load models and preprocessing functions
model1_base, deep_learning_pre1 = load_model_and_preprocess(model1_info)
model2_base, deep_learning_pre2 = load_model_and_preprocess(model2_info)
model3_base, deep_learning_pre3 = load_model_and_preprocess(model3_info)

if model1_base is None or model2_base is None or model3_base is None:
    print("\n✗ Error loading models. Exiting.")
    exit(1)

print(f"✓ {model1_name} loaded successfully")
print(f"✓ {model2_name} loaded successfully")
print(f"✓ {model3_name} loaded successfully\n")

# PARAMETRIZABLE MODEL NAMES (AUTO)
name_model = model1_name+"_"+model2_name+"_"+model3_name+"_ML_BAGGING"

# ======================================================
# TRAIN MODELS WITH FIXED PARAMETERS
# ======================================================
print("\n" + "="*80)
print("  🏋️  TRAINING MODELS (Each with fixed LR and WD from Config.py)")
print("="*80 + "\n")

model_params = [
    (learning_rate_one, weight_decay_one),
    (learning_rate_two, weight_decay_two),
    (learning_rate_three, weight_decay_three)
]

model_configs = []

# EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_auc', mode="max", patience=patience,min_delta=min_delta, restore_best_weights=True, 
        verbose=1)

# Define paths
train_ds, val_ds, test_ds = load_dataset(train_dir,test_dir, augment=True, deeplearning=False,preprocess_fn=None)

# Train each model with its assigned parameters
trained_models = []
histories = {}

for idx, (model_base, model_info, model_name) in enumerate([(model1_base, model1_info, model1_name),
                                                              (model2_base, model2_info, model2_name),
                                                              (model3_base, model3_info, model3_name)]):
    lr, wd = model_params[idx]
    
    # Store configuration
    config = {
        'name': model_name,
        'learning_rate': lr,
        'weight_decay': wd,
        'position': idx + 1
    }
    model_configs.append(config)
    
    print(f"Training [{idx+1}]: {model_name}")
    print(f"  Parameters:")
    print(f"    └─ Learning Rate: {lr}")
    print(f"    └─ Weight Decay:  {wd}\n")
    
    # Build model with assigned parameters
    if model_info.get('type') != 'custom':
        model_base.trainable = False  # Keep base frozen for transfer learning
    
    model_base.trainable = False  # Keep all models frozen
    trained_models.append(model_base)
    
    print(f"✓ {model_name} ready for feature extraction\n")

# Store training history (no training performed in hybrid, only extraction)
history = {}

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

model1_model = Model(inputs=model1_input, outputs=model1_feat)
model2_model = Model(inputs=model2_input, outputs=model2_feat)
model3_model = Model(inputs=model3_input, outputs=model3_feat)

# Combine features
combined_features = layers.Concatenate(name="combined_features")([model1_feat, model2_feat, model3_feat])

    # Model
feature_model = Model(inputs=[model1_input, model2_input, model3_input], outputs=combined_features, name="Combined_Feature_Extractor")
#feature_model.summary()

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

# Add individual and combined feature extractors to ml_models
ml_models[f'{model1_name}_Extractor'] = model1_model
ml_models[f'{model2_name}_Extractor'] = model2_model
ml_models[f'{model3_name}_Extractor'] = model3_model
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
    'history': history,
    'model_configs': model_configs
}
save_experiment(ensemble, name_model, experiment_data)