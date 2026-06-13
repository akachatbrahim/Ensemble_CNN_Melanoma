#!/usr/bin/env python3
"""
Interactive Melanoma Detection Tester
======================================
Simple interactive script to test single image prediction

Usage:
    python test_interactive.py
    
This script will guide you through testing an image.
"""

import sys
import os
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config.Config import input_shape,THRESHOLD,IMG_SIZE,save_model_path

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_PATH = save_model_path
#IMG_SIZE = IMG_SIZE
CLASS_NAMES = {0: "BENIGN", 1: "MALIGNANT"}

# ============================================================================
# FUNCTIONS
# ============================================================================

def print_section(title, char="="):
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"  {title}")
    print(f"{char * 80}\n")

def list_available_models(only_ensemble=True):
    """List all available trained models.

    If `only_ensemble` is True (default), only returns ENSEMBLE or BAGGING models 
    (actual ensemble combinations, not hybrid DL+ML).
    
    True Ensemble = Multiple models with voting/combination (ENSEMBLE, BAGGING).
    """
    models = sorted([f.stem for f in MODELS_PATH.glob("*.pkl")])
    if only_ensemble:
        # Filter for ONLY true ensemble models (contains ENSEMBLE or BAGGING keyword)
        filtered = [m for m in models if ('ENSEMBLE' in m or 'BAGGING' in m)]
        return filtered if filtered else models  # Fallback to all if no ensembles found
    return models

def get_model_choice(only_ensemble=True):
    """Let user choose a model.

    If `only_ensemble` is True (default), the list shows only ensemble models.
    """
    models = list_available_models(only_ensemble=only_ensemble)
    
    if only_ensemble:
        model_type_label = "ENSEMBLE MODELS (11)"
    else:
        model_type_label = "ALL AVAILABLE MODELS (30)"
    
    print_section(f"📋 {model_type_label}", "-")
    print("  0. Back to previous menu")
    for i, model in enumerate(models, 1):
        print(f"  {i:2d}. {model}")
    print()
    
    while True:
        try:
            choice = input("Select model number: ").strip()
            if not choice:
                # Default to first model if available
                if models:
                    return models[0]
                print("❌ No models available")
                return None
            choice = int(choice)
            if choice == 0:
                return "__BACK_TO_MENU__"
            if 1 <= choice <= len(models):
                return models[choice - 1]
            print(f"❌ Please enter a number between 0 and {len(models)}")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")

def choose_image_dialog():
    """Open a file dialog for the user to select an image."""
    if sys.platform != 'win32' and not os.environ.get('DISPLAY'):
        print("❌ File dialog unavailable: no display name and no $DISPLAY environment variable.")
        print("   Le dialogue ne peut pas s'ouvrir depuis ce serveur distant.")
        print("   Exécutez le script localement sur votre PC ou activez le transfert X11.")
        return None

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return file_path or None
    except Exception as e:
        print(f"❌ File dialog unavailable: {str(e)}")
        print("   Le dialogue local est requis, mais non disponible sur ce terminal.")
        return None


def get_image_path():
    """Get and validate image path from user."""
    while True:
        image_path = input(
            "\n📁 Enter image path (or press Enter to open local file dialog) / [b]ack to menu: "
        ).strip()

        if image_path.lower() in ['b', 'back', 'menu']:
            return "__BACK_TO_MENU__"

        if image_path == '':
            dialog_path = choose_image_dialog()
            if dialog_path is None:
                print("❌ No image selected from dialog.")
                continue
            image_path = dialog_path

        image_path = Path(image_path).expanduser()

        if not image_path.exists():
            print(f"❌ File not found: {image_path}")
            continue

        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            print("❌ Unsupported image format. Use: jpg, jpeg, png, bmp, gif")
            continue

        return str(image_path)

def load_image(image_path):
    """Load and preprocess image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return None

def load_model(model_name):
    """Load trained model from pickle."""
    try:
        model_path = MODELS_PATH / f"{model_name}.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        # Models are saved as dictionaries with 'model' key
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            # For ML_ENSEMBLE, return both model and data
            if 'ML_ENSEMBLE' in model_name and 'experiment_data' in data:
                return model, data
            else:
                return model
        else:
            return data
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

def normalize_image(img_array):
    """Normalize image to [0, 1] range."""
    return img_array / 255.0

def preprocess_image_for_model(img_array, model_name):
    """Apply exact same preprocessing as training for each model."""
    img = img_array / 255.0
    
    if 'EfficientNetV2' in model_name or 'EfficientNet' in model_name:
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        img = preprocess_input(img * 255.0)
    elif 'ResNet50' in model_name or 'ResNet' in model_name:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        img = preprocess_input(img * 255.0)
    elif 'DenseNet' in model_name:
        from tensorflow.keras.applications.densenet import preprocess_input
        img = preprocess_input(img * 255.0)
    elif 'MobileNetV2' in model_name or 'MobileNet' in model_name:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = preprocess_input(img * 255.0)
    elif 'VGG' in model_name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        img = preprocess_input(img * 255.0)
    elif 'ConvNeXt' in model_name:
        from tensorflow.keras.applications.convnext import preprocess_input
        img = preprocess_input(img * 255.0)
    else:
        img = img
    
    return img

def predict_image(model, img_array, model_name, model_data=None):
    """Make prediction on image."""
    try:
        # Special handling for ML_ENSEMBLE models
        if 'ML_ENSEMBLE' in model_name and model_data is not None:
            try:
                # Try different feature extractor names
                feature_extractor = None
                for key in model_data['experiment_data']['models'].keys():
                    if 'Combined' in key and 'Extractor' in key:
                        feature_extractor = model_data['experiment_data']['models'][key]
                        break
                
                # If no combined extractor, try EfficientNetV2B2 variants
                if feature_extractor is None:
                    for key in model_data['experiment_data']['models'].keys():
                        if 'EfficientNetV2B2' in key and 'Extractor' in key:
                            feature_extractor = model_data['experiment_data']['models'][key]
                            break
                
                # If still not found, try any extractor
                if feature_extractor is None:
                    for key in model_data['experiment_data']['models'].keys():
                        if 'Extractor' in key:
                            feature_extractor = model_data['experiment_data']['models'][key]
                            break
                
                if feature_extractor is None:
                    raise RuntimeError("No feature extractors found in ML_ENSEMBLE model")
                
                # Use proper preprocessing for feature extraction
                preprocessed_img = preprocess_image_for_model(img_array, 'EfficientNetV2B2')
                features = feature_extractor.predict(preprocessed_img, verbose=0)
                
                # Flatten features if needed
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                
                # Use VotingClassifier to predict
                prob_malignant = model.predict_proba(features)[0][1]
            except Exception as e:
                raise RuntimeError(f"ML_ENSEMBLE prediction failed: {str(e)}")
        
        # Check if it's an ML model
        elif 'ML' in model_name:
            img_flat = img_array.reshape(1, -1)
            prob_malignant = model.predict_proba(img_flat)[0][1]
        
        # DL model (Keras)
        else:
            preprocessed_img = preprocess_image_for_model(img_array, model_name)
            prediction = model.predict(preprocessed_img, verbose=0)
            prob_malignant = prediction[0][0] if prediction.shape[-1] == 1 else prediction[0][1]
        
        prob_malignant = float(np.clip(prob_malignant, 0, 1))
        prob_benign = 1.0 - prob_malignant
        predicted_class = 1 if prob_malignant >= THRESHOLD else 0
        
        return predicted_class, prob_malignant, prob_benign
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return None, None, None

def display_results(image_path, model_name, predicted_class, prob_malignant, prob_benign):
    """Display prediction results."""
    print_section("🔬 PREDICTION RESULTS", "=")
    
    print(f"📁 Image:       {image_path}")
    print(f"🤖 Model:       {model_name}\n")
    
    print(f"{'─' * 80}\n")
    
    # Diagnosis
    diagnosis = CLASS_NAMES[predicted_class]
    if predicted_class == 1:
        print(f"  🎯 DIAGNOSIS:  \033[91m{diagnosis}\033[0m")  # Red
    else:
        print(f"  🎯 DIAGNOSIS:  \033[92m{diagnosis}\033[0m")  # Green
    
    # Probabilities
    print(f"\n  📊 PROBABILITIES:")
    print(f"     Benign:     {prob_benign*100:6.2f}%")
    print(f"     Malignant:  {prob_malignant*100:6.2f}%")
    
    # Confidence
    max_prob = max(prob_malignant, prob_benign)
    confidence = (max_prob - 0.5) * 2 * 100
    print(f"\n  💪 Confidence:  {confidence:6.1f}%")
    
    print(f"\n{'─' * 80}\n")

def test_another():
    """Ask if user wants to test another image."""
    while True:
        choice = input("Test another image? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            return False
        print("❌ Please enter 'y' or 'n'")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    # Parse optional CLI args
    parser = argparse.ArgumentParser(description="Melanoma Detection Tester - Ensemble Models", add_help=False)
    parser.add_argument('--model', type=str, help='Model name to use (non-interactive)')
    parser.add_argument('--all-models', action='store_true', help='Show all models, not just ensembles')
    args, _ = parser.parse_known_args()
    
    print_section("🏥 MELANOMA DETECTION SYSTEM - TESTER", "=")
    print("  Input:  Image file (jpg, png, etc.)")
    print("  Output: Diagnosis (Benign or Malignant) with probabilities")
    
    # Determine if show ensemble only or all models
    show_ensemble_only = not args.all_models

    def choose_mode():
        print_section("📋 MODEL SELECTION MODE", "-")
        print("  1. Ensemble models only (11 models)")
        print("  2. All models available (30 models)")
        print()
        while True:
            choice = input("Choose mode (1 or 2): ").strip()
            if choice in ['1', '2']:
                return choice == '1'
            print("❌ Please enter 1 or 2")

    if not args.model and not args.all_models:
        show_ensemble_only = choose_mode()

    while True:
        mode_label = "🎯 ENSEMBLE MODELS ONLY" if show_ensemble_only else "📊 ALL MODELS"
        print(f"\n  Mode:   {mode_label}\n")

        # Get model choice
        if args.model:
            model_name = args.model
            print(f"\n✓ Model selected (CLI): {model_name}")
        else:
            model_name = get_model_choice(only_ensemble=show_ensemble_only)
            if model_name == "__BACK_TO_MENU__":
                if not args.all_models:
                    show_ensemble_only = choose_mode()
                continue
            if model_name is None:
                print("❌ No models available")
                return
            print(f"\n✓ Model selected: {model_name}")
        break
    print("🔄 Loading model...")
    model_result = load_model(model_name)
    if model_result is None:
        return
    # Handle both tuple and single return values
    if isinstance(model_result, tuple):
        model, model_data = model_result
    else:
        model = model_result
        model_data = None
    print("✓ Model loaded successfully\n")
    
    # Test loop
    while True:
        # Get image path
        image_path = get_image_path()
        if image_path == "__BACK_TO_MENU__":
            print("↩️ Returning to model selection...\n")
            model_name = get_model_choice(only_ensemble=show_ensemble_only)
            if model_name is None:
                print("❌ No models available")
                return
            print(f"\n✓ Model selected: {model_name}")
            print("🔄 Loading model...")
            model_result = load_model(model_name)
            if model_result is None:
                return
            if isinstance(model_result, tuple):
                model, model_data = model_result
            else:
                model = model_result
                model_data = None
            print("✓ Model loaded successfully\n")
            continue
        
        # Load and preprocess image
        print("🔄 Processing image...")
        img_array = load_image(image_path)
        if img_array is None:
            continue
        print("✓ Image processed\n")
        
        # Make prediction
        print("⏳ Analyzing image...")
        predicted_class, prob_malignant, prob_benign = predict_image(model, img_array, model_name, model_data)
        
        if predicted_class is None:
            continue
        
        # Display results
        display_results(image_path, model_name, predicted_class, prob_malignant, prob_benign)
        
        # Test another?
        if not test_another():
            break
    
    print_section("👋 Thank you for using Melanoma Detection Tester!", "=")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
