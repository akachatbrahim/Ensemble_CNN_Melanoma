#!/usr/bin/env python3
"""
Test Script for Melanoma Detection Application
================================================
Input: Image file path
Output: Diagnosis (Benign or Malignant) with probability percentages

Usage:
    python test_image_prediction.py <image_path> [model_name]
    
Example:
    python test_image_prediction.py ./image.jpg EfficientNetV2B2
    python test_image_prediction.py ./melanoma.png EfficientNetV2B2_ML_ENSEMBLE
"""

import sys
import os
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import argparse
from src.config.Config import save_model_path ,IMG_SIZE,THRESHOLD

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CONSTANTS
# ============================================================================

MODELS_PATH = save_model_path
#IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "Benign", 1: "Malignant"}
#THRESHOLD = 0.5

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def print_result(title, value, color_code=None):
    """Print a result line."""
    if color_code:
        print(f"\n  {title}: {color_code}{value}\033[0m")
    else:
        print(f"\n  {title}: {value}")

def load_image(image_path):
    """Load and preprocess an image."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

def load_model(model_name):
    """Load a trained model from pickle file."""
    model_path = Path(MODELS_PATH) / f"{model_name}.pkl"
    
    if not model_path.exists():
        available_models = [f.stem for f in Path(MODELS_PATH).glob("*.pkl")]
        raise FileNotFoundError(
            f"Model '{model_name}' not found.\n"
            f"Available models:\n" + "\n".join([f"  - {m}" for m in sorted(available_models)])
        )
    
    try:
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
        raise RuntimeError(f"Error loading model: {str(e)}")

def normalize_image(img_array):
    """Normalize image to [0, 1] range."""
    return img_array / 255.0

def preprocess_image_for_model(img_array, model_name):
    """Apply exact same preprocessing as training for each model."""
    # Normalize to [0, 1]
    img = img_array / 255.0
    
    # Apply model-specific preprocessing
    if 'EfficientNetV2' in model_name or 'EfficientNet' in model_name:
        from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        img = preprocess_input(img * 255.0)  # preprocess_input expects [0, 255]
    
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
        # For custom CNN, just normalize to [0, 1]
        img = img
    
    return img

def make_prediction(model, img_array, model_name, model_data=None):
    """Make prediction on an image."""
    try:
        # Special handling for ML_ENSEMBLE models
        if 'ML_ENSEMBLE' in model_name and model_data is not None:
            try:
                # Use proper preprocessing for feature extraction
                preprocessed_img = preprocess_image_for_model(img_array, 'EfficientNetV2B2')
                
                # Try combined feature extractor first (expects 2 inputs)
                combined_extractor = model_data['experiment_data']['models'].get('Combined_Feature_Extractor')
                if combined_extractor is not None:
                    # Pass same image twice to Combined_Feature_Extractor
                    features = combined_extractor.predict([preprocessed_img, preprocessed_img], verbose=0)
                else:
                    # Fallback to individual extractors
                    resnet_ext = model_data['experiment_data']['models'].get('ResNet50_Extractor')
                    effnet_ext = model_data['experiment_data']['models'].get('EfficientNetV2B2_Extractor')
                    
                    if resnet_ext is not None and effnet_ext is not None:
                        features_resnet = resnet_ext.predict(preprocessed_img, verbose=0)
                        features_effnet = effnet_ext.predict(preprocessed_img, verbose=0)
                        # Concatenate features
                        features = np.concatenate([features_resnet, features_effnet], axis=1)
                    else:
                        raise RuntimeError("No suitable feature extractors found")
                
                # Flatten features if needed
                if len(features.shape) > 2:
                    features = features.reshape(features.shape[0], -1)
                
                # Use VotingClassifier to predict
                prob_malignant = model.predict_proba(features)[0][1]
            except Exception as e:
                raise RuntimeError(f"ML_ENSEMBLE prediction failed: {str(e)}")
        
        # Check if it's an ML model (VotingClassifier)
        elif 'ML' in model_name:
            img_flat = img_array.reshape(1, -1)
            prob_malignant = model.predict_proba(img_flat)[0][1]
        
        # DL model (Keras) - use proper preprocessing for each model
        else:
            preprocessed_img = preprocess_image_for_model(img_array, model_name)
            prediction = model.predict(preprocessed_img, verbose=0)
            prob_malignant = prediction[0][0] if prediction.shape[-1] == 1 else prediction[0][1]
        
        # Ensure probability is in [0, 1] range
        prob_malignant = float(np.clip(prob_malignant, 0, 1))
        prob_benign = 1.0 - prob_malignant
        predicted_class = 1 if prob_malignant >= THRESHOLD else 0
        
        return predicted_class, prob_malignant, prob_benign
    
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")

def format_probability(prob):
    """Format probability as percentage."""
    return f"{prob * 100:.2f}%"

def print_prediction_results(image_path, model_name, predicted_class, prob_malignant, prob_benign):
    """Print prediction results in formatted manner."""
    print_header("🔬 MELANOMA DETECTION RESULTS")
    
    print(f"\n  📁 Image:        {image_path}")
    print(f"  🤖 Model:        {model_name}")
    
    print("\n" + "-"*80)
    print(f"\n  🎯 DIAGNOSIS: {CLASS_NAMES[predicted_class].upper()}")
    print(f"\n  📊 PROBABILITIES:")
    print(f"     • Benign:     {format_probability(prob_benign)}")
    print(f"     • Malignant:  {format_probability(prob_malignant)}")
    print(f"\n  ⚙️  Threshold:    {format_probability(THRESHOLD)}")
    
    # Confidence indicator
    max_prob = max(prob_malignant, prob_benign)
    confidence = (max_prob - 0.5) * 2 * 100
    print(f"  💪 Confidence:   {confidence:.1f}%")
    
    print("\n" + "="*80 + "\n")

def list_available_models():
    """List all available models."""
    model_dir = Path(MODELS_PATH)
    models = sorted([f.stem for f in model_dir.glob("*.pkl")])
    
    print_header("📋 AVAILABLE MODELS")
    for i, model in enumerate(models, 1):
        print(f"  {i:2d}. {model}")
    print("\n")
    
    return models

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Test Melanoma Detection Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_image_prediction.py image.jpg EfficientNetV2B2
  python test_image_prediction.py melanoma.png EfficientNetV2B2_ML_ENSEMBLE
  python test_image_prediction.py path/to/image.jpg ResNet50 --list
        """
    )
    
    parser.add_argument(
        "image",
        nargs='?',
        help="Path to the image file to test"
    )
    parser.add_argument(
        "model",
        nargs='?',
        default="EfficientNetV2B2",
        help="Name of the model to use (default: EfficientNetV2B2)"
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available models"
    )
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list:
        list_available_models()
        return
    
    # Check if image path is provided
    if not args.image:
        print("Error: Image path is required.")
        print("\nUsage: python test_image_prediction.py <image_path> [model_name]")
        print("   or: python test_image_prediction.py --list")
        sys.exit(1)
    
    try:
        print_header("🚀 MELANOMA DETECTION TEST")
        
        # Load image
        print("\n📂 Loading image...")
        img_array = load_image(args.image)
        print(f"✓ Image loaded successfully (shape: {img_array.shape})")
        
        # Load model
        print(f"\n🤖 Loading model: {args.model}...")
        model_result = load_model(args.model)
        # Handle both tuple and single return values
        if isinstance(model_result, tuple):
            model, model_data = model_result
        else:
            model = model_result
            model_data = None
        print(f"✓ Model loaded successfully")
        
        # Make prediction
        print("\n⏳ Making prediction...")
        predicted_class, prob_malignant, prob_benign = make_prediction(
            model, 
            img_array, 
            args.model,
            model_data
        )
        print(f"✓ Prediction completed")
        
        # Print results
        print_prediction_results(
            args.image,
            args.model,
            predicted_class,
            prob_malignant,
            prob_benign
        )
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTo see available models, run:")
        print("   python test_image_prediction.py --list")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
