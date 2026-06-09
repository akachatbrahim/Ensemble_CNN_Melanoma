#!/usr/bin/env python3
"""
Quick Demo - Test Melanoma Detection
=====================================
Batch test script for multiple images

Usage:
    python test_batch.py <image_dir_or_file> [model_name]
    
Examples:
    python test_batch.py ./test_images/
    python test_batch.py image.jpg EfficientNetV2B2_ML_ENSEMBLE
"""

import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
from tabulate import tabulate
from src.config.Config import save_model_path 


# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_PATH =   save_model_path
#Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma/Models/")
IMG_SIZE = (224, 224)
CLASS_NAMES = {0: "Benign", 1: "Malignant"}
THRESHOLD = 0.5

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_image(image_path):
    """Load and preprocess image."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        return None

def load_model(model_name):
    """Load trained model."""
    try:
        model_path = MODELS_PATH / f"{model_name}.pkl"
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'model' in data:
            model = data['model']
            if 'ML_ENSEMBLE' in model_name and 'experiment_data' in data:
                return model, data
            else:
                return model
        else:
            return data
    except Exception as e:
        print(f"❌ Error loading model '{model_name}': {str(e)}")
        return None

def normalize_image(img_array):
    """Normalize image."""
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
    """Make prediction."""
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

def get_image_files(path_arg):
    """Get list of images to test."""
    path = Path(path_arg)
    
    if path.is_file():
        # Single file
        if path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return [path]
        else:
            print(f"❌ Unsupported image format: {path.suffix}")
            return []
    
    elif path.is_dir():
        # Directory of images
        image_files = []
        for pattern in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
            image_files.extend(path.glob(pattern))
        for pattern in ['*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.GIF']:
            image_files.extend(path.glob(pattern))
        return sorted(set(image_files))
    
    else:
        print(f"❌ Path not found: {path}")
        return []

def list_models():
    """List available models."""
    models = sorted([f.stem for f in MODELS_PATH.glob("*.pkl")])
    print("\n" + "="*80)
    print("  AVAILABLE MODELS")
    print("="*80 + "\n")
    for i, model in enumerate(models, 1):
        print(f"  {i:2d}. {model}")
    print()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Batch test Melanoma Detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_batch.py ./test_images/
  python test_batch.py image.jpg EfficientNetV2B2
  python test_batch.py ./images/ ResNet50_ML_ENSEMBLE --list
        """
    )
    
    parser.add_argument(
        "path",
        nargs='?',
        help="Path to image file or directory containing images"
    )
    parser.add_argument(
        "model",
        nargs='?',
        default="EfficientNetV2B2",
        help="Model name (default: EfficientNetV2B2)"
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available models"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list:
        list_models()
        return
    
    # Require path
    if not args.path:
        print("❌ Error: Image path or directory is required.")
        print("\nUsage: python test_batch.py <image_path_or_dir> [model_name]")
        print("   or: python test_batch.py --list")
        sys.exit(1)
    
    # Get images
    image_files = get_image_files(args.path)
    if not image_files:
        sys.exit(1)
    
    print("\n" + "="*80)
    print(f"  MELANOMA DETECTION BATCH TEST")
    print("="*80)
    print(f"\n  📁 Images found: {len(image_files)}")
    print(f"  🤖 Model: {args.model}\n")
    
    # Load model
    print("🔄 Loading model...")
    model_result = load_model(args.model)
    if model_result is None:
        print("\n❌ Failed to load model")
        sys.exit(1)
    # Handle both tuple and single return values
    if isinstance(model_result, tuple):
        model, model_data = model_result
    else:
        model = model_result
        model_data = None
    print("✓ Model loaded\n")
    
    # Test images
    results = []
    print("⏳ Testing images...\n")
    
    for idx, image_path in enumerate(image_files, 1):
        # Load image
        img_array = load_image(str(image_path))
        if img_array is None:
            results.append([
                image_path.name,
                "ERROR",
                "N/A",
                "0.00%",
                "0.00%"
            ])
            continue
        
        # Predict
        predicted_class, prob_malignant, prob_benign = predict_image(
            model, 
            img_array, 
            args.model,
            model_data
        )
        
        if predicted_class is None:
            results.append([
                image_path.name,
                "ERROR",
                "N/A",
                "0.00%",
                "0.00%"
            ])
            continue
        
        # Store result
        diagnosis = CLASS_NAMES[predicted_class]
        results.append([
            image_path.name,
            diagnosis,
            f"{prob_benign*100:.2f}%",
            f"{prob_malignant*100:.2f}%",
            f"{max(prob_benign, prob_malignant)*100:.2f}%"
        ])
    
    # Display table
    print("\n" + "="*80)
    print("  RESULTS")
    print("="*80 + "\n")
    
    table = tabulate(
        results,
        headers=["Image", "Diagnosis", "Benign Prob", "Malignant Prob", "Confidence"],
        tablefmt="grid"
    )
    print(table)
    
    # Summary
    malignant_count = sum(1 for r in results if r[1] == "Malignant")
    benign_count = sum(1 for r in results if r[1] == "Benign")
    error_count = sum(1 for r in results if r[1] == "ERROR")
    
    print("\n" + "-"*80)
    print(f"\n  📊 SUMMARY:")
    print(f"     Benign:     {benign_count}")
    print(f"     Malignant:  {malignant_count}")
    print(f"     Errors:     {error_count}")
    print(f"     Total:      {len(image_files)}\n")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
