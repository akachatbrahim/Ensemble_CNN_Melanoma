#!/usr/bin/env python3
"""
Quick Start Diagnostic - Melanoma Detection Tester
=============================================

This is a simple Diagnostic showing how to use the melanoma detection system.

Run this script to see available options:
    python Diagnostic.py

Or test a specific image:
    python Diagnostic.py image.jpg
    python Diagnostic.py ./test_images/melanoma.png EfficientNetV2B2_ML_ENSEMBLE
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# ============================================================================
# SIMPLE DIAGNOSTIC AND GUIDE
# ============================================================================

def print_banner():
    """Print welcome banner."""
    banner = """
    
╔════════════════════════════════════════════════════════════════════════════════╗
║                   🏥 MELANOMA DETECTION - TEST SYSTEM 🏥                       ║
║                                                                                ║
║  Application: Image-based Melanoma Classification                             ║
║  Input:  Medical Image (JPG, PNG, BMP, etc.)                                  ║
║  Output: Diagnosis (Benign or Malignant) with Probability Percentages         ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print main menu."""
    menu = """
┌────────────────────────────────────────────────────────────────────────────────┐
│                           📋 AVAILABLE TESTERS                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  1. 🖼️  Command Line (Single Image)                                           │
│     python test_image_prediction.py <image_path> [model_name]                │
│                                                                                │
│  2. 🎮 Interactive Mode (Multiple Tests)                                      │
│     python test_interactive.py                                                │
│                                                                                │
│  3. 📊 Batch Processing (Multiple Images)                                     │
│     python test_batch.py <image_dir_or_file> [model_name]                    │
│                                                                                │
│  4. 📚 View Full Documentation                                                │
│     cat TEST_README.md                                                        │
│                                                                                │
│  5. 📝 List Available Models                                                  │
│     python test_image_prediction.py --list                                    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
    """
    print(menu)

def print_quick_examples():
    """Print quick examples."""
    examples = """
┌────────────────────────────────────────────────────────────────────────────────┐
│                          ⚡ QUICK EXAMPLES                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  # Test single image with default model                                       │
│  $ python test_image_prediction.py ./melanoma.jpg                             │
│                                                                                │
│  # Test single image with specific model                                      │
│  $ python test_image_prediction.py ./image.png ResNet50_ML_ENSEMBLE           │
│                                                                                │
│  # Interactive testing (recommended for multiple images)                      │
│  $ python test_interactive.py                                                 │
│                                                                                │
│  # Test entire directory                                                      │
│  $ python test_batch.py ./dataset/test_images/ EfficientNetV2B2               │
│                                                                                │
│  # List all available models                                                  │
│  $ python test_image_prediction.py --list                                     │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
    """
    print(examples)

def print_output_format():
    """Print output format example."""
    output = """
┌────────────────────────────────────────────────────────────────────────────────┐
│                       📊 OUTPUT FORMAT EXAMPLE                                 │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ════════════════════════════════════════════════════════════════════════════ │
│    🔬 MELANOMA DETECTION RESULTS                                             │
│  ════════════════════════════════════════════════════════════════════════════ │
│                                                                                │
│    📁 Image:        ./test_melanoma.jpg                                      │
│    🤖 Model:        EfficientNetV2B2                                          │
│                                                                                │
│  ──────────────────────────────────────────────────────────────────────────── │
│                                                                                │
│    🎯 DIAGNOSIS: MALIGNANT          ← Final classification result             │
│                                                                                │
│    📊 PROBABILITIES:                 ← Confidence scores                      │
│       • Benign:     32.45%                                                    │
│       • Malignant:  67.55%                                                    │
│                                                                                │
│    ⚙️  Threshold:    50.00%           ← Decision threshold (50%)              │
│    💪 Confidence:   35.10%           ← Model confidence level                 │
│                                                                                │
│  ════════════════════════════════════════════════════════════════════════════ │
│                                                                                │
│  Understanding the results:                                                  │
│  • Benign: Probability that the lesion is NOT cancerous                      │
│  • Malignant: Probability that the lesion IS cancerous                       │
│  • Higher confidence = More reliable diagnosis                               │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
    """
    print(output)

def print_models_info():
    """Print information about available models."""
    models = """
┌────────────────────────────────────────────────────────────────────────────────┐
│                      🤖 AVAILABLE MODELS CATEGORIES                            │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Deep Learning Models (11 architectures):                                     │
│  ┌─ ResNet50, MobileNetV2, EfficientNetV2 (B0, B2, B3, M)                    │
│  ├─ DenseNet (121, 169), VGG (16, 19)                                        │
│  └─ CNN, ConvNeXtSmall                                                       │
│                                                                                │
│  Hybrid Models (Deep Learning + Machine Learning):                           │
│  ├─ EfficientNetV2B2_ML, ResNet50_ML, DenseNet169_ML                        │
│  └─ [DL Model]_ML_ENSEMBLE variants                                          │
│                                                                                │
│  Ensemble Models (Multiple DL models):                                       │
│  ├─ Single Ensembles: ResNet50_EfficientNetV2B2_ENSEMBLE                     │
│  ├─ Dual Ensembles: EfficientNetV2B2_DenseNet169_ENSEMBLE                    │
│  └─ Triple Ensembles: ResNet50_EfficientNetV2B2_DenseNet169_ENSEMBLE         │
│                                                                                │
│  Recommended starting models:                                                │
│  🟢 EfficientNetV2B2              (balanced - good speed & accuracy)          │
│  🟡 ResNet50_ML_ENSEMBLE          (high accuracy - slower)                    │
│  🔵 MobileNetV2                   (fastest - good accuracy)                   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
    """
    print(models)

def print_installation_check():
    """Check and display installation status."""
    print("\n┌────────────────────────────────────────────────────────────────────────────────┐")
    print("│                          ✅ INSTALLATION CHECK                                 │")
    print("├────────────────────────────────────────────────────────────────────────────────┤")
    
    checks = []
    
    # Check Python
    try:
        import sys
        checks.append(("Python", f"{sys.version.split()[0]}", True))
    except:
        checks.append(("Python", "Not found", False))
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        checks.append(("TensorFlow", tf.__version__, True))
    except:
        checks.append(("TensorFlow", "Not installed", False))
    
    # Check Keras
    try:
        import keras
        checks.append(("Keras", keras.__version__, True))
    except:
        checks.append(("Keras", "Not installed", False))
    
    # Check PIL
    try:
        from PIL import Image
        checks.append(("PIL/Pillow", "Installed", True))
    except:
        checks.append(("PIL/Pillow", "Not installed", False))
    
    # Check NumPy
    try:
        import numpy as np
        checks.append(("NumPy", np.__version__, True))
    except:
        checks.append(("NumPy", "Not installed", False))
    
    # Check scikit-learn
    try:
        import sklearn
        checks.append(("scikit-learn", sklearn.__version__, True))
    except:
        checks.append(("scikit-learn", "Not installed", False))
    
    # Check models directory
    models_path = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma/Models")
    model_count = len(list(models_path.glob("*.pkl"))) if models_path.exists() else 0
    checks.append(("Models Available", f"{model_count} models", model_count > 0))
    
    # Print checks
    for package, version, status in checks:
        status_str = "✅" if status else "❌"
        print(f"│  {status_str} {package:20s} {version:30s}            │")
    
    print("└────────────────────────────────────────────────────────────────────────────────┘\n")

def print_requirements():
    """Print system requirements."""
    requirements = """
┌────────────────────────────────────────────────────────────────────────────────┐
│                        📋 SYSTEM REQUIREMENTS                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Minimum Requirements:                                                        │
│  • Python 3.8+                                                                │
│  • 4 GB RAM (8 GB recommended)                                               │
│  • 500 MB disk space for models                                              │
│                                                                                │
│  Required Python Packages:                                                   │
│  ✅ TensorFlow / Keras      - Deep Learning framework                        │
│  ✅ NumPy                   - Numerical computing                            │
│  ✅ Pillow (PIL)            - Image processing                               │
│  ✅ scikit-learn            - Machine Learning models                        │
│  ✅ tabulate (optional)     - Pretty printing tables                         │
│                                                                                │
│  Image Formats Supported:                                                    │
│  ✅ JPEG / JPG              ✅ PNG                                            │
│  ✅ BMP                     ✅ GIF                                            │
│  ✅ TIFF                                                                      │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
    """
    print(requirements)

def print_footer():
    """Print footer with important notes."""
    footer = """
┌────────────────────────────────────────────────────────────────────────────────┐
│                            📌 IMPORTANT NOTES                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ⚠️  Medical Disclaimer:                                                       │
│  • This system is for research/demonstration purposes only                    │
│  • Not approved for clinical diagnosis                                        │
│  • Always consult with qualified medical professionals                        │
│  • Never rely solely on AI for medical decisions                              │
│                                                                                │
│  💡 Tips for Best Results:                                                    │
│  • Use high-quality, clear images                                            │
│  • Ensure good lighting in the photo                                         │
│  • Test multiple images for consistency                                      │
│  • Use ensemble models for higher confidence                                 │
│  • Compare results across different model architectures                      │
│                                                                                │
│  📧 Support:                                                                  │
│  • Check TEST_README.md for detailed documentation                           │
│  • Use --help flag with any script for options                               │
│  • Review model performance metrics for accuracy info                        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
    """
    print(footer)


def run_demo_command(command_args):
    
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable] + command_args,
            cwd=os.path.dirname(__file__)
        )
        if result.returncode != 0:
            print(f"\n❌ Command failed with exit code {result.returncode}")
    except Exception as e:
        print(f"\n❌ Error running command: {e}")


def list_models():
    """Return list of available model names from the Models directory."""
    models_dir = Path(__file__).resolve().parent / "Models"
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return []

    models = sorted([f.stem for f in models_dir.glob("*.pkl")])
    return models


def choose_model_by_number(models):
    """Ask the user to choose a model by number from the list."""
    if not models:
        print("No models available.")
        return None

    default_index = 21
    if default_index > len(models):
        default_index = len(models)

    print("\nAvailable models:")
    for idx, model in enumerate(models, start=1):
        print(f"  {idx:2d}. {model}")

    while True:
        selection = input(f"\nChoose model number [default {default_index}, 0 to return]: ").strip()
        if selection == "":
            return models[default_index - 1]
        if selection == "0":
            return None
        if selection.isdigit():
            num = int(selection)
            if 1 <= num <= len(models):
                return models[num - 1]

        print(f"Invalid selection. Enter 0 or a number between 1 and {len(models)}.")


def interactive_mode():
    """Interactive mode for Diagnostic navigation."""
    while True:
        print("\n┌────────────────────────────────────────────────────────────────────────────────┐")
        print("│                          🧭 DIAGNOSTIC INTERACTIVE MODE                           │")
        print("├────────────────────────────────────────────────────────────────────────────────┤")
        print("│  1. Test single image                                                      │")
        print("│  2. Launch interactive tester                                               │")
        print("│  3. Run batch processing                                                    │")
        print("│  4. List available models                                                   │")
        print("│  5. Print documentation file path                                           │")
        print("│  q. Quit                                                                    │")
        print("└────────────────────────────────────────────────────────────────────────────────┘")

        choice = input("Choice [1-5, q]: ").strip().lower()
        if choice in ["q", "quit", "exit"]:
            print("\nBye !")
            break

        if choice == "1":
            image_path = input("Image path: ").strip()
            if not image_path:
                print("Image path is required.")
                continue

            models = list_models()
            if not models:
                continue

            model_name = choose_model_by_number(models)
            if model_name is None:
                print("Returning to main menu.")
                continue

            run_demo_command(["test_image_prediction.py", image_path, model_name])

        elif choice == "2":
            run_demo_command(["test_interactive.py"])

        elif choice == "3":
            path = input("Directory or file path to process: ").strip()
            if not path:
                print("Path is required.")
                continue

            models = list_models()
            if not models:
                continue

            model_name = choose_model_by_number(models)
            if model_name is None:
                print("Returning to main menu.")
                continue

            run_demo_command(["test_batch.py", path, model_name])

        elif choice == "4":
            run_demo_command(["test_image_prediction.py", "--list"])

        elif choice == "5":
            print("\nDocumentation file: TEST_README.md")
            print("Run: cat TEST_README.md")

        else:
            print("Invalid choice, please enter 1-5 or q.")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        # If arguments provided, show how to use them
        image_path = sys.argv[1]
        model_name = sys.argv[2] if len(sys.argv) > 2 else "EfficientNetV2B2"
        
        print_banner()
        print(f"\n🚀 Running test with:")
        print(f"   📁 Image: {image_path}")
        print(f"   🤖 Model: {model_name}")
        print(f"\nExecuting: python test_image_prediction.py {image_path} {model_name}\n")
        
        # Import and run the test
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "test_image_prediction.py", image_path, model_name],
                cwd=os.path.dirname(__file__)
            )
            sys.exit(result.returncode)
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            sys.exit(1)
    
    else:
        # Show the guide
        print_banner()
        print_installation_check()
        print_menu()
        print_quick_examples()
        print_output_format()
        print_models_info()
        print_requirements()
        print_footer()

        interactive_mode()
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
