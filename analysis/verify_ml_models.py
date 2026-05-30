#!/usr/bin/env python3
"""
Verification script to check if ml_models are properly populated in all model files.
"""

import os
import sys
import pickle
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import path_project

save_model_path = Path(path_project) / "Models"

# Expected model files (aligned with 7 experiment tasks + defaults)
expected_models = [
    "CNN_PERSONNEL",
    "DL",
    "DL_ML",
    "DL1_DL2_ML",
    "DL_BAGGING",
    "DL1_DL2_BAGGING",
    "DL_ML_BAGGING",
    "DL1_DL2_ML_BAGGING",
]

def verify_ml_models(models_dir=None):
    """Verify all saved model pickle files have proper structure."""
    
    if models_dir is None:
        models_dir = save_model_path
    
    models_dir_path = Path(models_dir)
    
    if not models_dir_path.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return False
    
    print("=" * 80)
    print("VERIFICATION: Checking ml_models in saved pickle files")
    print("=" * 80)
    
    valid_count = 0
    error_count = 0
    
    for model_name in expected_models:
        pkl_file = os.path.join(models_dir, f"{model_name}.pkl")
        
        if not os.path.exists(pkl_file):
            print(f"\n⚠ {model_name}.pkl - FILE NOT FOUND")
            error_count += 1
            continue
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract experiment data
            experiment_data = data.get("experiment_data", {})
            models = experiment_data.get("models", {})
            metrics = experiment_data.get("metrics", {})
            history = experiment_data.get("history", {})
            
            print(f"\n✓ {model_name}.pkl")
            print(f"  Size: {os.path.getsize(pkl_file) / (1024**2):.2f} MB")
            print(f"  Models in ml_models: {list(models.keys())}")
            print(f"  Metrics available for: {list(metrics.keys())}")
            print(f"  History type: {type(history).__name__}")
            
            # Validate structure
            if not models:
                print(f"  ⚠ WARNING: No models found in ml_models!")
                error_count += 1
            elif not metrics:
                print(f"  ⚠ WARNING: No metrics found!")
                error_count += 1
            else:
                valid_count += 1
                
        except Exception as e:
            print(f"\n❌ {model_name}.pkl - ERROR: {str(e)}")
            error_count += 1
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"✓ Valid:   {valid_count}/{len(expected_models)}")
    print(f"✗ Errors:  {error_count}/{len(expected_models)}")
    print("=" * 80)
    
    return error_count == 0

if __name__ == "__main__":
    try:
        success = verify_ml_models()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nVerification cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during verification: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
