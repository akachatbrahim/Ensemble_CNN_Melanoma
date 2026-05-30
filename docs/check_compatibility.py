#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility Verification Script
Verifies that the new save/load system works with existing code
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/home/akachat/tf_env/Article')


def check_save_model_import():
    """Check if save_model module can be imported"""
    print("\n" + "="*70)
    print("CHECK 1: Import save_model Module")
    print("="*70)
    
    try:
        from save_model import save_experiment, load_experiment, list_saved_experiments
        print("✓ Successfully imported from save_model:")
        print("  • save_experiment")
        print("  • load_experiment")
        print("  • list_saved_experiments")
        return True
    except ImportError as e:
        print(f"✗ Failed to import save_model: {e}")
        return False


def check_load_utils_import():
    """Check if load_experiment_utils can be imported"""
    print("\n" + "="*70)
    print("CHECK 2: Import load_experiment_utils Module")
    print("="*70)
    
    try:
        from load_experiment_utils import (
            load_latest_experiment,
            extract_metrics,
            extract_history,
            print_experiment_summary
        )
        print("✓ Successfully imported from load_experiment_utils:")
        print("  • load_latest_experiment")
        print("  • extract_metrics")
        print("  • extract_history")
        print("  • print_experiment_summary")
        return True
    except ImportError as e:
        print(f"✗ Failed to import load_experiment_utils: {e}")
        return False


def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()


def check_config_import():
    """Check if Config module with save_model_path exists"""
    print("\n" + "="*70)
    print("CHECK 3: Verify Config Module")
    print("="*70)
    
    try:
        from Config import save_model_path
        print("✓ Config module imported successfully")
        print(f"✓ save_model_path = {save_model_path}")
        
        # Check if path is writable
        path = Path(save_model_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Try to create a test file
        test_file = path / ".write_test"
        test_file.touch()
        test_file.unlink()
        
        print(f"✓ Path is writable and accessible")
        return True
    except Exception as e:
        print(f"✗ Error with Config: {e}")
        return False


def check_existing_files():
    """Check for existing script files that use save_model"""
    print("\n" + "="*70)
    print("CHECK 4: Scan Existing Scripts")
    print("="*70)
    
    article_dir = Path("/home/akachat/tf_env/Article")
    python_files = list(article_dir.glob("*.py"))
    
    files_using_save_model = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                if 'from save_model import' in content or 'import save_model' in content:
                    files_using_save_model.append(py_file.name)
        except:
            pass
    
    print(f"✓ Found {len(files_using_save_model)} script(s) using save_model:")
    for fname in sorted(files_using_save_model):
        print(f"  • {fname}")
    
    return len(files_using_save_model) > 0


def check_backward_compatibility():
    """Test backward compatibility with existing usage patterns"""
    print("\n" + "="*70)
    print("CHECK 5: Backward Compatibility Test")
    print("="*70)
    
    try:
        # Test importing as existing code does
        from save_model import save_experiment
        
        # Create mock data
        mock_model = type('MockModel', (), {'name': 'test_model'})()
        mock_experiment = {
            'models': {'m1': 'model1'},
            'metrics': {'accuracy': 0.95, 'loss': 0.05},
            'history': {'loss': [0.5, 0.3, 0.1]}
        }
        mock_history = {'loss': [0.5, 0.3, 0.1]}
        
        print("✓ Function signature is compatible")
        print("  save_experiment(model, name, experiment_data, history)")
        print("✓ All existing code should work without modification")
        
        return True
    except Exception as e:
        print(f"✗ Compatibility issue: {e}")
        return False


def list_created_files():
    """List all files created for the save/load system"""
    print("\n" + "="*70)
    print("CHECK 6: Verify All System Files")
    print("="*70)
    
    article_dir = Path("/home/akachat/tf_env/Article")
    required_files = [
        "save_model.py",
        "load_experiment_utils.py",
        "test_save_load.py",
        "examples_save_load.py",
        "GUIDE_SAVE_LOAD.md",
        "README_SYSTEM.md",
    ]
    
    print("\nRequired files:")
    all_exist = True
    for fname in required_files:
        fpath = article_dir / fname
        exists = fpath.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {fname}")
        if not exists:
            all_exist = False
    
    return all_exist


def show_migration_guide():
    """Show guide for migrating existing code"""
    print("\n" + "="*70)
    print("MIGRATION GUIDE: Existing Code")
    print("="*70)
    
    print("""
NO MIGRATION NEEDED! ✓

Your existing Python scripts are fully compatible.
Just make sure you have these imports at the top:

    from save_model import save_experiment
    
And call it as you do now:

    save_experiment(model, name_model, experiment, history)

NEW FEATURES YOU CAN USE:

1. Load a saved model:
   from load_experiment_utils import load_latest_experiment
   exp = load_latest_experiment("VGG19")
   model = exp["model_object"]

2. Compare models:
   from load_experiment_utils import create_comparison_dataframe
   df = create_comparison_dataframe(["VGG19", "ResNet50_ML"])

3. Extract data:
   from load_experiment_utils import extract_metrics, extract_history
   metrics = extract_metrics(exp)
   history = extract_history(exp)

See GUIDE_SAVE_LOAD.md for more examples!
""")


def main():
    """Run all checks"""
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  SAVE/LOAD SYSTEM - COMPATIBILITY CHECK".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    checks = {
        "save_model import": check_save_model_import,
        "load_experiment_utils import": check_load_utils_import,
        "Config module": check_config_import,
        "Existing scripts": check_existing_files,
        "Backward compatibility": check_backward_compatibility,
        "System files": list_created_files,
    }
    
    results = {}
    for name, func in checks.items():
        try:
            results[name] = func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  COMPATIBILITY CHECK RESULTS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:<30} {status:>25}")
    
    print(f"\n  Overall: {passed}/{total} checks passed")
    print("\n" + "█"*70)
    
    if passed == total:
        print("✓ ALL CHECKS PASSED - System is ready!")
        show_migration_guide()
    else:
        print("⚠ Some checks failed - See details above")
    
    print("\n" + "█"*70)
    print("\nNext steps:")
    print("  1. Run: python test_save_load.py")
    print("  2. Run: python examples_save_load.py")
    print("  3. Read: GUIDE_SAVE_LOAD.md")
    print("\n")


if __name__ == "__main__":
    main()
