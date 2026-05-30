#!/usr/bin/env python3
"""
Recovery tool for corrupted experiment files
Helps identify and clean up corrupted pickle files
"""

import os
import pickle
from pathlib import Path

def check_experiment_files(models_dir=None):
    """Check integrity of all saved experiment files."""
    
    if models_dir is None:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from src.config.Config import path_project
        models_dir = Path(path_project + "/ Models")
    else:
        models_dir = Path(models_dir)
    
    print("\n" + "="*70)
    print("EXPERIMENT FILE INTEGRITY CHECK")
    print("="*70 + "\n")
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return
    
    pkl_files = list(models_dir.glob("*.pkl"))
    
    if not pkl_files:
        print("No experiment files found.")
        return
    
    print(f"Found {len(pkl_files)} experiment file(s):\n")
    
    valid = []
    corrupted = []
    
    for pkl_file in sorted(pkl_files):
        name = pkl_file.stem
        size_mb = pkl_file.stat().st_size / (1024**2)
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if it has the expected structure
            has_metrics = bool(data.get('experiment_data', {}).get('metrics', {}))
            
            if has_metrics:
                metrics_count = len(data['experiment_data']['metrics'])
                print(f"✓ {name:<30} ({size_mb:7.2f} MB) - {metrics_count} models")
                valid.append(name)
            else:
                print(f"⚠ {name:<30} ({size_mb:7.2f} MB) - No metrics found")
                corrupted.append(name)
                
        except Exception as e:
            err_type = type(e).__name__
            print(f"✗ {name:<30} ({size_mb:7.2f} MB) - {err_type}")
            corrupted.append(name)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Valid:     {len(valid)} file(s)")
    print(f"✗ Corrupted: {len(corrupted)} file(s)")
    
    if corrupted:
        print("\n⚠️  CORRUPTED FILES:")
        for name in corrupted:
            print(f"   - {name}.pkl")
        
        print("\n💡 SOLUTIONS:")
        print("\n1. DELETE CORRUPTED FILES (Recommended)")
        print("   cd /home/akachat/tf_env/Ensemble_CNN_Melanoma/Models")
        for name in corrupted:
            print(f"   rm {name}.pkl")
        
        print("\n2. RE-RUN EXPERIMENTS")
        print("   /home/akachat/tf_env/bin/python main_project.py --run all")
        
        print("\n3. OR DELETE ALL AND START FRESH")
        print("   rm /home/akachat/tf_env/Ensemble_CNN_Melanoma/Models/*.pkl")
        print("   /home/akachat/tf_env/bin/python main_project.py --run 1")
        print("   (Then run more experiments as needed)")
    
    print("\n" + "="*70 + "\n")

def clean_corrupted_files(models_dir=None):
    """Remove corrupted pickle files."""
    
    if models_dir is None:
        models_dir = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma/Models")
    else:
        models_dir = Path(models_dir)
    corrupted = []
    
    for pkl_file in models_dir.glob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            has_metrics = bool(data.get('experiment_data', {}).get('metrics', {}))
            if not has_metrics:
                corrupted.append(pkl_file)
        except:
            corrupted.append(pkl_file)
    
    if corrupted:
        print(f"\n🗑️  Removing {len(corrupted)} corrupted file(s)...")
        for pkl_file in corrupted:
            pkl_file.unlink()
            print(f"   ✓ Deleted {pkl_file.name}")
        print(f"\n✓ Cleanup complete!")
        print(f"You can now re-run experiments to generate fresh files.")
    else:
        print("\n✓ No corrupted files found.")

if __name__ == "__main__":
    import sys
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--clean":
            clean_corrupted_files()
        else:
            check_experiment_files()
            
            # Ask user if they want to clean
            response = input("\nClean corrupted files? (y/n): ").strip().lower()
            if response == 'y':
                clean_corrupted_files()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
