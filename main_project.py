#!/usr/bin/env python3
"""
Main Project Dashboard for Melanoma Detection Experiments
==========================================

Interactive CLI tool to:
- View available experiment scripts
- Run selected experimentsModel 2 (2nd selected): EfficientNetV2B2
    └─ Learning Rate: 0.0003, Weight Decay: 0.0003
- Monitor progress and results
- Analyze and compare model performance
- Generate comprehensive reports
"""

import sys
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime
import argparse
import os

# Add Ensemble_CNN_Melanoma directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.modules import save_model
from src.modules.Model import DL_MODELS
import time
from src.modules.memory_optimized_test import test_memory_optimized
#optimized configuration for memory-constrained systems
test_memory_optimized()


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_FILES = {
    "1": {
        "name": "SINGLE_DL_MODEL",
        "file": "experiments/single_dl_model.py",
        "description": "Ex : EfficientNet_V2B2 transfer learning model",
        "type": "DL Only"
    },
    "2": {
        "name": "HYBRID_DL_ML",
        "file": "experiments/hybrid_dl_ml.py",
        "description": "Ex : EfficientNet_V2B2 + Machine Learning hybrid model",
        "type": "Hybrid (DL+ML)"
    },
    "3": {
        "name": "DUAL_DL_ML",
        "file": "experiments/dual_dl_ml.py",
        "description": "Ex : (EfficientNet_V2B2 + DenseNet169) + ML hybrid",
        "type": "Hybrid (Multi-DL+ML)"
    },
    "4": {
        "name": "TRIPLE_DL_ENSEMBLE",
        "file": "experiments/triple_dl_ensemble.py",
        "description": "Ex : (EfficientNet_V2B2 + DenseNet169 + ResNet50) ENSEMBLE",
        "type": "DL ENSEMBLE"
    },
    "5": {
        "name": "DUAL_DL_ENSEMBLE",
        "file": "experiments/dual_dl_ensemble.py",
        "description": "Ex : (EfficientNet_V2B2 + DenseNet169) ENSEMBLE",
        "type": "DL ENSEMBLE"
    },
    "6": {
        "name": "HYBRID_DL_ML_ENSEMBLE",
        "file": "experiments/hybrid_dl_ml_ensemble.py",
        "description": "Ex : EfficientNet_V2B2 + ML ENSEMBLE (SVM, KNN, DT, RF)",
        "type": "Hybrid ENSEMBLE"
    },
    "7": {
        "name": "DUAL_HYBRID_ENSEMBLE",
        "file": "experiments/dual_hybrid_ensemble.py",
        "description": "Ex : (EfficientNet_V2B2 + DenseNet169) + ML ENSEMBLE",
        "type": "Hybrid ENSEMBLE"
    },
    "8": {
        "name": "TRIPLE_DL_ML",
        "file": "experiments/triple_dl_ml.py",
        "description": "Ex : (EfficientNet_V2B2 + DenseNet169 + ResNet50) + 6 ML classifiers",
        "type": "Hybrid (Triple-DL+ML)"
    },
    "9": {
        "name": "TRIPLE_HYBRID_ENSEMBLE",
        "file": "experiments/triple_hybrid_ensemble.py",
        "description": "Ex : (EfficientNet_V2B2 + DenseNet169 + ResNet50) + ML ENSEMBLE",
        "type": "Hybrid ENSEMBLE"
    }
}

ARTICLE_DIR = Path("/home/akachat/tf_env/Ensemble_CNN_Melanoma")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subheader(title):
    """Print formatted subheader."""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print('─' * 80)

def print_success(msg):
    """Print success message."""
    print(f"✓ {msg}")

def print_error(msg):
    """Print error message."""
    print(f"✗ {msg}")

def print_info(msg):
    """Print info message."""
    print(f"ℹ {msg}")

def print_warning(msg):
    """Print warning message."""
    print(f"⚠ {msg}")

def run_experiment(exp_file):
    """
    Run an experiment script.
    
    Parameters:
    -----------
    exp_file : str
        Path to the Python experiment file
    """
    try:
        print_info(f"Starting experiment: {exp_file}")
        start_time = datetime.now()
        
        # Use python3 or the venv python
        python_executable = "/home/akachat/tf_env/bin/python"
        if not Path(python_executable).exists():
            python_executable = "python3"
        
        result = subprocess.run(
            [python_executable, str(exp_file)],
            cwd=str(ARTICLE_DIR),
            text=True
        )
        
        elapsed = datetime.now() - start_time
        
        if result.returncode == 0:
            print_success(f"Experiment completed successfully ({elapsed})")
            return True
        else:
            print_error(f"Experiment failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print_error(f"Failed to run experiment: {str(e)}")
        return False

# ============================================================================
# ORGANIZE PKL FILES
# ============================================================================

def organize_pkl_files():
    """
    Organize PKL files: Move them to subfolders with configuration parameters (learning_rate, weight_decay).
    Creates structure: save/models_lr{coeff}e_neg{exp}_wd{coeff}e_neg{exp}/001_ModelName/, etc.
    Example: models_lr1e_neg5_wd1e_neg5/ (for 1e-5 and 1e-5)
    """
    print_header("📁 ORGANIZE MODEL FILES")
    
    # Get source directory (where PKL files are currently saved)
    source_dir = Path(Config.save_model_path)
    
    if not source_dir.exists():
        print_error(f"Source directory not found: {source_dir}")
        return
    
    # Get learning_rate and weight_decay from Config
    try:
        from src.config.Config import learning_rate, weight_decay
        
        # Convert to scientific notation: e.g., 1e-5 → "1e_neg5", 3e-4 → "3e_neg4"
        def format_scientific(value):
            if value == 0:
                return "0"
            formatted = f"{value:.0e}"  # Convert to scientific notation like "1e-05"
            # Replace - with _neg: "1e-05" → "1e_neg05"
            parts = formatted.split('e')
            if len(parts) == 2:
                coeff = parts[0]
                exp = parts[1].replace('-', '_neg')
                return f"{coeff}e_{exp}"
            return formatted
        
        lr_str = format_scientific(learning_rate)
        wd_str = format_scientific(weight_decay)
        config_name = f"models_lr{lr_str}_wd{wd_str}"
    except ImportError:
        print_warning("Could not import learning_rate and weight_decay from Config")
        print_info("Using default folder name: 'models'")
        config_name = "models"
    
    # Create organized models directory with config parameters inside save/
    models_dir = ARTICLE_DIR / "save" / config_name
    if not models_dir.exists():
         models_dir.mkdir(parents=True, exist_ok=True)
    else:
            print_warning(f"Directory already exists: {models_dir}")
            print_info("Try to execute a program again. ")
            exit()
    
    print_info(f"Target directory: save/{config_name}/\n")
    
    # Get all PKL files from save directory (not recursively, just top level)
    pkl_files = list(source_dir.glob("*.pkl"))
    
    if not pkl_files:
        print_warning("No PKL files found to organize")
        return
    
    print_info(f"Found {len(pkl_files)} PKL file(s) to organize\n")
    
    # Create subdirectories with sequential numbers
    for idx, pkl_file in enumerate(sorted(pkl_files), 1):
        model_name = pkl_file.stem  # Name without extension
        numbered_dir = models_dir / f"{idx:03d}_{model_name}"
        
        try:
            numbered_dir.mkdir(parents=True, exist_ok=True)
            
            # Move PKL file to numbered directory
            destination = numbered_dir / pkl_file.name
            shutil.move(str(pkl_file), str(destination))
            
            print_success(f"[{idx}] Moved: {model_name}")
            print_info(f"     └─ save/{config_name}/{numbered_dir.name}/")
            
        except Exception as e:
            print_error(f"Failed to move {model_name}: {str(e)}")
    
    print_success(f"\n✓ All files organized in: save/{config_name}/")
    print_info(f"Total models: {len(pkl_files)}")
    print(f"\nFolder structure created:")
    print(f"  models/{config_name}/")
    print(f"  ├── 001_ModelName1/")
    print(f"  ├── 002_ModelName2/")
    print(f"  └── ...\n")

# ============================================================================
# EXTRACT METRICS TO DOCUMENT
# ============================================================================

def menu_extract_metrics():
    """Extract metrics from pkl or JSON files to Word document."""
    print_header("📥 EXTRACT MODELS METRICS TO WORD DOCUMENT")
    
    print("\n📋 Available extraction methods:\n")
    print("  1. 📊 Extract from JSON Report (Recommended)")
    print("     └─ Parses the latest JSON report generated by experiments")
    print("  2. 📦 Extract from PKL Files - Direct (All metrics)")
    print("     └─ Directly extracts from pickle files in /models/")
    print("  3. 📦 Extract from PKL Files - Direct (All metrics)")
    print("     └─ Directly extracts from pickle files in /save/")
    print("  4. 📑 Extract from PKL Files - Organized by Phase")
    print("     └─ Extracts and organizes metrics by experimental phase")
    print("  5. 📂 Extract from Organized Models Structure")
    print("     └─ Extracts from models/models_lr*_wd*/ organized directory")
    print("  0. 🔙 Back to main menu\n")
    
    choice = input("Select extraction method (0-5): ").strip()
    
    if choice == "1":
        extract_metrics_from_json()
    elif choice == "2":
        extract_metrics_from_pkl(choice)
    elif choice == "3":        
        extract_metrics_from_pkl(choice)    
    elif choice == "4":
        extract_metrics_by_phase()
    elif choice == "5":
        extract_organized_pkl_to_word()
    elif choice == "0":
        return
    else:
        print_error("Invalid choice.")

def extract_metrics_from_json():
    """Extract metrics from JSON report to Word document."""
    print_header("📊 EXTRACTING FROM JSON REPORT")
    
    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"
    
    extract_script = ARTICLE_DIR / "extract_metrics_json_to_docx.py"
    
    if not extract_script.exists():
        print_error(f"Script not found: {extract_script}")
        return
    
    print_info("Starting extraction from JSON report...")
    print_info(f"Script: {extract_script}")
    
    try:
        result = subprocess.run(
            [python_exec, str(extract_script)],
            cwd=str(ARTICLE_DIR),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_success("✓ Extraction completed successfully!")
            
            # Show output
            if result.stdout:
                print("\n📋 Output:")
                print(result.stdout)
            
            # Look for generated Word files
            word_files = list(ARTICLE_DIR.glob("*Metrics*.docx")) + list(ARTICLE_DIR.glob("*Report*.docx"))
            if word_files:
                latest_word = max(word_files, key=lambda x: x.stat().st_mtime)
                size_kb = latest_word.stat().st_size / 1024
                print_success(f"📄 Generated: {latest_word.name} ({size_kb:.1f} KB)")
        else:
            print_error("Extraction failed!")
            if result.stderr:
                print_error(f"Error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print_error("Extraction timeout (exceeded 5 minutes)")
    except Exception as e:
        print_error(f"Exception: {str(e)}")

def extract_metrics_from_pkl(chx):
    """Extract metrics directly from PKL files to Word document."""
    print_header("📦 EXTRACTING FROM PKL FILES")
    
    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"
    
    extract_script = ARTICLE_DIR / "extract_models_metrics_to_docx.py"
    extract_save_script = ARTICLE_DIR / "extract_save_models_metrics_to_docx.py"
    
    if not extract_script.exists():
        print_error(f"Script not found: {extract_script}")
        return
    
    try:
        chx_idx = int(chx)
        if chx_idx == 2:
            result = subprocess.run(
                [python_exec, str(extract_script)],
                cwd=str(ARTICLE_DIR),
                #capture_output=True,
                text=True,
                timeout=600
            )
        else:
            print("chx")
            result = subprocess.run(
                [python_exec, str(extract_save_script)],
                cwd=str(ARTICLE_DIR),
                #capture_output=True,
                text=True,
                timeout=600
            )
        
        if result.returncode == 0:
            print_success("✓ Extraction completed successfully!")
            
            # Show output
            if result.stdout:
                print("\n📋 Output:")
                print(result.stdout)
            
            # Look for generated Word files
            word_files = list(ARTICLE_DIR.glob("*Metrics*.docx")) + list(ARTICLE_DIR.glob("*Report*.docx"))
            if word_files:
                latest_word = max(word_files, key=lambda x: x.stat().st_mtime)
                size_kb = latest_word.stat().st_size / 1024
                print_success(f"📄 Generated: {latest_word.name} ({size_kb:.1f} KB)")
        else:
            print_error("Extraction failed!")
            if result.stderr:
                print_error(f"Error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print_error("Extraction timeout (exceeded 10 minutes)")
    except Exception as e:
        print_error(f"Exception: {str(e)}")


def extract_metrics_by_phase():
    """Extract metrics from PKL files organized by experimental phase."""
    print_header("📑 EXTRACTING BY EXPERIMENTAL PHASE")
    
    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"
    
    extract_script = ARTICLE_DIR / "extract_metrics_to_word.py"
    
    if not extract_script.exists():
        print_error(f"Script not found: {extract_script}")
        return
    
    models_dir = ARTICLE_DIR / "Models"
    
    if not models_dir.exists():
        print_error(f"Models directory not found: {models_dir}")
        return
    
    pkl_count = len(list(models_dir.glob("*.pkl")))
    
    if pkl_count == 0:
        print_error("No PKL files found in Models directory")
        return
    
    print_info(f"Found {pkl_count} PKL files to extract")
    print_info("Starting extraction organized by experimental phase...")
    print_info(f"Script: {extract_script}")
    
    try:
        result = subprocess.run(
            [python_exec, str(extract_script)],
            cwd=str(ARTICLE_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print_success("✓ Extraction completed successfully!")
            
            # Show output
            if result.stdout:
                print("\n📋 Output:")
                print(result.stdout)
            
            # Look for generated Word files
            word_files = list(ARTICLE_DIR.glob("reports/*.docx"))
            if word_files:
                latest_word = max(word_files, key=lambda x: x.stat().st_mtime)
                size_kb = latest_word.stat().st_size / 1024
                print_success(f"📄 Generated: {latest_word.name} ({size_kb:.1f} KB)")
                print_success(f"📁 Location: {latest_word.parent}")
        else:
            print_error("Extraction failed!")
            if result.stderr:
                print_error(f"Error: {result.stderr}")
    
    except subprocess.TimeoutExpired:
        print_error("Extraction timeout (exceeded 10 minutes)")
    except Exception as e:
        print_error(f"Exception: {str(e)}")


def extract_organized_pkl_to_word():
    """Extract metrics from organized PKL structure (models/models_lr*_wd*/) to Word document."""
    print_header("📂 EXTRACTING FROM ORGANIZED MODELS STRUCTURE")
    
    models_dir = ARTICLE_DIR / "models"
    
    if not models_dir.exists():
        print_error(f"Models directory not found: {models_dir}")
        return
    
    # Find organized models directories
    organized_dirs = sorted(list(models_dir.glob("models_lr*_wd*")))
    
    if not organized_dirs:
        print_error("No organized models directories found in /models/ (models_lr*_wd*/)")
        print_info("Please run 'Organize Models' first to organize PKL files by configuration")
        return
    
    # Display menu for user selection
    print("\n📁 Available organized models directories:\n")
    for idx, dir_path in enumerate(organized_dirs, 1):
        pkl_count = len(list(dir_path.glob("*.pkl")))
        print(f"  {idx}. {dir_path.name}")
        print(f"     └─ {pkl_count} PKL files")
    
    print(f"\n  0. Cancel\n")
    
    choice = input("Select directory (0-" + str(len(organized_dirs)) + "): ").strip()
    
    try:
        choice_idx = int(choice)
        if choice_idx == 0:
            print_info("Cancelled")
            return
        elif 1 <= choice_idx <= len(organized_dirs):
            selected_dir = organized_dirs[choice_idx - 1]
        else:
            print_error("Invalid choice")
            return
    except ValueError:
        print_error("Invalid input")
        return
    
    print_info(f"Processing {selected_dir.name}...")
    
    # Run the extraction script with the selected directory as argument
    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"
    
    extract_script = ARTICLE_DIR / "extract_organized_pkl_to_word.py"
    
    if not extract_script.exists():
        print_error(f"Script not found: {extract_script}")
        return
    
    try:
        result = subprocess.run(
            [python_exec, str(extract_script), str(selected_dir)],
            cwd=str(ARTICLE_DIR),
            timeout=600
        )
        
        if result.returncode == 0:
            print_success("✓ Extraction completed successfully!")
            
            # Look for generated Word files
            word_files = list(ARTICLE_DIR.glob("reports/*Organized*.docx"))
            if word_files:
                latest_word = max(word_files, key=lambda x: x.stat().st_mtime)
                size_kb = latest_word.stat().st_size / 1024
                print_success(f"📄 Generated: {latest_word.name} ({size_kb:.1f} KB)")
                print_success(f"📁 Location: {latest_word.parent}")
        else:
            print_error("Extraction failed!")
    
    except subprocess.TimeoutExpired:
        print_error("Extraction timeout (exceeded 10 minutes)")
    except Exception as e:
        print_error(f"Exception: {str(e)}")


def menu_main():
    """Main menu."""
    while True:
        print_header("MELANOMA DETECTION - MAIN PROJECT DASHBOARD")
        print("\n📋 Choose an option:\n")
        print("  1. 🔬 Run Experiments")
        print("  2. 📊 View Results")
        print("  3. 📈 Analyze Models")
        print("  4. 📄 Generate Report")
        print("  5. 📉 Plot Statistics")
        print("  6. 🔄 List All Experiments")
        print("  7. 📁 Organize Models (Move PKL to subfolders within /models/)")
        print("  8. 📥 Extract Models Metrics to Word")
        print("  9. 🎨 Extract Plots from PKL Files")
        print("  10. 🎨 Generate Plots from PKL Files")
        print("  11. 🚀 Run Full Test Suite (All Experiments)")
        print("  12. 🧪 Diagnostic Image")
        print("  13. ℹ️  About")
        print("  0. ❌ Exit\n")
        
        choice = input("Enter your choice (0-13): ").strip()
        
        if choice == "1":
            menu_run_experiments()
        elif choice == "2":
            menu_view_results()
        elif choice == "3":
            menu_analyze_models()
        elif choice == "4":
            menu_generate_report()
        elif choice == "5":
            menu_plot_statistics()
        elif choice == "6":
            show_all_experiments()
            input("\nPress Enter to continue...")
        elif choice == "7":
            organize_pkl_files()
            input("\nPress Enter to continue...")
        elif choice == "8":
            menu_extract_metrics()
            input("\nPress Enter to continue...")
        elif choice == "9":
            extract_pkl_to_word()
            input("\nPress Enter to continue...")
        elif choice == "10":
            extract_plots_from_pkl()
            input("\nPress Enter to continue...")
        elif choice == "11":
            run_full_test_suite()
            input("\nPress Enter to continue...")
        elif choice == "12":
            run_interactive_test()
            input("\nPress Enter to continue...")
        elif choice == "13":
            show_about()
            input("\nPress Enter to continue...")
        elif choice == "0":
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print_error("Invalid choice. Please try again.")
            time.sleep(1)

def configure_parameters_for_experiment(exp_id):
    """
    Configure model parameters for experiments that need them.
    
    Args:
        exp_id: Experiment ID (e.g., "4", "5", "8", "9")
        
    Returns:
        Path to config file or None
    """
    # Experiments that need 3 models
    experiments_3_models = ["4", "8", "9"]  # TRIPLE_DL_ENSEMBLE, TRIPLE_DL_ML, TRIPLE_HYBRID_ENSEMBLE
    # Experiments that need 2 models
    experiments_2_models = ["3", "5", "7"]  # DUAL_DL_ML, DUAL_DL_ENSEMBLE, DUAL_HYBRID_ENSEMBLE
    
    model_selector = ModelSelector()
    
    if exp_id in experiments_3_models:
        print_info("This experiment requires 3 DL models")
        selected_models = model_selector.select_three_models()
    elif exp_id in experiments_2_models:
        print_info("This experiment requires 2 DL models")
        selected_models = model_selector.select_two_models()
    else:
        return None
    
    # Ask for parameters
    model_params = ask_model_parameters(selected_models)
    config_file = save_model_parameters_config(model_params, selected_models)
    
    return config_file

def menu_run_experiments():
    """Run experiments menu."""
    while True:
        print_header("🔬 RUN EXPERIMENTS")
        print("\n📁 Available Experiments:\n")
        
        for key, exp in sorted(EXPERIMENT_FILES.items()):
            print(f"  {key}. {exp['name']:30} [{exp['type']}]")
            print(f"     {exp['description']}")
        
        print(f"\n  {len(EXPERIMENT_FILES)+1}. ▶️  Run ALL experiments")
        print("  0. 🔙 Back to main menu\n")
        
        choice = input("Select experiment to run (0-9): ").strip()
        
        if choice == "0":
            break
        elif choice == str(len(EXPERIMENT_FILES)+1):
            run_all_experiments()
        elif choice in EXPERIMENT_FILES:
            exp_info = EXPERIMENT_FILES[choice]
            
            # Configure parameters for ENSEMBLE experiments
            experiments_needing_config = ["3", "4", "5", "7", "8", "9"]
            if choice in experiments_needing_config:
                print()
                configure_parameters_for_experiment(choice)
            
            confirm = input(f"\nRun '{exp_info['name']}'? (y/n): ").strip().lower()
            if confirm == 'y':
                exp_path = ARTICLE_DIR / exp_info['file']
                if exp_path.exists():
                    run_experiment(exp_path)
                    input("\nPress Enter to continue...")
                else:
                    print_error(f"File not found: {exp_path}")
                    input("\nPress Enter to continue...")
            else:
                print_info("Cancelled.")
        else:
            print_error("Invalid choice.")
            time.sleep(1)

def menu_view_results():
    """View results menu."""
    while True:
        print_header("📊 VIEW RESULTS")
        
        experiments = save_model.list_saved_experiments()
        
        if not experiments:
            print("\n❌ No saved experiments found.")
            input("\nPress Enter to continue...")
            break
        
        print(f"\n📁 Saved Experiments ({len(experiments)}):\n")
        
        for i, exp_name in enumerate(experiments, 1):
            exp_path = Path(Config.save_model_path) / f"{exp_name}.pkl"
            size_mb = exp_path.stat().st_size / (1024**2) if exp_path.exists() else 0
            print(f"  {i}. {exp_name:35} ({size_mb:7.2f} MB)")
        
        print("\n  0. 🔙 Back to main menu\n")
        
        choice = input(f"Select experiment to view (0-{len(experiments)}): ").strip()
        
        if choice == "0":
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(experiments):
                show_experiment_details(experiments[idx])
                input("\nPress Enter to continue...")
            else:
                print_error("Invalid choice.")
                time.sleep(1)
        except ValueError:
            print_error("Please enter a valid number.")
            time.sleep(1)

def menu_analyze_models():
    """Analyze models menu."""
    while True:
        print_header("📈 ANALYZE MODELS")
        
        experiments = save_model.list_saved_experiments()
        
        if not experiments:
            print("\n❌ No saved experiments found.")
            input("\nPress Enter to continue...")
            break
        
        print(f"\n📁 Available Experiments ({len(experiments)}):\n")
        
        for i, exp_name in enumerate(experiments, 1):
            print(f"  {i}. {exp_name}")
        
        print(f"\n  {len(experiments)+1}. 📊 Compare all models")
        print("  0. 🔙 Back to main menu\n")
        
        choice = input(f"Select experiment to analyze (0-{len(experiments)+1}): ").strip()
        
        if choice == "0":
            break
        elif choice == str(len(experiments)+1):
            analyze_all_models(experiments)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(experiments):
                    analyze_single_model(experiments[idx])
                    input("\nPress Enter to continue...")
                else:
                    print_error("Invalid choice.")
                    time.sleep(1)
            except ValueError:
                print_error("Please enter a valid number.")
                time.sleep(1)

def menu_generate_report():
    """Generate report menu."""
    print_header("📄 GENERATE REPORT")
    
    experiments = save_model.list_saved_experiments()
    
    if not experiments:
        print("\n❌ No saved experiments found.")
        input("\nPress Enter to continue...")
        return
    
    print(f"\n✓ Found {len(experiments)} saved experiments")
    
    report_type = input("\nGenerate report type:\n  1. Summary (text)\n  2. Detailed (JSON)\n  0. Cancel\n\nChoice: ").strip()
    
    if report_type == "1":
        generate_summary_report(experiments)
    elif report_type == "2":
        generate_detailed_report(experiments)
    
    input("\nPress Enter to continue...")

def menu_plot_statistics():
    """Plot statistics menu."""
    print_header("📉 PLOT STATISTICS")
    
    experiments = save_model.list_saved_experiments()
    
    if not experiments:
        print("\n❌ No saved experiments found.")
        input("\nPress Enter to continue...")
        return
    
    print(f"\n📁 Available Experiments ({len(experiments)}):\n")
    
    for i, exp_name in enumerate(experiments, 1):
        exp_path = Path(Config.save_model_path) / f"{exp_name}.pkl"
        size_mb = exp_path.stat().st_size / (1024**2) if exp_path.exists() else 0
        print(f"  {i}. {exp_name:35} ({size_mb:7.2f} MB)")
    
    print("\n  0. 🔙 Back to main menu\n")
    
    choice = input("Select experiment to plot (0-9): ").strip()
    
    if choice == "0":
        return
    
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(experiments):
            plot_experiment(experiments[idx])
            input("\nPress Enter to continue...")
        else:
            print_error("Invalid choice.")
            time.sleep(1)
    except ValueError:
        print_error("Please enter a valid number.")
        time.sleep(1)

# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def show_all_experiments():
    """Display all available experiments."""
    print_header("🔬 ALL AVAILABLE EXPERIMENTS")
    
    print("\n📊 Summary:\n")
    print(f"{'ID':<4} {'Name':<30} {'Type':<20} {'Description':<35}")
    print("─" * 90)
    
    for key, exp in sorted(EXPERIMENT_FILES.items()):
        exp_type = exp['type']
        desc = exp['description'][:32] + "..." if len(exp['description']) > 32 else exp['description']
        print(f"{key:<4} {exp['name']:<30} {exp_type:<20} {desc:<35}")

# ============================================================================
# MODEL SELECTOR CLASS
# ============================================================================

class ModelSelector:
    """Gestionnaire de sélection des modèles Deep Learning."""
    
    def __init__(self):
        self.selected_models = {}
    
    def select_three_models(self):
        """
        Demander à l'utilisateur de sélectionner 3 modèles DL DIFFÉRENTS.
        Ces 3 modèles seront utilisés pour toutes les expériences.
        
        Returns:
            list: Liste de 3 IDs de modèles différents
        """
        print_header("🧬 SÉLECTION DE 3 MODÈLES DEEP LEARNING")
        print_info("Sélectionnez 3 modèles DL DIFFÉRENTS qui seront utilisés")
        print_info("pour automatiser l'exécution de TOUS les cas d'expériences")
        print()
        
        selected = []
        
        for i in range(1, 4):
            print_subheader(f"Modèle {i}/3")
            print_info("Modèles disponibles:")
            print()
            
            for key in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
                model_name = DL_MODELS[key]['name']
                status = " ✓" if key in selected else ""
                print(f"  {key}. {model_name}{status}")
            
            print()
            
            if selected:
                print_info("Modèles déjà sélectionnés:")
                for idx, model_id in enumerate(selected, 1):
                    print_info(f"  {idx}. {DL_MODELS[model_id]['name']}")
                print()
            
            while True:
                choice = input(f"  Sélectionnez le modèle {i} (1-{len(DL_MODELS)}): ").strip()
                
                if choice == "":
                    continue
                
                if choice in DL_MODELS:
                    if choice not in selected:
                        selected.append(choice)
                        print_success(f"✓ Sélectionné: {DL_MODELS[choice]['name']}")
                        print()
                        break
                    else:
                        print_error(f"Le modèle {DL_MODELS[choice]['name']} est déjà sélectionné")
                        print_info("Choisissez un modèle différent")
                        print()
                else:
                    print_error(f"Choix invalide. Sélectionnez 1-{len(DL_MODELS)}")
        
        print_header("✅ MODÈLES SÉLECTIONNÉS")
        for i, model_id in enumerate(selected, 1):
            print_success(f"Modèle {i}: {DL_MODELS[model_id]['name']}")
        print()
        
        return selected

    def select_two_models(self):
        """
        Demander à l'utilisateur de sélectionner 2 modèles DL DIFFÉRENTS.
        Ces 2 modèles seront utilisés pour les expériences.
        
        Returns:
            list: Liste de 2 IDs de modèles différents
        """
        print_header("🧬 SÉLECTION DE 2 MODÈLES DEEP LEARNING")
        print_info("Sélectionnez 2 modèles DL DIFFÉRENTS pour cette expérience")
        print()
        
        selected = []
        
        for i in range(1, 3):
            print_subheader(f"Modèle {i}/2")
            print_info("Modèles disponibles:")
            print()
            
            for key in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
                model_name = DL_MODELS[key]['name']
                status = " ✓" if key in selected else ""
                print(f"  {key}. {model_name}{status}")
            
            print()
            
            if selected:
                print_info("Modèle déjà sélectionné:")
                for idx, model_id in enumerate(selected, 1):
                    print_info(f"  {idx}. {DL_MODELS[model_id]['name']}")
                print()
            
            while True:
                choice = input(f"  Sélectionnez le modèle {i} (1-{len(DL_MODELS)}): ").strip()
                
                if choice == "":
                    continue
                
                if choice in DL_MODELS:
                    if choice not in selected:
                        selected.append(choice)
                        print_success(f"✓ Sélectionné: {DL_MODELS[choice]['name']}")
                        print()
                        break
                    else:
                        print_error(f"Le modèle {DL_MODELS[choice]['name']} est déjà sélectionné")
                        print_info("Choisissez un modèle différent")
                        print()
                else:
                    print_error(f"Choix invalide. Sélectionnez 1-{len(DL_MODELS)}")
        
        print_header("✅ MODÈLES SÉLECTIONNÉS")
        for i, model_id in enumerate(selected, 1):
            print_success(f"Modèle {i}: {DL_MODELS[model_id]['name']}")
        print()
        
        return selected


# ============================================================================
# PARAMETER CONFIGURATION FUNCTIONS
# ============================================================================

def ask_model_parameters(selected_models):
    """
    Load learning rate and weight decay from Config.py for each selected model.
    These parameters will be FIXED for all experiments using these 3 models.
    
    Args:
        selected_models: List of 3 model IDs
        
    Returns:
        dict: Mapping from model_id to (learning_rate, weight_decay) tuple
    """
    print_header("⚙️  CONFIGURATION DES PARAMÈTRES - LR ET WEIGHT DECAY")
    print_info("✓ Paramètres CHARGÉS AUTOMATIQUEMENT depuis Config.py")
    print_info("Ces paramètres sont FIXES pour tous les experiments")
    print()
    
    # Load parameters from Config.py
    from src.config.Config import (
        learning_rate_one, weight_decay_one,
        learning_rate_two, weight_decay_two,
        learning_rate_three, weight_decay_three
    )
    
    config_params = [
        (learning_rate_one, weight_decay_one),
        (learning_rate_two, weight_decay_two),
        (learning_rate_three, weight_decay_three)
    ]
    
    model_params = {}
    
    for i, model_id in enumerate(selected_models, 1):
        model_name = DL_MODELS[model_id]['name']
        lr_value, wd_value = config_params[i-1]
        
        model_params[model_id] = (lr_value, wd_value)
        print_success(f"✓ {model_name}: LR={lr_value}, WD={wd_value}")
    
    print()
    return model_params

def save_model_parameters_config(model_params, selected_models):
    """
    Save model parameters to a JSON config file for use by experiments.
    
    Args:
        model_params: Dict mapping model_id to (lr, wd) tuples
        selected_models: List of 3 selected model IDs
        
    Returns:
        Path: Path to saved config file
    """
    config_data = {
        "selected_models": selected_models,
        "timestamp": datetime.now().isoformat(),
        "parameters": {}
    }
    
    for model_id in selected_models:
        lr, wd = model_params[model_id]
        model_name = DL_MODELS[model_id]['name']
        config_data["parameters"][model_id] = {
            "model_name": model_name,
            "learning_rate": float(lr),
            "weight_decay": float(wd)
        }
    
    config_file = ARTICLE_DIR / "current_model_parameters.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        print_success(f"Paramètres sauvegardés: {config_file}")
        return config_file
    except Exception as e:
        print_error(f"Erreur sauvegarde paramètres: {str(e)}")
        return None

def run_all_experiments():
    """Run all experiments sequentially with model selection and parameter configuration."""
    
    # ========================================================================
    # ÉTAPE 2: Sélectionner 3 modèles pour les autres expériences
    # ========================================================================
    print_header("🤖 ÉTAPE 2: SÉLECTION DE 3 MODÈLES")
    print_info("Sélectionnez 3 modèles pour les expériences")
    
    model_selector = ModelSelector()
    selected_models = model_selector.select_three_models()
    
    print_success(f"✓ 3 modèles sélectionnés:")
    for i, model_id in enumerate(selected_models, 1):
        print_info(f"  {i}. {DL_MODELS[model_id]['name']}")
    
    # ========================================================================
    # ÉTAPE 2.5: Configurer les paramètres (LR et WD) pour chaque modèle
    # ========================================================================
    model_params = ask_model_parameters(selected_models)
    config_file = save_model_parameters_config(model_params, selected_models)
    
    print_header("✅ CONFIGURATION DES PARAMÈTRES COMPLÉTÉE")
    print_info("Les paramètres suivants seront FIXES pour tous les experiments:")
    print()
    for model_id in selected_models:
        lr, wd = model_params[model_id]
        model_name = DL_MODELS[model_id]['name']
        print(f"  {model_name}:")
        print(f"    └─ Learning Rate: {lr}")
        print(f"    └─ Weight Decay: {wd}")
    print()
    
    # ========================================================================
    # ÉTAPE 3: Exécuter tous les expériences
    # ========================================================================
    confirm = input("\n⚠️  Proceed to run all experiments with these parameters? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print_header("🔬 EXÉCUTION DE TOUS LES EXPÉRIENCES")
    
    results = {}
    total = len(EXPERIMENT_FILES)
    
    for idx, (key, exp) in enumerate(sorted(EXPERIMENT_FILES.items()), 1):
        print(f"\n[{idx}/{total}] Running: {exp['name']}")
        print("─" * 80)
        
        exp_path = ARTICLE_DIR / exp['file']
        
        if exp_path.exists():
            success = run_experiment(exp_path)
            results[exp['name']] = "✓ Success" if success else "✗ Failed"
        else:
            results[exp['name']] = "✗ File not found"
            print_error(f"File not found: {exp_path}")
    
    print_header("📊 EXECUTION SUMMARY")
    
    for exp_name, status in results.items():
        print(f"{status} - {exp_name}")
    
    input("\nPress Enter to continue...")

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def show_experiment_details(exp_name):
    """Show detailed information about an experiment."""
    print_header(f"📋 EXPERIMENT DETAILS: {exp_name}")
    
    info = save_model.get_experiment_info(exp_name)
    
    if not info:
        print_error("Could not load experiment information.")
        return
    
    print(f"\n📦 Name: {info['name']}")
    
    metrics = info.get('metrics', {})
    if metrics:
        print(f"\n📊 Models and Metrics ({len(metrics)} models):")
        print("─" * 80)
        
        for model_name, metric_dict in metrics.items():
            print(f"\n  {model_name}:")
            if isinstance(metric_dict, dict):
                for metric_key, metric_val in metric_dict.items():
                    if metric_key not in ['y_true', 'y_pred']:
                        if isinstance(metric_val, (int, float)):
                            print(f"    {metric_key:<20}: {metric_val:.4f}")
                        else:
                            print(f"    {metric_key:<20}: {metric_val}")
    else:
        print_info("⚠️  No metrics available for this experiment.")
        print("    (File may be corrupted or metrics were not saved)")

def analyze_single_model(exp_name):
    """Analyze a single model."""
    print_header(f"📈 MODEL ANALYSIS: {exp_name}")
    
    info = save_model.get_experiment_info(exp_name)
    
    if not info:
        print_error("Could not load experiment information.")
        return
    
    metrics = info.get('metrics', {})
    
    if not metrics:
        print_info("⚠️  No metrics available for this experiment.")
        print("    (File may be corrupted or metrics were not saved)")
        return
    
    print("\n📊 Performance Metrics Comparison:\n")
    print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Sensitivity':<12} {'Specificity':<12} {'AUC-ROC':<10}")
    print("─" * 100)
    
    for model_name, metric_dict in metrics.items():
        acc = metric_dict.get('accuracy', 0)
        prec = metric_dict.get('precision', 0)
        sens = metric_dict.get('sensitivity', 0)
        spec = metric_dict.get('specificity', 0)
        auc = metric_dict.get('auc-roc', 0)
        
        print(f"{model_name:<35} {acc:<12.4f} {prec:<12.4f} {sens:<12.4f} {spec:<12.4f} {auc:<10.4f}")

def analyze_all_models(experiments):
    """Compare all saved models."""
    print_header("📊 COMPARING ALL MODELS")
    
    all_metrics = {}
    skipped = []
    
    for exp_name in experiments:
        info = save_model.get_experiment_info(exp_name)
        if info and info.get('metrics'):
            all_metrics[exp_name] = info.get('metrics', {})
        elif info:
            skipped.append(exp_name)
    
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} experiment(s) with empty/corrupted metrics:")
        for exp in skipped:
            print(f"   - {exp}")
    
    if not all_metrics:
        print_error("No valid metrics available in any experiment.")
        return
    
    print(f"\n📊 Comparing {len(all_metrics)} experiment(s) with valid metrics:\n")
    print("🏆 Best Models by Metric:\n")
    
    metrics_to_compare = ['accuracy', 'precision', 'sensitivity', 'specificity', 'auc-roc']
    
    for metric_name in metrics_to_compare:
        best_score = 0
        best_model = None
        best_exp = None
        
        for exp_name, models in all_metrics.items():
            for model_name, metric_dict in models.items():
                score = metric_dict.get(metric_name, 0)
                if isinstance(score, (int, float)) and score > best_score:
                    best_score = score
                    best_model = model_name
                    best_exp = exp_name
        
        if best_model:
            print(f"  🥇 {metric_name.upper():<15}: {best_score:.4f}")
            print(f"     📦 Model: {best_model}")
            print(f"     📁 Experiment: {best_exp}\n")

# ============================================================================
# REPORT FUNCTIONS
# ============================================================================

def generate_summary_report(experiments):
    """Generate a summary report."""
    print_header("📄 EXPERIMENT SUMMARY REPORT")
    
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Total Experiments: {len(experiments)}\n")
    
    print("Experiments:\n")
    for i, exp_name in enumerate(experiments, 1):
        print(f"  {i}. {exp_name}")
    
    print("\n" + "─" * 80)
    print("\n✓ Report generated successfully.")

def generate_detailed_report(experiments):
    """Generate a detailed JSON report."""
    print_header("📄 DETAILED EXPERIMENT REPORT")
    
    report = {
        "generated": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "experiments": {}
    }
    
    for exp_name in experiments:
        try:
            info = save_model.get_experiment_info(exp_name)
            if info:
                report["experiments"][exp_name] = info
        except Exception as e:
            print_info(f"Could not load {exp_name}: {str(e)}")
    
    # Save report
    report_path = ARTICLE_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print_success(f"Report saved: {report_path}")
    except Exception as e:
        print_error(f"Failed to save report: {str(e)}")

def plot_experiment(exp_name):
    """
    Plot statistics and visualizations for an experiment using Statistics.py
    
    Parameters:
    -----------
    exp_name : str
        Name of the experiment to plot (without .pkl or .h5 extension)
    """
    print_header(f"📉 PLOTTING STATISTICS: {exp_name}")
    
    try:
        print_info(f"Loading plots for {exp_name}...")

        print("(This may take a moment to generate visualizations)\n")
        
        # Import Statistics module and call plot_model
        from src.utils.statistics import plot_model
        from src.config.Config import PLOTS_DIR
        
        # Construct full path to model file
        from pathlib import Path
        model_path = Path(Config.save_model_path) / exp_name
        
        # Call plot_model with the full path
        plot_model(str(model_path))
        
        print_success("Plot generation completed!")
        print_info(f"Plots saved to: {PLOTS_DIR}")
        print("\n✓ Generated plots:")
        print(f"   - {exp_name}_confusion_matrix.png")
        print(f"   - {exp_name}_metrics.png")
        print(f"   - {exp_name}_training_history.png (if available)")
        input("\nPress Enter to continue...")
        
    except ImportError as e:
        print_error(f"Could not import Statistics module: {e}")
        print_info("Statistics.py may not be available")
    except FileNotFoundError as e:
        print_error(f"Could not find experiment file: {exp_name}")
        print_info(f"Expected: {Config.save_model_path}/{exp_name}.pkl or .h5")
    except Exception as e:
        print_error(f"Error plotting experiment: {e}")
        print_info(f"Experiment '{exp_name}' may not have required data for visualization")
        import traceback
        traceback.print_exc()

def extract_pkl_to_word():
    """Extract plot data from PKL files to Word document."""
    print_header("🎨 EXTRACT PLOTS FROM PKL FILES")
    
    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"
    
    extract_script = ARTICLE_DIR / "extract_pkl_to_word.py"
    
    if not extract_script.exists():
        print_error(f"Script not found: {extract_script}")
        return
    
    print_info("Starting plot extraction from PKL files...")
    print_info(f"Script: {extract_script}")
    print()
    
    try:
        result = subprocess.run(
            [python_exec, str(extract_script)],
            cwd=str(ARTICLE_DIR),
            text=True
        )
        
        if result.returncode == 0:
            print_success("✓ Plot extraction completed successfully!")
            
            # Check generated Word files
            word_files = list(ARTICLE_DIR.glob("*PKL*Report*.docx"))
            if word_files:
                latest_word = max(word_files, key=lambda x: x.stat().st_mtime)
                size_kb = latest_word.stat().st_size / 1024
                print_success(f"📄 Generated: {latest_word.name} ({size_kb:.1f} KB)")
        else:
            print_error("Plot extraction failed!")
    
    except subprocess.TimeoutExpired:
        print_error("Extraction timeout (exceeded time limit)")
    except Exception as e:
        print_error(f"Exception: {str(e)}")

def extract_plots_from_pkl():
    """Extract and generate plots from all PKL files."""
    print_header("🎨 EXTRACTING PLOTS FROM PKL FILES")
    
    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"
    
    extract_script = ARTICLE_DIR / "extract_plots_from_pkl.py"
    
    if not extract_script.exists():
        print_error(f"Script not found: {extract_script}")
        return
    
    print_info("Starting plot extraction from all PKL files...")
    print_info(f"Script: {extract_script}")
    print()
    
    try:
        result = subprocess.run(
            [python_exec, str(extract_script)],
            cwd=str(ARTICLE_DIR),
            text=True
        )
        
        if result.returncode == 0:
            print_success("✓ Plot extraction completed successfully!")
            
            # Check generated plots
            plots_dir = ARTICLE_DIR / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                if plot_files:
                    print_info(f"\n📁 Generated {len(plot_files)} plot files")
                    print_info(f"Location: {plots_dir}")
                    print_info("\nPlots generated:")
                    for plot_file in sorted(plot_files)[:10]:  # Show first 10
                        size_kb = plot_file.stat().st_size / 1024
                        print_info(f"  - {plot_file.name} ({size_kb:.1f} KB)")
                    if len(plot_files) > 10:
                        print_info(f"  ... and {len(plot_files) - 10} more")
        else:
            print_error("Plot extraction failed!")
    
    except subprocess.TimeoutExpired:
        print_error("Extraction timeout (exceeded time limit)")
    except Exception as e:
        print_error(f"Exception: {str(e)}")

# ============================================================================
# FULL TEST SUITE EXECUTION
# ============================================================================

def run_full_test_suite():
    """Run the full test suite using run_all_experiments.py script."""
    print_header("🚀 FULL TEST SUITE EXECUTION")
    
    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"
    
    test_runner_script = ARTICLE_DIR / "run_all_experiments.py"
    
    if not test_runner_script.exists():
        print_error(f"Test runner script not found: {test_runner_script}")
        print_info("Make sure 'run_all_experiments.py' exists in the project root")
        return
    
    print_info("Starting full test suite...")
    print_info(f"Script: {test_runner_script}")
    print_info("This will run all experiments with model selection")
    print_info("(This may take a considerable amount of time)\n")
    
    confirm = input("Continue with full test suite? (y/n): ").strip().lower()
    
    if confirm != 'y' and confirm != 'yes':
        print_warning("Full test suite cancelled")
        return
    
    print_success("Starting experiments...\n")
    
    try:
        # Run the test suite without timeout to allow full execution
        result = subprocess.run(
            [python_exec, str(test_runner_script)],
            cwd=str(ARTICLE_DIR),
            text=True
        )
        
        if result.returncode == 0:
            print_success("✓ Full test suite completed successfully!")
        else:
            print_error(f"Test suite failed with return code {result.returncode}")
        
        # Check for generated reports
        print_info("\nLooking for generated reports...")
        results_dir = ARTICLE_DIR / "run_results"
        
        if results_dir.exists():
            report_files = list(results_dir.glob("experiments_report_*.json"))
            log_files = list(results_dir.glob("run_all_*.log"))
            
            if report_files:
                print_success(f"Found {len(report_files)} report file(s):")
                for report_file in sorted(report_files)[-3:]:  # Show last 3
                    size_mb = report_file.stat().st_size / (1024**2)
                    print_info(f"  📊 {report_file.name} ({size_mb:.1f} MB)")
            
            if log_files:
                print_success(f"Found {len(log_files)} log file(s):")
                for log_file in sorted(log_files)[-3:]:  # Show last 3
                    size_kb = log_file.stat().st_size / 1024
                    print_info(f"  📝 {log_file.name} ({size_kb:.1f} KB)")
        else:
            print_info("No results directory found")
    
    except KeyboardInterrupt:
        print_warning("\nTest suite interrupted by user")
    except Exception as e:
        print_error(f"Exception during test suite: {str(e)}")

# ============================================================================
# INTERACTIVE TESTER
# ============================================================================

def run_interactive_test():
    """Launch the demo script instead of the interactive tester."""
    print_header("🧪 DEMO LAUNCHER")

    python_exec = "/home/akachat/tf_env/bin/python"
    if not Path(python_exec).exists():
        python_exec = "python3"

    demo_script = ARTICLE_DIR / "demo.py"

    if not demo_script.exists():
        print_error(f"Demo script not found: {demo_script}")
        print_info("Make sure 'demo.py' is present in the project root")
        return

    print_info("Launching demo script...")
    print_info(f"Script: {demo_script}")
    print()

    try:
        result = subprocess.run(
            [python_exec, str(demo_script)],
            cwd=str(ARTICLE_DIR),
            text=True
        )

        if result.returncode == 0:
            print_success("✓ Demo finished successfully!")
        else:
            print_error(f"Demo exited with return code {result.returncode}")

    except KeyboardInterrupt:
        print_warning("\nDemo interrupted by user")
    except Exception as e:
        print_error(f"Exception while launching demo: {str(e)}")

# ============================================================================
# INFO FUNCTIONS
# ============================================================================

def show_about():
    """Show about information."""
    print_header("ℹ️  ABOUT MELANOMA DETECTION PROJECT")
    
    print("""
This is a comprehensive machine learning and deep learning project 
for melanoma detection from dermatological images.

📊 PROJECT STRUCTURE:

1. Deep Learning (DL) Models
   - CNN: Custom convolutional neural network
   - VGG19: Transfer learning with VGG19
   - ENSEMBLE: ResNet50, EfficientNetV2, DenseNet169

2. Hybrid Models (DL + ML)
   - Feature extraction from DL models
   - Classification with traditional ML algorithms
   - ENSEMBLE voting strategies

3. ENSEMBLE Methods
   - Multiple DL architectures combined
   - ENSEMBLE with ML classifiers
   - Soft/Hard voting mechanisms

📁 MODELS INCLUDED:
   - ResNet50
   - VGG19
   - EfficientNetV2B0
   - DenseNet169

📚 ALGORITHMS:
   - Support Vector Machines (SVM)
   - Random Forests
   - K-Nearest Neighbors (KNN)
   - Decision Trees

🎯 OBJECTIVE:
   Binary classification: Malignant vs Benign lesions

📈 METRICS TRACKED:
   - Accuracy
   - Precision
   - Sensitivity (Recall)
   - Specificity
   - AUC-ROC
   - F1-Score
    """)

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Melanoma Detection - Main Project Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_project.py                    # Interactive mode
  python main_project.py --run all          # Run all experiments
  python main_project.py --run 1            # Run experiment 1
  python main_project.py --list             # List all experiments
  python main_project.py --analyze all      # Analyze all models
  python main_project.py --report summary   # Generate summary report
        """
    )
    
    parser.add_argument('--run', metavar='EXP', 
                       help='Run experiment (number 1-9 or "all")')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--analyze', metavar='EXP',
                       help='Analyze experiment')
    parser.add_argument('--report', choices=['summary', 'detailed'],
                       help='Generate report')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle command line arguments
    if args.list:
        show_all_experiments()
        sys.exit(0)
    elif args.run:
        if args.run == 'all':
            run_all_experiments()
        else:
            try:
                key = str(int(args.run))
                if key in EXPERIMENT_FILES:
                    exp_info = EXPERIMENT_FILES[key]
                    exp_path = ARTICLE_DIR / exp_info['file']
                    if exp_path.exists():
                        run_experiment(exp_path)
                    else:
                        print_error(f"File not found: {exp_path}")
                else:
                    print_error(f"Invalid experiment number: {args.run}")
            except ValueError:
                print_error("Invalid experiment specification")
        sys.exit(0)
    elif args.analyze:
        experiments = save_model.list_saved_experiments()
        if args.analyze == 'all':
            analyze_all_models(experiments)
        else:
            if args.analyze in experiments:
                analyze_single_model(args.analyze)
            else:
                print_error(f"Experiment not found: {args.analyze}")
        sys.exit(0)
    elif args.report:
        experiments = save_model.list_saved_experiments()
        if args.report == 'summary':
            generate_summary_report(experiments)
        else:
            generate_detailed_report(experiments)
        sys.exit(0)
    else:
        # Interactive mode
        try:
            menu_main()
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            sys.exit(0)

if __name__ == "__main__":
    main()
