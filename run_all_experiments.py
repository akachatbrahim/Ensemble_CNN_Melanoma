#!/usr/bin/env python3
"""
🔬 AUTOMATIC TEST RUNNER FOR ALL EXPERIMENTS - MELANOMA DETECTION PROJECT
==========================================================================

Cette script exécute automatiquement TOUS les cas d'expériences du projet
et génère un rapport complet sans modifier les fichiers du projet.

Cas d'exécution disponibles:
1. DL            - Deep Learning (1 modèle)
2. DL_ML         - Hybrid DL+ML (1 modèle DL + ML)
3. DEUX_DL_ML    - Multi-DL+ML Hybrid (2 modèles DL + ML)
4. DL_ENSEMBLE    - DL Ensemble (3 modèles DL)
5. DEUX_DL_ENSEMBLE - DL Ensemble (2 modèles DL)
6. DL_ML_ENSEMBLE - ML ENSEMBLE (1 modèle DL + ML ENSEMBLE)
7. DEUX_DL_ML_ENSEMBLE - Dual DL + ML ENSEMBLE (2 modèles DL + ML ENSEMBLE)
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import combinations
import logging
from tqdm import tqdm
from src.modules.Model import DL_MODELS

# ============================================================================
# CONFIGURATION
# ============================================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config.Config import path_project
PROJECT_ROOT = Path(path_project)
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LOGS_DIR = PROJECT_ROOT / "Logs"
RESULTS_DIR = PROJECT_ROOT / "run_results"
MODELS_PATH = PROJECT_ROOT / "Models"  # Where pkl files are saved

# Créer répertoires si nécessaire
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)

# Python executable
PYTHON_EXEC = "/home/akachat/tf_env/bin/python"
if not Path(PYTHON_EXEC).exists():
    PYTHON_EXEC = "python3"

# Verbose mode - Set to 1 to display all process outputs in real-time, 0 for silent mode
VERBOSE = 1

# ============================================================================
# DÉFINITION DES EXPÉRIENCES À TESTER
# ============================================================================

EXPERIMENTS = {
    "1": {
        "name": "DL - Deep Learning",
        "file": "experiments/single_dl_model.py",
        "description": "Single DL model with transfer learning",
        "type": "DL Only",
        "models_needed": ["DL"]
    },
    "2": {
        "name": "DL_ML - Hybrid DL+ML",
        "file": "experiments/hybrid_dl_ml.py",
        "description": "Single DL model + Machine Learning classifiers",
        "type": "Hybrid (DL+ML)",
        "models_needed": ["DL", "ML"]
    },
    "3": {
        "name": "DEUX_DL_ML - Multi-DL+ML Hybrid",
        "file": "experiments/dual_dl_ml.py",
        "description": "Two DL models combined + Machine Learning",
        "type": "Hybrid (Multi-DL+ML)",
        "models_needed": ["DL", "DL", "ML"]
    },
    "4": {
        "name": "DL_ENSEMBLE - DL Ensemble (3 Models)",
        "file": "experiments/triple_dl_ensemble.py",
        "description": "Ensemble of 3 DL models with averaging",
        "type": "DL Ensemble",
        "models_needed": ["DL", "DL", "DL"]
    },
    "5": {
        "name": "DEUX_DL_ENSEMBLE - DL Ensemble (2 Models)",
        "file": "experiments/dual_dl_ensemble.py",
        "description": "Ensemble of 2 DL models"
    },
    "6": {
        "name": "DL_ML_ENSEMBLE - ML ENSEMBLE",
        "file": "experiments/hybrid_dl_ml_ensemble.py",
        "description": "Single DL model + ML ENSEMBLE (SVM, KNN, DT, RF)",
        "type": "Hybrid Ensemble",
        "models_needed": ["DL", "ML"]
    },
    "7": {
        "name": "DEUX_DL_ML_ENSEMBLE - Dual DL + ML ENSEMBLE",
        "file": "experiments/dual_hybrid_ensemble.py",
        "description": "Two DL models + ML ENSEMBLE",
        "type": "Hybrid Ensemble",
        "models_needed": ["DL", "DL", "ML"]
    },
    "8": {
        "name": "TROIS_DL_ML - Triple DL + ML",
        "file": "experiments/triple_dl_ml.py",
        "description": "Three DL models combined + Machine Learning classifiers",
        "type": "Hybrid (Triple-DL+ML)",
        "models_needed": ["DL", "DL", "DL", "ML"]
    },
    "9": {
        "name": "TROIS_DL_ML_ENSEMBLE - Triple DL + ML ",
        "file": "experiments/triple_hybrid_ensemble.py",
        "description": "Three DL models + ML ensemble",
        "type": "Hybrid Ensemble",
        "models_needed": ["DL", "DL", "DL", "ML"]
    }
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file):
    """Configure logging to file and console."""
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def ask_pkl_handling_strategy():
    """
    Ask user how to handle existing pkl files.
    
    Returns:
        bool: True to skip if pkl exists, False to always execute
    """
    print_header("❓ STRATÉGIE D'EXÉCUTION - FICHIERS PKL EXISTANTS")
    print_info("Comment désirez-vous gérer les fichiers pkl existants?")
    print()
    print("  1. PASSER À L'ÉTAPE SUIVANTE - Skip experiments si le fichier pkl existe")
    print("     (Recommandé pour éviter les recalculs de modèles déjà existants)")
    print()
    print("  2. TOUJOURS EXÉCUTER - Exécute tous les experiments même si pkl existe")
    print("     (Utilisé pour régénérer/mettre à jour les fichiers pkl)")
    print()
    
    while True:
        choice = input("  Votre choix (1 ou 2) [default: 1]: ").strip()
        
        if choice == "" or choice == "1":
            print_success("✓ Mode: PASSER À L'ÉTAPE SUIVANTE (skip if pkl exists)")
            return True
        elif choice == "2":
            print_warning("⚠ Mode: TOUJOURS EXÉCUTER (execute all, overwrite pkl)")
            return False
        else:
            print_error("Choix invalide. Entrez '1' ou '2'")
    
    print()

def ask_model_parameters(selected_models):
    """
    Ask user for learning rate and weight decay for each selected model.
    These parameters will be FIXED for all experiments using these 3 models.
    
    Args:
        selected_models: List of 3 model IDs
        
    Returns:
        dict: Mapping from model_id to (learning_rate, weight_decay) tuple
    """
    print_header("⚙️ CONFIGURATION DES PARAMÈTRES - LR ET WEIGHT DECAY")
    print_info("Définissez les paramètres qui seront FIXES pour chaque modèle")
    print_info("Ces paramètres seront utilisés dans TOUS les experiments")
    print()
    
    model_params = {}
    
    for i, model_id in enumerate(selected_models, 1):
        model_name = DL_MODELS[model_id]['name']
        print_subheader(f"Paramètres du Modèle {i}/3: {model_name}")
        
        # Default learning rates
        default_lr = "3e-4"
        default_wd = "1e-5"
        
        while True:
            lr_input = input(f"  Learning Rate pour {model_name} [default: {default_lr}]: ").strip()
            if lr_input == "":
                lr_input = default_lr
            
            try:
                # Convert scientific notation to float
                lr_value = float(lr_input)
                break
            except ValueError:
                print_error(f"Format invalide. Entrez un nombre (ex: 1e-4 ou 0.0001)")
        
        while True:
            wd_input = input(f"  Weight Decay pour {model_name} [default: {default_wd}]: ").strip()
            if wd_input == "":
                wd_input = default_wd
            
            try:
                # Convert scientific notation to float
                wd_value = float(wd_input)
                break
            except ValueError:
                print_error(f"Format invalide. Entrez un nombre (ex: 1e-5 ou 0.00001)")
        
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
    
    config_file = PROJECT_ROOT / "current_model_parameters.json"
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        print_success(f"Paramètres sauvegardés: {config_file}")
        return config_file
    except Exception as e:
        print_error(f"Erreur sauvegarde paramètres: {str(e)}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_separator(char="=", length=80):
    """Print separator line."""
    print(char * length)

def print_header(title):
    """Print formatted header."""
    print("\n")
    print_separator("=", 80)
    print(f"  {title}")
    print_separator("=", 80)

def print_subheader(title):
    """Print formatted subheader."""
    print(f"\n  {title}")
    print_separator("-", 80)

def print_success(msg):
    """Print success message."""
    print(f"  ✓ {msg}")

def print_error(msg):
    """Print error message."""
    print(f"  ✗ {msg}")

def print_info(msg):
    """Print info message."""
    print(f"  ℹ {msg}")

def print_warning(msg):
    """Print warning message."""
    print(f"  ⚠ {msg}")

# ============================================================================
# EXPERIMENT EXECUTION
# ============================================================================

class ModelSelector:
    """Gestionnaire de sélection des modèles Deep Learning."""
    
    def __init__(self):
        self.selected_models = {}
    
    def display_models(self):
        """Afficher la liste des modèles disponibles."""
        print()
        for key in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
            model_info = DL_MODELS[key]
            print(f"  {key}. {model_info['name']}")
        print()
    
    def select_single_model(self, prompt):
        """
        Demander à l'utilisateur de sélectionner un modèle DL.
        
        Args:
            prompt: Message d'invite
            
        Returns:
            str: Clé du modèle sélectionné
        """
        print_info(prompt)
        self.display_models()
        
        while True:
            choice = input("  Votre choix [default: 1 (ResNet50)]: ").strip()
            
            if choice == "":
                choice = "1"
            
            if choice in DL_MODELS:
                selected = DL_MODELS[choice]['name']
                print_success(f"Sélectionné: {selected}")
                return choice
            else:
                print_error(f"Choix invalide. Entrez 1-{len(DL_MODELS)}")
    
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

class ExperimentRunner:
    """Runner pour les expériences."""
    
    def __init__(self, logger, selected_models=None, skip_if_pkl_exists=True):
        self.logger = logger
        self.results = defaultdict(dict)
        self.start_time = None
        self.total_time = 0
        self.selected_models = selected_models or []
        self.skip_if_pkl_exists = skip_if_pkl_exists  # If True, skip when pkl exists; if False, always execute
    
    def get_stdin_for_experiment(self, exp_id, model_combination):
        """
        Retourner les entrées stdin pour une expérience donnée avec une combinaison de modèles.
        
        Args:
            exp_id: ID de l'expérience
            model_combination: Liste des IDs modèles à utiliser pour cette exécution
            
        Returns:
            str: Entrées stdin pour l'expérience
        """
        if exp_id not in EXPERIMENTS:
            return ""
        
        if not model_combination:
            return ""
        
        # Pour les expériences qui nécessitent des modèles DL
        stdin_lines = []
        
        for model_id in model_combination:
            stdin_lines.append(model_id)
        
        stdin_lines.append("")  # Ligne finale
        return "\n".join(stdin_lines)
    
    def generate_experiment_combinations(self, exp_id, selected_models):
        """
        Générer toutes les combinaisons de modèles pour une expérience.
        
        Args:
            exp_id: ID de l'expérience
            selected_models: Liste de 3 modèles sélectionnés
            
        Returns:
            list: Liste de tuples (combination, label) où combination est une liste de modèle IDs
        """
        if exp_id not in EXPERIMENTS:
            return []
        
        exp_info = EXPERIMENTS[exp_id]
        models_needed = len([m for m in exp_info["models_needed"] if m == "DL"])
        
        # Pas de modèles DL nécessaires
        if models_needed == 0:
            return [([], "")]
        
        # 1 modèle DL nécessaire: exécuter avec chacun des 3 modèles
        if models_needed == 1:
            return [
                ([selected_models[0]], f"{DL_MODELS[selected_models[0]]['name']}"),
                ([selected_models[1]], f"{DL_MODELS[selected_models[1]]['name']}"),
                ([selected_models[2]], f"{DL_MODELS[selected_models[2]]['name']}")
            ]
        
        # 2 modèles DL nécessaires: toutes combinaisons de 2 parmi 3
        if models_needed == 2:
            combos = list(combinations(selected_models, 2))
            results = []
            for combo in combos:
                label = f"{DL_MODELS[combo[0]]['name']} + {DL_MODELS[combo[1]]['name']}"
                results.append((list(combo), label))
            return results
        
        # 3 modèles DL nécessaires: utiliser tous les 3
        if models_needed == 3:
            label = f"{DL_MODELS[selected_models[0]]['name']} + {DL_MODELS[selected_models[1]]['name']} + {DL_MODELS[selected_models[2]]['name']}"
            return [(selected_models, label)]
        
        return []
    
    def get_expected_pkl_filename(self, model_id=None, model_combination=None):
        """
        Generate expected pkl filename based on models used.
        
        Args:
            model_id: For single model experiments (exp 1)
            model_combination: For multi-model experiments
            
        Returns:
            str: Expected pkl filename without extension
        """
        if model_id:
            # Single model: just the model name
            return DL_MODELS[model_id]['name']
        elif model_combination and len(model_combination) > 0:
            # Multiple models: concatenate names with underscore
            model_names = [DL_MODELS[mid]['name'] for mid in model_combination]
            return "_".join(model_names)
        return None
    
    def check_pkl_file_exists(self, pkl_filename):
        """
        Check if pkl file already exists in Models directory.
        
        Args:
            pkl_filename: Expected pkl filename (without extension)
            
        Returns:
            Path: Path to pkl file if it exists, None otherwise
        """
        if not pkl_filename:
            return None
        
        pkl_path = MODELS_PATH / f"{pkl_filename}.pkl"
        if pkl_path.exists():
            return pkl_path
        return None
    
    def run_experiment(self, exp_id, exp_info, model_id=None, model_combination=None, combination_label=""):
        """
        Exécuter une seule expérience.
        
        Args:
            exp_id: ID de l'expérience
            exp_info: Dictionnaire d'info de l'expérience
            model_id: (Optionnel) Pour dl.py, le modèle spécifique à utiliser
            model_combination: (Optionnel) Liste des modèles à utiliser pour les autres exps
            combination_label: Label pour identifier la combinaison
            
        Returns:
            dict: Résultats de l'exécution
        """
        exp_name = exp_info["name"]
        exp_file = PROJECT_ROOT / exp_info["file"]
        
        # Pour dl.py, afficher le modèle utilisé
        if model_id and exp_id == "1":
            model_name = DL_MODELS[model_id]['name']
            print_subheader(f"Expérience {exp_id}: {exp_name} - {model_name}")
        elif combination_label:
            print_subheader(f"Expérience {exp_id}: {exp_name} - {combination_label}")
        else:
            print_subheader(f"Expérience {exp_id}: {exp_name}")
        print_info(f"Type: {exp_info['type']}")
        print_info(f"Description: {exp_info['description']}")
        print_info(f"Fichier: {exp_info['file']}")
        
        result = {
            "id": exp_id,
            "name": exp_name,
            "type": exp_info["type"],
            "file": exp_info["file"],
            "description": exp_info["description"],
            "status": "PENDING",
            "start_time": None,
            "end_time": None,
            "duration": None,
            "error": None,
            "command": None,
            "models_used": None,
            "combination_label": combination_label
        }
        
        # =====================================================================
        # CHECK IF PKL FILE ALREADY EXISTS - SKIP IF CONFIGURED TO DO SO
        # =====================================================================
        pkl_filename = self.get_expected_pkl_filename(model_id, model_combination)
        pkl_file = self.check_pkl_file_exists(pkl_filename)
        
        if pkl_file and self.skip_if_pkl_exists:
            print_warning(f"Fichier pkl déjà existant: {pkl_file.name}")
            print_info(f"Chemin: {pkl_file}")
            result["status"] = "SKIPPED"
            result["error"] = f"Pkl file already exists: {pkl_file.name}"
            end_time = datetime.now()
            result["end_time"] = end_time.isoformat()
            result["duration"] = 0  # No execution time since it was skipped
            self.logger.info(f"⊘ {exp_name} skipped (pkl exists): {pkl_file.name}")
            return result
        
        if pkl_file and not self.skip_if_pkl_exists:
            print_warning(f"Fichier pkl existant sera remplacé: {pkl_file.name}")
            print_info(f"Chemin: {pkl_file}")
        
        # Vérifier que le fichier existe
        if not exp_file.exists():
            print_error(f"Fichier non trouvé: {exp_file}")
            self.logger.error(f"Fichier non trouvé: {exp_file}")
            result["status"] = "FAILED"
            result["error"] = f"File not found: {exp_file}"
            return result
        
        # Préparer la commande
        command = [PYTHON_EXEC, "-u", str(exp_file)]
        result["command"] = " ".join(command)
        
        print_info(f"Commande: {result['command']}")
        
        # Préparer les entrées stdin
        if model_id and exp_id == "1":
            # Pour dl.py, utiliser le modèle spécifique
            stdin_input = model_id + "\n"
            result["models_used"] = [DL_MODELS[model_id]['name']]
            print_info(f"Modèle test: {DL_MODELS[model_id]['name']}")
        elif model_combination:
            # Pour les autres expériences avec une combinaison spécifique
            stdin_input = self.get_stdin_for_experiment(exp_id, model_combination)
            result["models_used"] = [DL_MODELS[mid]['name'] for mid in model_combination]
            print_info(f"Modèles: {', '.join(result['models_used'])}")
        else:
            stdin_input = ""
        
        # Exécuter l'expérience
        try:
            start_time = datetime.now()
            result["start_time"] = start_time.isoformat()
            
            print_info("▶ Exécution en cours...")
            
            # Préparer les variables d'environnement COMPLÈTES pour afficher les barres de progression
            env = os.environ.copy()
            # Configuration pour les barres de progression Keras/TensorFlow
            env['PYTHONUNBUFFERED'] = '1'  # Output unbuffered
            env['PYTHONDONTWRITEBYTECODE'] = '1'  # Pas de fichiers .pyc
            env['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Affiche INFO, WARNING, ERROR logs
            env['TF_ENABLE_ONEDNN_OPTS'] = '0'
            env['KERAS_PROGRESS'] = '1'  # Enable Keras progress bars
            env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # GPU memory management
            env['COLUMNS'] = '200'  # Permet les barres de progression plus larges
            
            # Afficher les outputs en temps réel (VERBOSE MODE)
            stdout_lines = []
            
            # Configurer le subprocess selon le mode VERBOSE
            if VERBOSE == 1:
                # Mode VERBOSE: Ne pas rediriger stdout/stderr pour que les barres de progression s'affichent directement
                # Les barres ANSI de model.fit s'affichent automatiquement à la console
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.PIPE,
                    env=env,
                    bufsize=0,  # Unbuffered
                    universal_newlines=False
                )
            else:
                # Mode silencieux: Rediriger stdout/stderr pour une capture silencieuse
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env
                )
            
            try:
                # Envoyer stdin et fermer pour éviter les blocages
                if stdin_input:
                    if VERBOSE == 1:
                        process.stdin.write(stdin_input.encode() if isinstance(stdin_input, str) else stdin_input)
                    else:
                        process.stdin.write(stdin_input)
                    process.stdin.close()
                
                # Lire et afficher les outputs en temps réel (mode silencieux)
                if VERBOSE == 0:
                    for line in process.stdout:
                        if line:
                            line_clean = line.rstrip('\n')
                            stdout_lines.append(line_clean)
                
                # Attendre la fin du process
                process.wait(timeout=3600)
                
            except subprocess.TimeoutExpired:
                process.kill()
                result["status"] = "TIMEOUT"
                result["error"] = "Execution timeout after 1 hour"
                self.logger.error(f"Timeout: {exp_name}")
                print_error(f"Timeout après 1 heure")
                end_time = datetime.now()
                result["end_time"] = end_time.isoformat()
                result["duration"] = (end_time - start_time).total_seconds()
                return result
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result["end_time"] = end_time.isoformat()
            result["duration"] = duration
            
            if process.returncode == 0:
                result["status"] = "SUCCESS"
                print_success(f"Expérience completed en {duration:.2f}s")
                self.logger.info(f"✓ {exp_name} completed in {duration:.2f}s")
            else:
                result["status"] = "FAILED"
                result["error"] = f"Process returned code {process.returncode}"
                print_error(f"Échec avec code {process.returncode}")
                self.logger.error(f"✗ {exp_name} failed: {result['error']}")
                
                # Afficher les dernières lignes d'erreur
                if stdout_lines:
                    print_warning("Dernier messages d'erreur:")
                    for line in stdout_lines[-5:]:
                        if line.strip():
                            print_info(f"  {line}")
            
            # Sauvegarder les outputs
            result["stdout_lines"] = len(stdout_lines)
            result["stderr_lines"] = 0
            
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            print_error(f"Exception: {str(e)}")
            self.logger.exception(f"Exception in {exp_name}: {str(e)}")
            end_time = datetime.now()
            result["end_time"] = end_time.isoformat()
            result["duration"] = (end_time - start_time).total_seconds()
        
        return result
    
    def run_all(self, exp_ids=None):
        """
        Exécuter toutes les expériences ou sélectionnées.

        Pour dl.py (exp_id="1"): exécute avec TOUS les modèles de DL_MODELS
        Pour les autres expériences: exécute avec TOUTES les combinaisons de modèles sélectionnés
        
        Args:
            exp_ids: List d'IDs à exécuter. Si None, exécute tous.
        """
        if exp_ids is None:
            exp_ids = list(EXPERIMENTS.keys())
        
        # Compter le nombre total d'exécutions
        total_runs = 0
        for exp_id in exp_ids:
            if exp_id == "1":
                total_runs += len(DL_MODELS)  # dl.py exécuté pour chaque modèle
            else:
                # Générer les combinaisons pour compter
                combinations = self.generate_experiment_combinations(exp_id, self.selected_models)
                total_runs += len(combinations)
        
        print_header(f"🔬 EXÉCUTION AUTOMATIQUE - {len(exp_ids)} EXPÉRIENCES")
        print_info(f"Total d'exécutions: {total_runs}")
        print_info(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_info(f"Python: {PYTHON_EXEC}")
        
        self.start_time = datetime.now()
        all_results = []
        execution_count = 0
        
        for exp_id in exp_ids:
            if exp_id not in EXPERIMENTS:
                print_warning(f"Expérience {exp_id} inconnue, ignorée")
                continue
            
            exp_info = EXPERIMENTS[exp_id]
            
            # Si c'est dl.py, exécuter avec chaque modèle
            if exp_id == "1":
                print_header(f"🔬 EXPÉRIENCE {exp_id}: {exp_info['name']} - TOUS LES MODÈLES")
                print_info(f"Exécution de {len(DL_MODELS)} tests avec tous les modèles disponibles")
                print()
                
                model_ids = sorted(DL_MODELS.keys(), key=lambda x: int(x))
                for model_id in tqdm(model_ids, desc="Modèles", unit="model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
                    execution_count += 1
                    print(f"\n┌─ Test {execution_count}/{total_runs} - {DL_MODELS[model_id]['name']} ─┐")
                    result = self.run_experiment(exp_id, exp_info, model_id=model_id)
                    all_results.append(result)
                    
                    # Utiliser une clé unique pour chaque modèle
                    result_key = f"{exp_id}_{model_id}"
                    self.results[result_key] = result
                    
                    # Petit délai entre modèles
                    time.sleep(1)
            else:
                # Pour les autres expériences, générer et exécuter avec TOUTES les combinaisons
                combinations = self.generate_experiment_combinations(exp_id, self.selected_models)
                
                if len(combinations) > 1:
                    print_header(f"🔬 EXPÉRIENCE {exp_id}: {exp_info['name']} - {len(combinations)} COMBINAISONS")
                else:
                    print_header(f"🔬 EXPÉRIENCE {exp_id}: {exp_info['name']}")
                
                combinations_enum = list(enumerate(combinations, 1))
                for combo_idx, (model_combination, combination_label) in tqdm(combinations_enum, desc=f"Expérience {exp_id}", unit="combo", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
                    execution_count += 1
                    print(f"\n┌─ Exécution {execution_count}/{total_runs} (Combinaison {combo_idx}/{len(combinations)}) ─┐")
                    
                    result = self.run_experiment(
                        exp_id, 
                        exp_info, 
                        model_combination=model_combination,
                        combination_label=combination_label
                    )
                    all_results.append(result)
                    
                    # Utiliser une clé unique pour chaque combinaison
                    result_key = f"{exp_id}_{combo_idx}"
                    self.results[result_key] = result
                    
                    # Petit délai entre combinaisons
                    time.sleep(1)
            
            # Petit délai entre expériences
            if exp_id != exp_ids[-1]:
                print_info("Attente avant prochaine expérience...")
                time.sleep(2)
        
        return all_results
    
    def generate_summary(self):
        """Générer un résumé des résultats."""
        print_header("📊 RÉSUMÉ DES RÉSULTATS")
        
        total_exps = len(self.results)
        successful = sum(1 for r in self.results.values() if r["status"] == "SUCCESS")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAILED")
        errors = sum(1 for r in self.results.values() if r["status"] == "ERROR")
        timeouts = sum(1 for r in self.results.values() if r["status"] == "TIMEOUT")
        skipped = sum(1 for r in self.results.values() if r["status"] == "SKIPPED")
        
        total_duration = sum(r["duration"] or 0 for r in self.results.values())
        
        print(f"\n  Total exécutions: {total_exps}")
        print(f"  ✓ Succès: {successful}")
        print(f"  ✗ Échouées: {failed}")
        print(f"  ⚠ Erreurs: {errors}")
        print(f"  ⏱ Timeouts: {timeouts}")
        print(f"  ⊘ Ignorées (pkl existe): {skipped}")
        print(f"  Durée totale: {total_duration:.2f}s ({total_duration/60:.1f}min)")
        
        print_subheader("Détails par exécution")
        
        # Tri personnalisé pour gérer les clés comme "1_1", "2_1", "2_2", etc.
        def sort_key(x):
            parts = x.split('_')
            if len(parts) == 2 and parts[1].isdigit():
                return (int(parts[0]), int(parts[1]))
            else:
                return (int(x), 0)
        
        for result_key in sorted(self.results.keys(), key=sort_key):
            result = self.results[result_key]
            status_icon = {
                "SUCCESS": "✓",
                "FAILED": "✗",
                "ERROR": "⚠",
                "TIMEOUT": "⏱",
                "SKIPPED": "⊘",
                "PENDING": "◯"
            }.get(result["status"], "?")
            
            duration_str = f"{result['duration']:.2f}s" if result["duration"] else "N/A"
            
            # Afficher le label de la combinaison si présent
            if result.get("combination_label"):
                display_name = f"{result['name']} ({result['combination_label']})"
            else:
                display_name = result['name']
            
            print(f"  {status_icon} [{result['id']}] {display_name:<60} [{result['status']:<8}] {duration_str}")
            
            if result["models_used"]:
                print(f"      Modèles: {', '.join(result['models_used'])}")
            
            if result["error"]:
                print(f"      └─ Erreur: {result['error']}")
    
    def analyze_existing_pkl_files(self, exp_ids=None):
        """
        Analyze which experiments will be skipped based on existing pkl files.
        
        Args:
            exp_ids: List of experiment IDs to analyze
            
        Returns:
            dict: Analysis results with skipped and to_run counts
        """
        if exp_ids is None:
            exp_ids = list(EXPERIMENTS.keys())
        
        analysis = {
            "total_planned": 0,
            "will_be_skipped": 0,
            "will_be_executed": 0,
            "skipped_experiments": [],
            "to_run_experiments": []
        }
        
        for exp_id in exp_ids:
            exp_info = EXPERIMENTS[exp_id]
            
            # For experiment 1 (dl.py), check all models
            if exp_id == "1":
                for model_id in sorted(DL_MODELS.keys(), key=lambda x: int(x)):
                    pkl_filename = self.get_expected_pkl_filename(model_id=model_id)
                    pkl_file = self.check_pkl_file_exists(pkl_filename)
                    analysis["total_planned"] += 1
                    
                    model_name = DL_MODELS[model_id]['name']
                    exp_label = f"[{exp_id}] {exp_info['name']} - {model_name}"
                    
                    if pkl_file and self.skip_if_pkl_exists:
                        analysis["will_be_skipped"] += 1
                        analysis["skipped_experiments"].append({
                            "exp_id": exp_id,
                            "label": exp_label,
                            "pkl_file": pkl_file.name
                        })
                    else:
                        analysis["will_be_executed"] += 1
                        analysis["to_run_experiments"].append({
                            "exp_id": exp_id,
                            "label": exp_label
                        })
            else:
                # For other experiments, check all combinations
                combinations_list = self.generate_experiment_combinations(exp_id, self.selected_models)
                
                for combo_idx, (model_combination, combination_label) in enumerate(combinations_list, 1):
                    pkl_filename = self.get_expected_pkl_filename(model_combination=model_combination)
                    pkl_file = self.check_pkl_file_exists(pkl_filename)
                    analysis["total_planned"] += 1
                    
                    exp_label = f"[{exp_id}] {exp_info['name']} - {combination_label}"
                    
                    if pkl_file and self.skip_if_pkl_exists:
                        analysis["will_be_skipped"] += 1
                        analysis["skipped_experiments"].append({
                            "exp_id": exp_id,
                            "label": exp_label,
                            "pkl_file": pkl_file.name
                        })
                    else:
                        analysis["will_be_executed"] += 1
                        analysis["to_run_experiments"].append({
                            "exp_id": exp_id,
                            "label": exp_label
                        })
        
        return analysis
    
    def display_pkl_analysis(self, analysis):
        """
        Display pre-execution analysis of pkl files.
        
        Args:
            analysis: Analysis dict from analyze_existing_pkl_files()
        """
        print_header("🔍 ANALYSE DES FICHIERS PKL EXISTANTS")
        
        print(f"\n  📊 Résumé")
        print(f"  ──────────────────────────────────────")
        print(f"  Total planifié:        {analysis['total_planned']} tests")
        print(f"  ✓ À exécuter:          {analysis['will_be_executed']} tests")
        print(f"  ⊘ Seront ignorés:      {analysis['will_be_skipped']} tests")
        
        if analysis['will_be_skipped'] > 0:
            percentage = (analysis['will_be_skipped'] / analysis['total_planned']) * 100
            print(f"  ({percentage:.1f}% des tests seront ignorés)")
        
        # Show skipped tests
        if analysis['skipped_experiments']:
            print_subheader("Fichiers PKL existants (seront ignorés)")
            for item in analysis['skipped_experiments']:
                print(f"  ⊘ {item['label']}")
                print(f"     → {item['pkl_file']}")
        
        # Show tests to run
        if analysis['to_run_experiments']:
            print_subheader("Tests à exécuter (pkl inexistant)")
            for item in analysis['to_run_experiments']:
                print(f"  ↻ {item['label']}")
    
    def save_results(self, output_file=None):
        """Sauvegarder les résultats en JSON."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = RESULTS_DIR / f"experiments_report_{timestamp}.json"
        
        # Préparer les données
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(PROJECT_ROOT),
            "python_executable": PYTHON_EXEC,
            "models_directory": str(MODELS_PATH),
            "experiments": dict(self.results),
            "summary": {
                "total": len(self.results),
                "successful": sum(1 for r in self.results.values() if r["status"] == "SUCCESS"),
                "failed": sum(1 for r in self.results.values() if r["status"] == "FAILED"),
                "errors": sum(1 for r in self.results.values() if r["status"] == "ERROR"),
                "timeouts": sum(1 for r in self.results.values() if r["status"] == "TIMEOUT"),
                "skipped": sum(1 for r in self.results.values() if r["status"] == "SKIPPED"),
                "total_duration_seconds": sum(r["duration"] or 0 for r in self.results.values())
            }
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print_success(f"Rapport sauvegardé: {output_file}")
            print_info(f"Lien: file://{output_file}")
            return output_file
        except Exception as e:
            print_error(f"Erreur sauvegarde rapport: {str(e)}")
            return None

# ============================================================================
# CONFIGURATION ANALYSIS
# ============================================================================

class ConfigAnalyzer:
    """Analyser la configuration du projet."""
    
    @staticmethod
    def check_data_directories():
        """Vérifier que les répertoires de données existent."""
        train_dir = PROJECT_ROOT / "data/train"
        test_dir = PROJECT_ROOT / "data/test"
        
        print_subheader("📁 Vérification des répertoires de données")
        
        if train_dir.exists():
            print_success(f"Train dir trouvé: {train_dir}")
            train_count = len(list(train_dir.rglob("*")))
            print_info(f"  Éléments: {train_count}")
        else:
            print_error(f"Train dir manquant: {train_dir}")
        
        if test_dir.exists():
            print_success(f"Test dir trouvé: {test_dir}")
            test_count = len(list(test_dir.rglob("*")))
            print_info(f"  Éléments: {test_count}")
        else:
            print_error(f"Test dir manquant: {test_dir}")
    
    @staticmethod
    def check_source_files():
        """Vérifier les fichiers source."""
        print_subheader("📄 Vérification des fichiers sources")
        
        required_modules = [
            "src/modules/Model.py",
            "src/modules/Dataset.py",
            "src/modules/Evaluate.py",
            "src/modules/save_model.py",
            "src/config/Config.py"
        ]
        
        for module in required_modules:
            module_path = PROJECT_ROOT / module
            if module_path.exists():
                print_success(f"Trouvé: {module}")
            else:
                print_error(f"Manquant: {module}")
    
    @staticmethod
    def check_experiment_files():
        """Vérifier les fichiers d'expériences."""
        print_subheader("🧪 Vérification des fichiers d'expériences")
        
        for exp_id in sorted(EXPERIMENTS.keys(), key=lambda x: int(x)):
            exp_info = EXPERIMENTS[exp_id]
            exp_file = PROJECT_ROOT / exp_info["file"]
            
            if exp_file.exists():
                size = exp_file.stat().st_size
                print_success(f"[{exp_id}] {exp_info['name']:<40} ({size} bytes)")
            else:
                print_error(f"[{exp_id}] {exp_info['name']:<40} MANQUANT")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Setup logging
    log_file = RESULTS_DIR / f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(str(log_file))
    
    logger.info("="*80)
    logger.info("AUTOMATIC TEST RUNNER STARTED")
    logger.info("="*80)
    
    print_header("🔬 AUTOMATIC TEST RUNNER - MELANOMA DETECTION PROJECT")
    
    # Analyze configuration
    print_subheader("📋 ANALYSE DE LA CONFIGURATION DU PROJET")
    ConfigAnalyzer.check_data_directories()
    ConfigAnalyzer.check_source_files()
    ConfigAnalyzer.check_experiment_files()
    
    # Show available experiments
    print_header("🧪 CAS D'EXÉCUTION DISPONIBLES")
    for exp_id in sorted(EXPERIMENTS.keys(), key=lambda x: int(x)):
        exp_info = EXPERIMENTS[exp_id]
        print(f"\n  [{exp_id}] {exp_info['name']}")
        print(f"      Type: {exp_info['type']}")
        print(f"      Description: {exp_info['description']}")
        print(f"      Fichier: {exp_info['file']}")
    
    # ========================================================================
    # CHOIX: Stratégie d'exécution pour les fichiers pkl existants
    # ========================================================================
    skip_if_pkl_exists = ask_pkl_handling_strategy()
    
    # ========================================================================
    # CHOIX: Exécuter dl.py avec TOUS les modèles ?
    # ========================================================================
    print_header("❓ CHOIX: EXÉCUTER DL.PY AVEC TOUS LES MODÈLES")
    print_info("Voulez-vous exécuter dl.py avec les 11 modèles disponibles?")
    print_info("(Cette étape teste chaque modèle individuellement)")
    print()
    
    while True:
        choice = input("  Exécuter dl.py? (oui/non) [default: oui]: ").strip().lower()
        
        if choice == "" or choice == "o" or choice == "oui" or choice == "yes" or choice == "y":
            execute_dl = True
            print_success("✓ dl.py sera exécuté avec tous les modèles")
            break
        elif choice == "n" or choice == "non" or choice == "no":
            execute_dl = False
            print_warning("⊘ dl.py sera ignoré")
            break
        else:
            print_error("Choix invalide. Entrez 'oui' ou 'non'")
    
    print()
    
    # ========================================================================
    # ÉTAPE 1: Exécuter dl.py avec TOUS les modèles (si choisi)
    # ========================================================================
    if execute_dl:
        print_header("🚀 ÉTAPE 1: EXÉCUTION DE DL.PY AVEC TOUS LES MODÈLES")
        print_info("Exécution de dl.py avec les 11 modèles disponibles...")
        
        runner = ExperimentRunner(logger, skip_if_pkl_exists=skip_if_pkl_exists)
        exp_ids_dl = ["1"]  # Juste dl.py
        
        # Afficher l'analyse des fichiers pkl existants pour cette étape
        analysis_step1 = runner.analyze_existing_pkl_files(exp_ids_dl)
        runner.display_pkl_analysis(analysis_step1)
        
        print()
        runner.run_all(exp_ids_dl)
    else:
        print_header("⊘ ÉTAPE 1: IGNORÉE")
        print_info("dl.py ne sera pas exécuté")
        runner = ExperimentRunner(logger, skip_if_pkl_exists=skip_if_pkl_exists)  # Créer un runner vide
    
    # ========================================================================
    # ÉTAPE 2: Sélectionner 3 modèles pour les autres expériences
    # ========================================================================
    print_header("🤖 ÉTAPE 2: SÉLECTION DE 3 MODÈLES")
    print_info("Sélectionnez 3 modèles pour les autres expériences (2-7)")
    
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
    # ÉTAPE 3: Exécuter les autres expériences avec les 3 modèles sélectionnés
    # ========================================================================
    print_header("🚀 ÉTAPE 3: EXÉCUTION DES AUTRES EXPÉRIENCES")
    print_info("Exécution des expériences 2-9 avec les 3 modèles sélectionnés...")
    
    # Créer un nouveau runner avec les modèles sélectionnés, en gardant les résultats précédents
    runner2 = ExperimentRunner(logger, selected_models=selected_models, skip_if_pkl_exists=skip_if_pkl_exists)
    
    # Exécuter les expériences 2-9
    exp_ids_others = list(EXPERIMENTS.keys())[1:]  # Toutes sauf dl.py
    
    # Afficher l'analyse des fichiers pkl existants pour cette étape
    analysis_step3 = runner2.analyze_existing_pkl_files(exp_ids_others)
    runner2.display_pkl_analysis(analysis_step3)
    
    print_info("Démarrage des expériences...")
    time.sleep(1)
    
    runner2.run_all(exp_ids_others)
    
    # Fusionner les résultats des deux runners
    runner.results.update(runner2.results)
    
    # Generate summary
    runner.generate_summary()
    
    # Save results
    print_header("💾 SAUVEGARDE DES RÉSULTATS")
    report_file = runner.save_results()
    
    # Final summary
    print_header("✅ EXÉCUTION TERMINÉE")
    print_info(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_info(f"Log: {log_file}")
    if report_file:
        print_info(f"Rapport: {report_file}")
    
    logger.info("="*80)
    logger.info("AUTOMATIC TEST RUNNER COMPLETED")
    logger.info("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Exécution interrompue par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Erreur critique: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
