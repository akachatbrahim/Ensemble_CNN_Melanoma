"""
Utility to load model parameters from current_model_parameters.json.
Ensures consistent use of fixed lr and weight_decay across all experiments.
"""
import os
import json
from pathlib import Path
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
from src.config.Config import path_project


def load_model_parameters(project_root=path_project):
    """
    Load model parameters from current_model_parameters.json.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        dict: Mapping of model_id to {"model_name", "learning_rate", "weight_decay"}
              Returns None if file doesn't exist
    """
    config_file = Path(project_root) / "current_model_parameters.json"
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        return config_data.get("parameters", {})
    except Exception as e:
        print(f"⚠ Error loading model parameters: {e}")
        return None

def get_model_parameter(model_id, project_root="/home/akachat/tf_env/Ensemble_CNN_Melanoma"):
    """
    Get learning rate and weight decay for a specific model.
    
    Args:
        model_id: ID of the model (as string, e.g., "1", "2", "3")
        project_root: Root directory of the project
        
    Returns:
        tuple: (learning_rate, weight_decay) if found, None otherwise
    """
    params = load_model_parameters(project_root)
    
    if params and str(model_id) in params:
        model_params = params[str(model_id)]
        return model_params["learning_rate"], model_params["weight_decay"]
    
    return None

def get_model_parameters_for_ids(model_ids, project_root="/home/akachat/tf_env/Ensemble_CNN_Melanoma"):
    """
    Get parameters for multiple models in order.
    
    Args:
        model_ids: List of model IDs (strings)
        project_root: Root directory of the project
        
    Returns:
        list: List of (learning_rate, weight_decay) tuples in same order as model_ids
    """
    params = load_model_parameters(project_root)
    result = []
    
    if params:
        for model_id in model_ids:
            model_id_str = str(model_id)
            if model_id_str in params:
                model_data = params[model_id_str]
                result.append((model_data["learning_rate"], model_data["weight_decay"]))
            else:
                result.append(None)
    
    return result
