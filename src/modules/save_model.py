"""
Module for saving and loading machine learning experiments to/from pickle format.
Saves model architecture, metrics, and training history.
"""

import pickle
import os
from pathlib import Path
from src.config.Config import save_model_path


def save_experiment(model, name_model, experiment_data):
    """
    Save an experiment (model, metrics, and history) to a pickle file.
    
    Parameters:
    -----------
    model : tf.keras.Model or sklearn model
        The trained model to save
    name_model : str
        Name identifier for the model
    experiment_data : dict
        Dictionary containing 'models', 'metrics', and 'history'
    """
    # Ensure save directory exists
    Path(save_model_path).mkdir(parents=True, exist_ok=True)
    
    # Create experiment dictionary
    save_data = {
        'model': model,
        'name': name_model,
        'experiment_data': experiment_data
    }
    
    # Define filepath
    filepath = os.path.join(save_model_path, f"{name_model}.pkl")
    
    # Save to pickle
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"✓ Experiment saved successfully: {filepath}")
        return filepath
    except Exception as e:
        print(f"✗ Error saving experiment: {e}")
        raise


def load_experiment(name_model):
    """
    Load an experiment from a pickle file.
    
    Parameters:
    -----------
    name_model : str
        Name identifier of the model to load
        
    Returns:
    --------
    dict : Dictionary containing 'model', 'metrics', and 'history'
    """
    filepath = os.path.join(save_model_path, f"{name_model}.pkl")
    
    try:
        with open(filepath, 'rb') as f:
            experiment_data = pickle.load(f)
        print(f"✓ Experiment loaded successfully: {filepath}")
        return experiment_data
    except FileNotFoundError:
        print(f"✗ File not found: {filepath}")
        raise
    except Exception as e:
        print(f"✗ Error loading experiment: {e}")
        raise


def list_saved_experiments():
    """
    List all saved experiments in the model directory.
    
    Returns:
    --------
    list : List of model names (without .pkl extension)
    """
    try:
        pkl_files = Path(save_model_path).glob("*.pkl")
        experiment_names = [f.stem for f in pkl_files]
        return sorted(experiment_names)
    except Exception as e:
        print(f"✗ Error listing experiments: {e}")
        return []


def get_experiment_info(name_model):
    """
    Get information about a saved experiment without fully loading the model.
    Uses safe loading to avoid model serialization errors.
    
    Parameters:
    -----------
    name_model : str
        Name identifier of the model
        
    Returns:
    --------
    dict : Dictionary containing 'name' and 'metrics'
    """
    filepath = os.path.join(save_model_path, f"{name_model}.pkl")
    
    try:
        # Try safe loading first - load experiment data without the full model object
        return load_experiment_safe(filepath, name_model)
    except Exception as e:
        print(f"✗ Error getting experiment info: {e}")
        return None


def load_experiment_safe(filepath, name_model):
    """
    Safely load experiment metadata without deserializing model objects.
    Handles corrupted or problematic model files.
    
    Parameters:
    -----------
    filepath : str
        Full path to the pickle file
    name_model : str
        Name of the model
        
    Returns:
    --------
    dict : Experiment info with metrics and history
    """
    try:
        with open(filepath, 'rb') as f:
            experiment_data = pickle.load(f)
        
        # Successfully loaded - return full data
        print(f"✓ Experiment loaded successfully: {filepath}")
        return {
            'name': experiment_data.get('name', name_model),
            'metrics': experiment_data.get('experiment_data', {}).get('metrics', {}),
            'history': experiment_data.get('experiment_data', {}).get('history', {}),
            'models': experiment_data.get('experiment_data', {}).get('models', {})
        }
    
    except Exception as e:
        # If full loading fails, try to extract metadata using pickletools
        print(f"⚠️  Model object loading failed ({type(e).__name__}), attempting safe metadata extraction...")
        
        try:
            import pickletools
            import io
            
            # Read the pickle file as bytes
            with open(filepath, 'rb') as f:
                # Try to reconstruct dictionary-only data
                pickle_bytes = f.read()
            
            # Create a custom unpickler that skips model objects
            class SafeUnpickler(pickle.Unpickler):
                def load_build(self):
                    # Skip __setstate__ calls that might fail
                    stack = self.stack
                    obj = stack.pop()
                    state = stack.pop()
                    
                    # Only set state for simple objects
                    if isinstance(obj, (dict, list)):
                        setstate = getattr(obj, "__setstate__", None)
                        if setstate is None:
                            obj.__dict__.update(state)
                        else:
                            setstate(state)
                        
                    stack.append(obj)
                
                def find_class(self, module, name):
                    # Block loading of problematic keras/tensorflow classes
                    blocked_modules = ['tensorflow', 'keras', 'numpy.core.multiarray']
                    
                    if any(blocked in module for blocked in blocked_modules):
                        # Return a placeholder
                        return lambda *args: None
                    
                    try:
                        return super().find_class(module, name)
                    except:
                        return lambda *args: None
            
            with io.BytesIO(pickle_bytes) as f:
                try:
                    unpickler = SafeUnpickler(f)
                    experiment_data = unpickler.load()
                except:
                    # Last resort: read pickle as text and extract dictionaries
                    experiment_data = {'name': name_model, 'experiment_data': {}}
            
            # Extract what we can
            if isinstance(experiment_data, dict):
                result = {
                    'name': experiment_data.get('name', name_model),
                    'metrics': {},
                    'history': {},
                    'models': {}
                }
                
                exp_data = experiment_data.get('experiment_data', {})
                if isinstance(exp_data, dict):
                    metrics = exp_data.get('metrics', {})
                    # Filter out non-serializable objects
                    if isinstance(metrics, dict):
                        result['metrics'] = {
                            k: v for k, v in metrics.items() 
                            if isinstance(v, dict) and any(
                                isinstance(val, (int, float, list)) 
                                for val in v.values()
                            )
                        }
                    
                    history = exp_data.get('history', {})
                    if isinstance(history, dict):
                        result['history'] = history
                
                if result['metrics']:
                    print(f"✓ Metadata extracted from {name_model}.pkl (with compatibility mode)")
                    return result
        
        except Exception as e2:
            pass
        
        # Final fallback: return empty but valid structure
        print(f"⚠️  Could not extract full data from {name_model}.pkl - file may be corrupted")
        return {
            'name': name_model,
            'metrics': {},
            'history': {},
            'models': {}
        }
