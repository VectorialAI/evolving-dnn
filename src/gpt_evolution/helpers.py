import os
import random
import logging

import numpy as np
import torch

def set_random_seeds(seed: int):
    """Set random seeds for all random number generators used in the project"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA operations
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True  # TODO revisit these for full runs
    # torch.backends.cudnn.benchmark = False

def deep_merge_dicts(default_dict, override_dict):
    """
    Recursively merge two dictionaries, where override_dict values
    take precedence over default_dict values at the leaf level.
    
    Args:
        default_dict: The base dictionary with default values
        override_dict: The dictionary with override values
        
    Returns:
        A new dictionary with deep-merged values
    """
    import copy
    
    # Start with a deep copy of the default dict
    result = copy.deepcopy(default_dict)
    
    def _merge_recursive(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                # Both are dicts, merge recursively
                _merge_recursive(base[key], value)
            else:
                # Override the value (could be a new key or a leaf value)
                base[key] = copy.deepcopy(value)
    
    _merge_recursive(result, override_dict)
    return result

def validate_flops_config(training_config):
    """Validate FLOPs budget configuration parameter."""
    flops_budget = training_config.get("flops_budget")
    
    # Validate FLOPs budget
    if flops_budget is not None:
        if not isinstance(flops_budget, (int, float)) or flops_budget <= 0:
            raise ValueError(f"flops_budget must be a positive number, got: {flops_budget}")
        logging.info(f"FLOPs budget set to: {flops_budget:,}")
    else:
        logging.info("No FLOPs budget set, using training_total_batches")

def configure_logger(experiment_path, logging_config):
    debug_log_file = os.path.join(experiment_path, "evolution_run_debug.log")
    info_log_file = os.path.join(experiment_path, "evolution_run.log")
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Determine file mode based on config
    file_mode = 'w' if logging_config.get("overwrite_logs", False) else 'a'
    
    # Handler for DEBUG and above (all messages) - goes to debug file
    debug_handler = logging.FileHandler(debug_log_file, mode=file_mode)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    
    # Handler for WARNING and above only - goes to warnings file
    info_handler = logging.FileHandler(info_log_file, mode=file_mode)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    
    # Console handler for INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(debug_handler)
    logger.addHandler(info_handler)
    logger.addHandler(console_handler)

    # Silence verbose loggers
    # for logger_name in ["urllib3", "datasets", "huggingface_hub", "fsspec"]:
    for logger_name in ["urllib3", "fsspec"]:  
        logging.getLogger(logger_name).setLevel(logging.WARNING)

if __name__ == "__main__":
    default_dict = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": {
            "e": 4,
            "f": 5
        }
    }
    override_dict = {"b": 4, "d": {"e": 6, "g": 7}}    
    merged_dict = deep_merge_dicts(default_dict, override_dict)
    print(merged_dict)
