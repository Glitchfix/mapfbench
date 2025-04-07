from typing import List, Dict, Any, Optional
import logging
import time

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger for the project.
    
    Args:
        name (str): Name of the logger
        log_level (int): Logging level (default: logging.INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

def timer(func):
    """
    Decorator to measure and log execution time of a function.
    
    Args:
        func (callable): Function to be timed
    
    Returns:
        callable: Wrapped function with timing functionality
    """
    def wrapper(*args, **kwargs):
        logger = setup_logger(func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that a configuration dictionary contains all required keys.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to validate
        required_keys (List[str]): List of keys that must be present
    
    Returns:
        bool: True if all required keys are present, False otherwise
    
    Raises:
        ValueError: If any required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    return True
