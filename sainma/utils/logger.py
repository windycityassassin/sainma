import logging
import sys
from typing import Optional

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with consistent formatting and settings.
    
    Args:
        name: Name of the logger (typically __name__)
        level: Optional logging level (defaults to INFO if not specified)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set default level if not already set
    if level is not None:
        logger.setLevel(level)
    elif not logger.hasHandlers():
        logger.setLevel(logging.INFO)
    
    # Only add handler if logger doesn't already have handlers
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger
