"""
logger.py
---------
Sets up a clean, consistent logger for the entire project.
Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started")
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_to_file: bool = False, log_dir: str = "outputs/logs") -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name:        Logger name (use __name__ for module-level logging).
        log_to_file: If True, also writes logs to a timestamped file.
        log_dir:     Directory where log files are saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Format: [2025-04-01 10:30:00] INFO  src.data.loader — Loading dataset...
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-5s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler — INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler — DEBUG and above (optional)
    if log_to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"run_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
