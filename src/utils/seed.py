"""
seed.py
-------
Sets global random seeds for full reproducibility.
Call set_seed() at the very start of every script.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value (default 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TensorFlow seed — imported here to avoid slow startup everywhere
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
