"""Provides a `seed_everything` method to ensure deterministic execution."""

# pylint: disable=import-outside-toplevel


def seed_everything(seed: int | None = 42):

    if seed is None:
        return

    import random

    random.seed(seed)

    import os

    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import tensorflow as tf

        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # TODO: test first (>TF 2.0)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"  # TODO: test first (>TF 2.0)
        tf.random.set_seed(seed)

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except (ImportError, ModuleNotFoundError):
        pass
