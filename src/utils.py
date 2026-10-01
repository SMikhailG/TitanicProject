import random

import numpy as np
import torch


def set_seed(seed):
    """
    Фиксирует генераторы случайных чисел
    для воспроизводимости экспериментов.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Возвращает доступное устройство для PyTorch.
    """
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )