# 从各个模块导入所需的类和函数
from .inference_dataset import *
from .inference_dataset_config import *
from .sequence_dataset import *
from .utils import *

__all__ = ["inference_dataset", "inference_dataset_config", "sequence_dataset", "utils"]

import torch
from torch.utils.data import DataLoader


def get_data_loader(
    dataset: torch.utils.data.Dataset,
    pin_memory: bool = False,
) -> DataLoader:
    loader = DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        pin_memory=pin_memory,
        collate_fn=lambda x: x,
    )
    return loader
