from typing import Dict, List, Tuple
import torch
from hubmap_vasculature.training.datasets.utils import BatchFields


def collate_fn(dataset_items: List[Dict]) -> Tuple[torch.Tensor, List]:
    """
    Collate and pad fields in dataset items
    """
    images = []
    for item in dataset_items:
        images.append(item[BatchFields.IMAGE].unsqueeze(0))
        del item[BatchFields.IMAGE]

    return torch.cat(images, dim=0), dataset_items
