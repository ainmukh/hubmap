from abc import ABC, abstractmethod
from typing import Any, Dict
from torch.utils.data import Dataset


class BaseSegmentationDataset(ABC, Dataset):
    """
    The dataset should return the following:
    target: a dict containing the following fields
        boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format
        labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
        image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset
        area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric
        iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        (optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
        (optionally) keypoints (FloatTensor[N, K, 3]):
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError
