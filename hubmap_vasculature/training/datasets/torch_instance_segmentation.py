from typing import Any, Dict, Tuple
import cv2
import loguru
import torch
import numpy as np

from .base_hubmap_dataset import BaseHuBMAPDataset
from .base_segmentation_dataset import BaseSegmentationDataset
from .utils import BatchFields


class TorchInstanceSegmentationDataset(BaseHuBMAPDataset, BaseSegmentationDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {BatchFields.IMAGE_ID: self.ids[index]}

        item[BatchFields.IMAGE] = self._get_image(item[BatchFields.IMAGE_ID])
        item[BatchFields.MASKS] = self._get_masks(
            item[BatchFields.IMAGE_ID], item[BatchFields.IMAGE]
        )
        item[BatchFields.BOXES], item[BatchFields.AREA] = self._get_boxes(item[BatchFields.MASKS])

        n_objects = item[BatchFields.BOXES].size(0)
        item[BatchFields.LABELS] = torch.ones((n_objects,), dtype=torch.int64)
        item[BatchFields.IS_CROWD] = torch.zeros((n_objects,))

        return item

    def _get_image(self, image_id: str) -> torch.Tensor:
        image = cv2.imread(
            str(self.data_path / self.part / f"{image_id}.tif"),
            cv2.IMREAD_UNCHANGED
        ).transpose(2, 0, 1)
        image = torch.from_numpy(image)

        for transform in self.image_transforms:
            image = transform(image)

        return image

    def _get_contour_mask(self, image_id: str, image: torch.Tensor) -> np.ndarray:
        mask = np.zeros((image.size(1), image.size(2)))

        for annotation in self.polygons[image_id]:
            if annotation["type"] != "blood_vessel":
                continue
            for c in annotation["coordinates"]:
                # TODO: why not 0, 1 but 1, 0?
                row, col = np.array([x[1] for x in c]), np.array([y[0] for y in c])
                mask[row, col] = 1

        return mask.astype(np.uint8)

    def _get_filled_mask(self, image_id: str, image: torch.Tensor) -> np.ndarray:
        contour_mask = self._get_contour_mask(image_id, image)

        contours, _ = cv2.findContours(contour_mask * 255, 1, 2)
        mask = np.zeros((image.size(1), image.size(1), 3), dtype="uint8")

        for c in contours:
            cv2.fillPoly(mask, [c], (255, 255, 255))

        contours, hierarchy = cv2.findContours(contour_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            cv2.fillPoly(
                mask,
                [contours[i][:, 0, :]],
                (255 - 4 * (i + 1), 255 - 4 * (i + 1), 255 - 4 * (i + 1)),
                lineType=cv2.LINE_8,
                shift=0
            )
        return mask

    def _get_masks(self, image_id: str, image: torch.Tensor) -> torch.Tensor:
        mask = self._get_filled_mask(image_id, image)

        ids = np.unique(mask)[1:]
        masks = np.array(
            [np.where(mask == i, 1, 0) for i in ids],
            dtype=np.uint8
        )
        return torch.from_numpy(masks)

    @staticmethod
    def _get_boxes(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = []
        for mask in masks:
            mask = np.nonzero(np.array(mask))
            boxes.append([
                np.min(mask[1]), np.min(mask[0]),
                np.max(mask[1]), np.max(mask[0]),
            ])
        boxes = torch.Tensor(boxes)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return boxes, torch.Tensor(area)
