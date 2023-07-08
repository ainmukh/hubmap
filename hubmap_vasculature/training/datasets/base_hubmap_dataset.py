from abc import ABC
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from pathlib import Path
from tqdm.auto import tqdm
import json

import pandas as pd
from torch.utils.data import Dataset


class BaseHuBMAPDataset(ABC, Dataset):
    def __init__(
            self,
            data_path: Union[str, Path],
            part: str,
            image_transforms: Union[Sequence[Callable], Callable] = None,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Data path {data_path} does not exist")
        self.data_path = data_path
        self.ids, self.polygons = self._load_data(data_path, part=part)
        self.part = part

        if image_transforms is None:
            self.image_transforms = []
        elif not isinstance(image_transforms, Sequence):
            self.image_transforms = [image_transforms]
        else:
            self.image_transforms = image_transforms

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _load_data(path: Union[str, Path], part: str) -> Tuple[List[str], Dict[str, Any]]:
        # TODO: splits and folds
        tile_meta = pd.read_csv(path / "tile_meta.csv")
        if part == "test":
            tile_meta = tile_meta[tile_meta["dataset"] >= 3]
        else:
            tile_meta = tile_meta[tile_meta["dataset"] < 3]

        with open(str(path / "polygons.jsonl"), "r") as f:
            polygons_jsonl = list(f)

        polygons = {}
        for json_str in tqdm(polygons_jsonl, total=len(polygons_jsonl), desc="Read polygons.jsonl file"):
            json_data = json.loads(json_str)
            polygons[json_data["id"]] = json_data["annotations"]

        ids = list(tile_meta.id)
        return ids, polygons
