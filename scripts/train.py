from pathlib import Path

import loguru
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from hubmap_vasculature.training.datasets import TorchInstanceSegmentationDataset
from hubmap_vasculature.training.lightning_modules import BaseLightningModule
from hubmap_vasculature.training.models.torchvision_model import get_model_instance_segmentation
from hubmap_vasculature.training.collate_fn.collate import collate_fn
from hubmap_vasculature.training.transforms.constant_normalization import ConstantNormalization


PROJECT_PATH = Path(__file__).parent.parent


def main():
    train_dataset = TorchInstanceSegmentationDataset(
        data_path=PROJECT_PATH / "data",
        part="train",
        image_transforms=ConstantNormalization(),
    )
    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)

    pl_module = BaseLightningModule(
        lr=3e-4,
        model=get_model_instance_segmentation(2),
    )

    wandb_logger = WandbLogger(
        project="hubmap",
        name="some_name"
    )

    trainer = pl.Trainer(max_epochs=25, logger=wandb_logger, log_every_n_steps=50)
    trainer.fit(pl_module, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    main()
