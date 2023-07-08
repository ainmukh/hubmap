from typing import Any, Dict, Tuple, Union
import torch
import pytorch_lightning as pl
from torch import Tensor


# TODO: refactor
class BaseLightningModule(pl.LightningModule):
    def __init__(self, lr: float, model: torch.nn.Module):
        super().__init__()
        self.lr = lr
        self.model = model

    def transfer_batch_to_device(
            self, batch: Union[Tuple, Dict[str, Any]], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Any]:
        if isinstance(batch, dict):
            for field, value in batch.items():
                if isinstance(value, Tensor):
                    batch[field] = value.to(device)
        elif isinstance(batch, tuple):
            images, targets = batch
            images = images.to(device)
            targets = [self.transfer_batch_to_device(target, device, dataloader_idx) for target in targets]
            batch = images, targets
        return batch

    def forward(self, batch: Dict[str, Tensor]) -> Any:
        return self.model(*batch)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> torch.Tensor:
        loss_dict = self.forward(batch)
        return sum(loss_dict.values())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            list(self.parameters()),
            lr=self.lr
        )
