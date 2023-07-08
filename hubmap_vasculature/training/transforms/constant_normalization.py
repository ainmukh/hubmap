import torch


class ConstantNormalization:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image / 127.5 - 1
