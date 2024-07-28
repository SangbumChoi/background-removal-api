from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize

class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """

        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"

def to_numpy(tensor):
    return tensor.cpu().numpy()

def preprocess(x, img_size):
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy().astype(np.float32)

    return x