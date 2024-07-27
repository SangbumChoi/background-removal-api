import cv2
from typing import Any, Tuple, Union
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = ArgumentParser()

parser.add_argument('--model', default='vit_t', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--image', default=None, type=str)

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

if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()

    raw_img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    raw_img_shape = raw_img.shape
    print('raw_image_shape', raw_img_shape)
    raw_img = preprocess(raw_img, img_size=1024)

    output = f"RepViT/onnx/{args.model}_encoder_{args.resolution}.onnx"

    # set cpu provider default
    providers = ["CPUExecutionProvider"]

    ort_encoder_inputs = {"x": raw_img}
    ort_session = onnxruntime.InferenceSession(output, providers=providers)
    ort_outputs = ort_session.run(None, ort_encoder_inputs)
    print('feature_shape', ort_outputs[0].shape)

    output = f"RepViT/onnx/{args.model}_decoder_{args.resolution}.onnx"
    mask_input_size = [x for x in [256, 256]]
    point_coords = np.random.randint(low=0, high=args.resolution, size=(1, 1, 2)).astype(np.float32)
    mask_input = np.zeros((1, 1, *mask_input_size), dtype=np.float32)
    has_mask_input = np.zeros(1, dtype=np.float32)
    point_labels = np.array([[1]], dtype=np.float32)

    ort_decoder_inputs = {
        "image_embeddings": ort_outputs[0],
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
        "orig_im_size": np.array(raw_img_shape[:2]).astype(np.float32),
    }

    ort_session = onnxruntime.InferenceSession(output, providers=providers)
    ort_outputs = ort_session.run(None, ort_decoder_inputs)

    print(ort_outputs[0][0][0])

    # 이미지 시각화
    plt.imshow(ort_outputs[0][0][0] > 0)
    plt.scatter(point_coords[0][0][0] * raw_img_shape[0] / args.resolution, point_coords[0][0][1] * raw_img_shape[0] / args.resolution, color='red', marker='*', s=200)  # s는 marker의 크기
    plt.title("Random Image")
    plt.axis('off')  # 축을 표시하지 않음
    plt.show()
