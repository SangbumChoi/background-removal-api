import torch

from timm import create_model

import torch
import torchvision
from argparse import ArgumentParser
from timm.models import create_model
import repvit_sam.modeling

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = ArgumentParser()

parser.add_argument('--model', default='vit_t', type=str)
parser.add_argument('--resolution', default=224, type=int)
parser.add_argument('--ckpt', default=None, type=str)
parser.add_argument('--samckpt', default=None, type=str)
parser.add_argument('--precision', default='fp16', type=str)

if __name__ == "__main__":
    # Load a pre-trained version of MobileNetV2
    args = parser.parse_args()
    model = create_model(args.model)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt)['model'])
    if args.samckpt:
        state = torch.load(args.samckpt, map_location='cpu')
        new_state = {}
        for k, v in state.items():
            if not 'image_encoder' in k:
                continue
            new_state[k.replace('image_encoder.', '')] = v
        model.load_state_dict(new_state)
    model.eval()

    # Trace the model with random data.
    resolution = args.resolution
    example_input = torch.rand(1, 3, resolution, resolution)
    dummy_inputs = {"x": example_input}
    output_names = ["image_embeddings"]
    output = f"onnx/{args.model}_encoder_{resolution}.onnx"

    with open(output, "wb") as f:
        print(f"Exporting onnx model to {output}...")
        torch.onnx.export(model, example_input, f, 
                export_params=True,
                verbose=False,
                opset_version=11,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()), 
                output_names=output_names)

    def to_numpy(tensor):
        return tensor.cpu().numpy()

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        # set cpu provider default
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")