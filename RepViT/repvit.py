from utils import preprocess
import numpy as np

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

encoder_file = f"RepViT/onnx/repvit_encoder_1024.onnx"
decoder_file = f"RepViT/onnx/repvit_decoder_1024.onnx"

# set cpu provider default
providers = ["CPUExecutionProvider"]

ort_encoder_session = onnxruntime.InferenceSession(encoder_file, providers=providers)
ort_decoder_session = onnxruntime.InferenceSession(decoder_file, providers=providers)


# currently random point
def repvit(image):
    # 이미지를 열기        
    raw_img_shape = image.shape
    print('raw_image_shape', raw_img_shape)
    raw_img = preprocess(image, img_size=1024)

    ort_encoder_inputs = {"x": raw_img}
    ort_outputs = ort_encoder_session.run(None, ort_encoder_inputs)
    print('feature_shape', ort_outputs[0].shape)

    mask_input_size = [x for x in [256, 256]]
    point_coords = np.random.randint(low=0, high=1024, size=(1, 1, 2)).astype(np.float32)
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

    ort_outputs = ort_decoder_session.run(None, ort_decoder_inputs)[0]
    return ort_outputs