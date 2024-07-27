from flask import Flask, jsonify, request
from utils import preprocess
import cv2
import numpy as np
import json

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

app = Flask(__name__)

encoder_file = f"RepViT/onnx/repvit_encoder_1024.onnx"
decoder_file = f"RepViT/onnx/repvit_decoder_1024.onnx"

# set cpu provider default
providers = ["CPUExecutionProvider"]

ort_encoder_session = onnxruntime.InferenceSession(encoder_file, providers=providers)
ort_decoder_session = onnxruntime.InferenceSession(decoder_file, providers=providers)


# 기본 엔드포인트
@app.route('/')
def hello_world():
    return 'This API is mask generator using machine learning model.'

# JSON 데이터를 반환하는 엔드포인트
@app.route('/api/rep_vit/point_based_mask', methods=['POST'])
def get_data():
    # 요청에서 파일이 있는지 확인
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400
    
    image_file = request.files['image']

    # 이미지 파일을 메모리에서 읽음
    image_bytes = np.fromstring(image_file.read(), np.uint8)
    image = cv2.cvtColor(cv2.imdecode(image_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    try:
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
        json_outputs = json.dumps({'ort_outputs': ort_outputs.tolist()})

        response = {
            'message': 'Image received',
            'ort_outputs': json_outputs
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    # /usr/bin/python3 /Users/choisangbum/Downloads/background-removal-api/mask_generate_api.py
    # curl -X POST http://127.0.0.1:5000/api/rep_vit/point_based_mask -F "image=@/Users/choisangbum/Downloads/background-removal-api/examples/example1.jpg"
