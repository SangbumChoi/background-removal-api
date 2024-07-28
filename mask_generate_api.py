from flask import Flask, jsonify, request
from InSpyReNet.inspyrenet import process_image, check_alpha_layer, get_polygon_from_mask
# from RepViT.repvit import repvit
# import cv2
# import numpy as np
import json
from PIL import Image
import io
import os


app = Flask(__name__)

def check_input_variables(request):
    # 요청에서 파일이 있는지 확인
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400
    
    if 'x' not in request.form or 'y' not in request.form:
        return jsonify({'error': 'No x or y coordinate in request'}), 400

    return True

def save_result(image_file, save_directory, image):
    original_filename = image_file.filename
    name, ext = os.path.splitext(original_filename)
    new_filename = f"result_{name}.png"
    save_path = os.path.join(save_directory, new_filename)
    # Save the image to the new path
    image.save(save_path, 'PNG')


# 기본 엔드포인트
@app.route('/')
def hello_world():
    return 'This API is mask generator using machine learning model.'

# JSON 데이터를 반환하는 엔드포인트
# @app.route('/api/rep_vit/point_based_mask', methods=['POST'])
# def generate_rep_vit_mask():
#     # 요청에서 파일이 있는지 확인
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file in request'}), 400
    
#     image_file = request.files['image']

#     # 이미지 파일을 메모리에서 읽음
#     image_bytes = np.fromstring(image_file.read(), np.uint8)
#     image = cv2.cvtColor(cv2.imdecode(image_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

#     try:
#         ort_outputs = repvit(image=image)
#         json_outputs = json.dumps({'ort_outputs': ort_outputs.tolist()})

#         response = {
#             'message': 'Image received',
#             'ort_outputs': json_outputs
#         }
#         return jsonify(response), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
# JSON 데이터를 반환하는 엔드포인트
@app.route('/api/inspyrenet/point_based_mask', methods=['POST'])
def generate_inspyrenet_mask():
    # 요청에서 파일이 있는지 확인
    check = check_input_variables(request=request)
    if not check:
        return check
    
    # Get the coordinates from the request
    x = int(request.form['x'])
    y = int(request.form['y'])

    # Get the images from the request
    image_file = request.files['image']

    # Read the image file from memory
    image = Image.open(io.BytesIO(image_file.read()))

    try:
        outputs = process_image(input_image=image, output_type="default")
        is_in_mask, alpha = check_alpha_layer(image=outputs, x=x, y=y)
        if not is_in_mask:
            return jsonify({'error': 'input point has no mask'}), 500
        contour, bounding_box = get_polygon_from_mask(image=image, mask=alpha, x=x, y=y)
        print(bounding_box)
        outputs = outputs.crop(bounding_box)
        # Save the image to a desired location
        save_result(image_file=image_file, save_directory='/Users/choisangbum/Downloads/background-removal-api/examples', image=outputs)
        json_outputs = json.dumps({'ort_outputs': outputs.tolist()})

        response = {
            'message': 'Image received',
            'ort_outputs': json_outputs
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

'''
/usr/bin/python3 /Users/choisangbum/Downloads/background-removal-api/mask_generate_api.py
curl -X POST http://127.0.0.1:5000/api/rep_vit/point_based_mask -F "image=@/Users/choisangbum/Downloads/background-removal-api/examples/example1.jpg"
curl -X POST http://127.0.0.1:5000/api/inspyrenet/point_based_mask -F "image=@/Users/choisangbum/Downloads/background-removal-api/examples/example1.jpg" -F "x=1280" -F "y=900"
curl -X POST http://127.0.0.1:5000/api/inspyrenet/point_based_mask -F "image=@/Users/choisangbum/Downloads/background-removal-api/examples/example2.jpg" -F "x=1200" -F "y=750"
'''
if __name__ == '__main__':
    app.run(debug=True)