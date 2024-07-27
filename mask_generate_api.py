from flask import Flask, jsonify, request

app = Flask(__name__)

# 기본 엔드포인트
@app.route('/')
def hello_world():
    return 'This API is mask generator using machine learning model.'

# JSON 데이터를 반환하는 엔드포인트
@app.route('/api/rep_vit/point_based_mask', methods=['GET'])
def get_data(point):
    data = {
        'name': 'John Doe',
        'age': 30,
        'city': 'New York'
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
    # curl http://127.0.0.1:5000/api/data