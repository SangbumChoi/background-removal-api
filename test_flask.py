from flask import Flask, jsonify, request

app = Flask(__name__)

# 기본 엔드포인트
@app.route('/')
def hello_world():
    return 'Hello, World!'

# JSON 데이터를 반환하는 엔드포인트
@app.route('/api/data', methods=['GET'])
def get_data():
    data = {
        'name': 'John Doe',
        'age': 30,
        'city': 'New York'
    }
    return jsonify(data)

# POST 요청을 처리하는 엔드포인트
@app.route('/api/data', methods=['POST'])
def post_data():
    new_data = request.get_json()
    response = {
        'message': 'Data received',
        'data': new_data
    }
    return jsonify(response), 201

if __name__ == '__main__':
    app.run(debug=True)
    # curl http://127.0.0.1:5000/api/data