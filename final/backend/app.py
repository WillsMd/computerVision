from flask import Flask, request, jsonify
from face_analysis import analyze_face
from object_detection import detect_objects
from filters import apply_filter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/face-analysis', methods=['POST'])
def face_analysis():
    data = request.files['image']
    result = analyze_face(data)
    return jsonify(result)

@app.route('/api/object-detection', methods=['POST'])
def object_detection():
    data = request.files['image']
    result = detect_objects(data)
    return jsonify(result)

@app.route('/api/apply-filters', methods=['POST'])
def apply_filters():
    print("yea")
    data = request.files['image']
    filter_type = request.form.get('filter_type', 'grayscale')
    result = apply_filter(data, filter_type)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)