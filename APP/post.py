from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        feature_values = [data['feature1'], data['feature2'], data['feature3'], data['feature4'], data['feature5'], data['feature6']]
        feature_values = np.array(feature_values).reshape(1, -1)
        result = model.predict(feature_values)
        return jsonify({'prediction': str(result[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
