import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and scaler
knn_model = joblib.load('model/knn_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('image_data')  # Get the 16x16 grid from frontend
    
    if not data:
        return jsonify({'error': 'No image data received'}), 400

    image_data = np.array(data).reshape(1, -1)  # Convert to numpy array
    image_data_scaled = scaler.transform(image_data)  # Scale data

    prediction = knn_model.predict(image_data_scaled)  # Get prediction
    print(f"Predicted: {prediction}")  # Debugging line

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
