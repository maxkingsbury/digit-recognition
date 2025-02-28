import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from flask import Flask, render_template, request
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model and scaler
knn = pickle.load(open('model/knn_model.pkl', 'rb'))
scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        img_data = request.form['image']
        print("Received image data:", img_data)  # Debug line to check if image data is received
        
        # Remove the base64 header
        img_data = img_data.split(",")[1]
        
        # Decode the image
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes)).convert('L')

        # Resize the image to 8x8 pixels (as the model was trained on 8x8 images)
        new_size = (8, 8)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert the image to a numpy array
        image = np.array(img)

        # Flatten the image to a 1D array with 64 elements (8x8)
        image_flat = image.flatten().reshape(1, -1)

        # Scale the image
        image_scaled = scaler.transform(image_flat)

        # Predict using the KNN model
        prediction = knn.predict(image_scaled)

        # Return the prediction as a string
        return str(prediction[0])  # Ensure the return is properly parsed
    except Exception as e:
        print("Error:", e)  # Print the error to understand what went wrong
        return f"Error: {str(e)}"
    
    @app.route('/predict_static', methods=['GET'])
    def predict_static():
        try:
            # Test with a static image
            img = Image.open('static/test_image.png').convert('L')
            img = img.resize((8, 8), Image.Resampling.LANCZOS)
            image = np.array(img)
            image_flat = image.flatten().reshape(1, -1)
            image_scaled = scaler.transform(image_flat)
            prediction = knn.predict(image_scaled)
            return str(prediction[0])
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
