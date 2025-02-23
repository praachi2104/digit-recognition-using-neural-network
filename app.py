from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model (update the path if needed)
model = load_model('model/mnist_cnn_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict-drawing', methods=['POST'])
def predict_drawing():
    try:
        data = request.get_json()
        image_data = data['image']

        # Decode the image
        img_data = base64.b64decode(image_data.split(',')[1])
        img = Image.open(BytesIO(img_data)).convert('L')
        img = np.array(img)

        # Visualize the original image (for debugging)
        print("Original Image Shape:", img.shape)  # Should be (height, width), e.g., (280, 280)

        # Resize and normalize the image for prediction
        img = cv2.resize(img, (28, 28))  # Resize to 28x28 pixels (MNIST standard)
        print("Resized Image Shape:", img.shape)  # Should be (28, 28)

        # Normalize to [0, 1]
        img = img / 255.0
        print("Normalized Image:", img.min(), img.max())  # Ensure values are between 0 and 1

        # Reshape for model input (1, 28, 28, 1) because the model expects this shape
        img = img.reshape(1, 28, 28, 1)

        # Make a prediction
        prediction = model.predict(img)
        label = np.argmax(prediction)

        # Return the prediction
        return jsonify({
            'prediction': str(label),
            'imagePath': image_data
        })
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    app.run(debug=True)
