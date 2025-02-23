from flask import Flask, request, render_template
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model (update the path if needed)
model = load_model('model/mnist_model.h5')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file found", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    if file:
        # Save the uploaded file temporarily
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read and process the saved image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "The uploaded file is not a valid image.", 400

        img = cv2.resize(img, (28, 28)) / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Make a prediction
        prediction = model.predict(img)
        label = np.argmax(prediction)

        # Return results with the uploaded image
        return render_template('index.html', prediction=label, image_path=file_path)

    return "Something went wrong", 500


if __name__ == "__main__":
    app.run(debug=True)
