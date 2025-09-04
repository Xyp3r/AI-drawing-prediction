# This Python script demonstrates a backend server for a Quick Draw-like application.
# It uses the Flask framework to create a simple web server.
#
# To run this script:
# 1. Install Flask and a deep learning library like TensorFlow:
#    pip install Flask tensorflow
# 2. Run the script from your terminal:
#    python app.py
#
# Note: This is a simplified example. In a real-world application,
# you would need to train a robust model on a large dataset like
# the Google Quick Draw dataset. This script uses a placeholder
# model that will return a hardcoded prediction.

import os
import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify 
from flask_cors import CORS
from PIL import Image
from PIL.Image import LANCZOS
import io

# Initialize Flask application
app = Flask(__name__)
# Enable CORS for the frontend application
CORS(app)

# --- Placeholder AI Model Setup ---
# A real Quick Draw model would be a Convolutional Neural Network (CNN)
# trained on a dataset of millions of drawings.
# This function simulates the prediction process.

def load_placeholder_model():
    # sourcery skip: inline-immediately-returned-variable, move-assign-in-block
    """
    Simulates loading a trained model.
    In a real application, you would load your TensorFlow/Keras model here.
    """
    class_names = ['apple', 'banana', 'cat', 'dog', 'tree']
    print("Placeholder model loaded.")
    return class_names

# Load the "model"
CLASS_NAMES = load_placeholder_model()

def preprocess_image(image_bytes):
    # sourcery skip: inline-immediately-returned-variable
    """
    Preprocesses the raw image data from the canvas.
    - Converts the image to grayscale.
    - Resizes the image to 28x28 pixels (common for simple models).
    - Inverts the colors (white drawing on black background).
    - Normalizes the pixel values.
    """
    # Open the image using Pillow (PIL)
    img = Image.open(io.BytesIO(image_bytes))
    # Convert to grayscale
    img = img.convert('L')
    # Resize to 28x28 pixels
    img = img.resize((28, 28), LANCZOS)
    # Convert to a numpy array
    img_array = np.array(img, dtype=np.float32)

    # Invert colors (Quick Draw dataset is usually black on white, but models
    # are often trained on white on black like MNIST)
    img_array = 255 - img_array
    
    # Normalize pixel values to be between 0 and 1
    img_array /= 255.0
    
    # Reshape for the model (add a batch dimension)
    img_array = img_array.reshape(1, 28, 28)
    
    return img_array

def get_prediction_from_model(processed_image):
    """
    This is a placeholder for the actual prediction logic.
    It returns a hardcoded prediction based on the input data,
    since we don't have a pre-trained model to load.
    
    In a real scenario, you would use:
    `predictions = model.predict(processed_image)`
    `predicted_class_index = np.argmax(predictions)`
    `prediction = CLASS_NAMES[predicted_class_index]`
    
    For this demonstration, we'll just simulate a prediction.
    """
    # The image data is just a placeholder to show it's being used.
    # The actual prediction is simulated by picking a random class name.
    # A real model would use the `processed_image` to make an accurate prediction.
    
    # Placeholder: Return a random class from our list
    random_index = np.random.randint(0, len(CLASS_NAMES))
    return CLASS_NAMES[random_index]

# --- Flask Routes ---

@app.route('/predict', methods=['POST'])
def predict_drawing():
    """
    Endpoint to receive drawing data and return a prediction.
    """
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Get the base64 encoded image string
    image_base64 = data['image']
    # Remove the "data:image/png;base64," prefix
    image_base64 = image_base64.split(',')[1]

    try:
        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(image_base64)
        
        # Preprocess the image for the model
        processed_image = preprocess_image(image_bytes)
        
        # Get the prediction from the model
        prediction = get_prediction_from_model(processed_image)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

# --- Main entry point ---
if __name__ == '__main__':
    # Run the Flask app on localhost, port 5000
    app.run(debug=True, port=5000)
