from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

app = Flask(__name__)

# Load the trained model with custom metric function
model = tf.keras.models.load_model('/home/raj/Temp/Alzheimer-Detection-Using-Hybrid-Approach/Model/alzheimer_cnn_model',
                                   custom_objects={'F1Score': tfa.metrics.F1Score})

# Define classes
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['image']
    
    # Load the image using PIL
    img = Image.open(img_file)
    
    # Resize the image to the required dimensions and convert to array
    img_array = np.array(img.resize((176, 176))) / 255.0
    
    # Ensure that the image array has 3 channels (RGB)
    if img_array.shape[-1] != 3:
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension if it's missing
        img_array = np.repeat(img_array, 3, axis=-1)    # Repeat grayscale values across channels
    
    # Reshape the image array to match the expected input shape of the model
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = CLASSES[np.argmax(prediction)]
    
    return render_template('results.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
