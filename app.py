from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

app = Flask(__name__)

# Define custom metric function
def f1_score(y_true, y_pred):
    # Implementation of F1 score metric
    pass

# Load the trained model with custom metric function
model = tf.keras.models.load_model('/home/raj/Temp/Alzheimer-Detection-Using-Hybrid-Approach/Model/alzheimer_cnn_model',
                                   custom_objects={'f1_score': f1_score})

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
    
    # Convert the image to RGB if it has a single channel
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize the image to the required dimensions
    img = img.resize((176, 176))
    
    # Convert the image to numpy array
    img_array = np.array(img) / 255.0  # Normalize the image
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Get the predicted class
    predicted_class = CLASSES[np.argmax(prediction)]
    
    return render_template('results.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
