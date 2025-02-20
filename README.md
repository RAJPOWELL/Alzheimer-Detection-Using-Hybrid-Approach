# Alzheimer's Disease Prediction using Flask and TensorFlow

This is a web application for predicting Alzheimer's disease stages from MRI images using a deep learning model. The application is built with Flask and TensorFlow.

## Features
- Upload MRI images for classification  
- Get predictions with confidence scores  
- Simple and interactive web interface  

## Requirements  
Ensure you have the following dependencies installed:  

```
pip install -r requirements.txt
```

### Key Dependencies:  
- Python 3.6+  
- Flask  
- TensorFlow/Keras  
- OpenCV (cv2)  
- NumPy  
- Pillow  

## Installation  

1. Clone the repository:  
   ```
   git clone https://github.com/yourusername/alzheimers-prediction.git  
   cd alzheimers-prediction  
   ```

2. Set up a virtual environment (optional but recommended):  
   ```
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   ```

3. Install dependencies:  
   ```
   pip install -r requirements.txt  
   ```

4. Run the Flask app:  
   ```
   python app.py  
   ```

5. Open the application in your browser:  
   ```
   http://127.0.0.1:5000  
   ```

## Project Structure  
```
/alzheimers-prediction  
│── /static           # CSS, JS, images  
│── /templates        # HTML templates  
│── /models           # Trained model files  
│── app.py           # Flask application  
│── requirements.txt  # Dependencies  
│── README.md         # Documentation  
```

## Usage  
- Upload an MRI image  
- The model will predict the Alzheimer’s disease stage  
- Confidence scores will be displayed  




