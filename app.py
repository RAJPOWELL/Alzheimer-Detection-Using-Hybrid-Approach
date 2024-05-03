# Import statements
from flask import Flask, request, render_template, session, redirect, url_for, flash
from PIL import Image
import numpy as np
import os
import sqlite3
from sqlite3 import Error
import hashlib
import tensorflow as tf
import tensorflow_addons as tfa
import functools  # Import functools for wraps decorator

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong, random secret key

# Load the trained model with custom metric function
model = tf.keras.models.load_model('/home/raj/Temp/Alzheimer-Detection-Using-Hybrid-Approach/Models/alzheimer_inception_cnn_model',
                                   custom_objects={'F1Score': tfa.metrics.F1Score})

# Define classes
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Function to create a database connection
def create_connection(db_file):
    """Creates a database connection to the specified file."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

# Function to execute a SQL query
def execute_query(conn, sql_query):
    """Executes a provided SQL query on the database connection."""
    try:
        c = conn.cursor()
        c.execute(sql_query)
    except Error as e:
        print(e)

# Function to hash the password securely using bcrypt
def hash_password(password):
    """Hashes the password using bcrypt for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

# Function to verify password during login
def verify_password(hashed_password, password):
    """Verifies the provided password against the hashed password stored in the database."""
    return hashed_password == hashlib.sha256(password.encode()).hexdigest()

# Function to validate if the uploaded file is an image
def allowed_file(filename):
    """Checks if the filename extension corresponds to a supported image format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Route for user signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Hash the password securely
        hashed_password = hash_password(password)

        # Create SQLite database connection
        conn = create_connection("/home/raj/Temp/BACKUP/0.4/Alzheimer-Detection-Using-Hybrid-Approach/Models/alzheimer_inception_cnn_model")

        if conn is not None:
            # Insert user data into User table
            sql_query = f"INSERT INTO User (username, password, email) VALUES ('{username}', '{hashed_password}', '{email}')"
            execute_query(conn, sql_query)
            conn.commit()
            conn.close()
            flash('Signup successful! Please log in.')
            return redirect(url_for('login'))
        else:
            flash('Error! Cannot create the database connection.')

    return render_template('signup.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Create SQLite database connection
        conn = create_connection("/home/raj/Temp/BACKUP/0.4/Alzheimer-Detection-Using-Hybrid-Approach/alzheimer_detection.db")

        if conn is not None:
            # Query the database for the user
            sql_query = f"SELECT * FROM User WHERE username = '{username}'"
            cursor = conn.cursor()
            cursor.execute(sql_query)
            user = cursor.fetchone()

            if user and verify_password(user[2], password):
                # User authenticated successfully
                session['logged_in'] = True
                session['username'] = username
                flash('Login successful!')
                return redirect(url_for('index'))
            else:
                flash('Invalid username or password. Please try again.')

    return render_template('login.html')

# Route for user logout
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

# Route for the Home Page (now the starting point)
@app.route('/')
def index():
    return render_template('home.html')

# Login required decorator (checks if user is logged in)
def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('You must be logged in to access this page.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Restrict access to MRI and Cognitive pages (use decorator)
@app.route('/mri')
@login_required
def mri():
    return render_template('mri.html')

@app.route('/cognitive')
@login_required
def cognitive():
    # Execute Application-1 here
    return redirect(url_for('question'))

# Route for cognitive test questions
@app.route('/question')
@login_required
def question():
    return render_template('questions.html')

# Route for image prediction
@app.route('/predict', methods=['POST'])
@login_required
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
    
    # Store predicted class and image source in session
    session['predicted_class'] = predicted_class
    session['image_filename'] = img_file.filename
    
    return render_template('results.html', predicted_class=predicted_class, image_filename=img_file.filename)

if __name__ == '__main__':
    app.run(debug=True)
