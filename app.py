import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Flask app initialization
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Constants
MODEL_PATH = "Model/keras_model.h5"
LABELS_PATH = "Model/labels.txt"

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load model once at startup
model = load_model(MODEL_PATH, compile=False)

# Load labels
def load_labels():
    with open(LABELS_PATH, "r") as f:
        return [line.strip().split(maxsplit=1)[-1] for line in f.readlines()]

LABELS = load_labels()

# Preprocess image to match model input
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# Welcome screen with redirect to login
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Home page with navigation links
@app.route('/home')
def home():
    return render_template('home.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Handle signup data if needed
        return redirect(url_for('login'))
    return render_template('signup.html')

# Accuracy page with performance graphs
@app.route('/accuracy')
def accuracy():
    return render_template('accuracy.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password:
            session['user'] = username
            return redirect(url_for('form'))
        else:
            return render_template('login.html', error="Invalid Credentials!")
    return render_template('login.html')

# Form page for user info
@app.route('/form', methods=['GET', 'POST'])
def form():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        session['name'] = request.form.get('name')
        session['age'] = request.form.get('age')
        session['gender'] = request.form.get('gender')
        return redirect(url_for('index'))
    return render_template('form.html')

# Index: image upload & prediction UI
@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

# Upload page (alternative route)
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Watch video page
@app.route('/watch_video')
def watch_video():
    return render_template('watch_video.html')

# Prediction route (API)
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        image = Image.open(file).convert("RGB")
    except:
        return jsonify({'error': 'Invalid image'}), 400

    input_tensor = preprocess_image(image)
    prediction = model.predict(input_tensor)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = LABELS[predicted_index]
    confidence_score = float(prediction[predicted_index])
    probabilities = {LABELS[i]: float(prob) for i, prob in enumerate(prediction)}

    return jsonify({
        'predicted_label': predicted_label,
        'confidence_score': confidence_score,
        'probabilities': probabilities,
        'name': session.get('name', 'Unknown'),
        'age': session.get('age', 'Unknown'),
        'gender': session.get('gender', 'Unknown')
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
