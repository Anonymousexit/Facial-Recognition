import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import cv2
import logging

# ===========================
# Flask App Setup
# ===========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging setup (Render displays these in logs)
logging.basicConfig(level=logging.INFO)
logger = app.logger

# ===========================
# Model Setup (Lazy Loading)
# ===========================
model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def get_model():
    """Load model only once and reuse across requests"""
    global model
    if model is None:
        model_path = os.path.join(os.getcwd(), 'face_emotionModel.h5')
        logger.info(f"Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
    return model

# ===========================
# Helper Function
# ===========================
def predict_emotion(image_path):
    """Run emotion prediction on uploaded image"""
    model = get_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image file")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "No face detected"

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (48, 48))
    roi = roi_gray.astype('float') / 255.0
    roi = np.expand_dims(roi, axis=0)
    roi = np.expand_dims(roi, axis=-1)

    predictions = model.predict(roi)
    max_index = int(np.argmax(predictions))
    emotion = emotion_labels[max_index]
    return emotion

# ===========================
# Routes
# ===========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Allow Render to serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved upload to {filepath}")

        emotion = predict_emotion(filepath)
        logger.info(f"Predicted emotion: {emotion}")

        return jsonify({'emotion': emotion})

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ===========================
# Run the app
# ===========================
if __name__ == '__main__':
    # Run locally
    app.run(host='0.0.0.0', port=10000, debug=True)
