from flask import Flask, render_template, request
import sqlite3
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Directories
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)

# Load model safely
MODEL_PATH = 'face_emotionModel.h5'
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully.")
else:
    logger.error("❌ Model file not found! Please upload face_emotionModel.h5.")
    model = None

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Database setup
def init_db():
    try:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT,
                      email TEXT,
                      emotion TEXT,
                      image_path TEXT)''')
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized.")
    except Exception as e:
        logger.error(f"Database error: {e}")

init_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return "Model not found on server. Please re-upload your model file.", 500

        name = request.form['name']
        email = request.form['email']
        file = request.files['image']

        if not file:
            return "No image uploaded.", 400

        filename = secure_filename(file.filename)
        tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(tmp_path)

        # Preprocess image
        img = load_img(tmp_path, target_size=(48, 48), color_mode='grayscale')
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        # Predict
        pred = model.predict(img)
        emotion = classes[np.argmax(pred)]

        # Copy to static for display
        public_path = os.path.join('static/uploads', filename)
        shutil.copy(tmp_path, public_path)

        # Save to DB
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, emotion, image_path) VALUES (?, ?, ?, ?)",
                  (name, email, emotion, public_path))
        conn.commit()
        conn.close()

        return render_template('index.html', image_file=filename, emotion=emotion, name=name)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"Server error: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
