from flask import Flask, render_template, request
import sqlite3
import os
import logging
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model safely
MODEL_PATH = os.path.join(os.getcwd(), 'face_emotionModel.h5')
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
else:
    logger.error(f"❌ Model file not found at {MODEL_PATH}")

# Emotion classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize database
def init_db():
    conn = sqlite3.connect('/tmp/database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT,
                  emotion TEXT,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model not loaded. Check server logs for details.")
    try:
        name = request.form['name']
        email = request.form['email']
        file = request.files['image']

        if not file:
            return render_template('index.html', error="No image uploaded.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        img = load_img(filepath, target_size=(48, 48), color_mode='grayscale')
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        # Predict emotion
        pred = model.predict(img)
        emotion = classes[np.argmax(pred)]
        confidence = round(float(np.max(pred)) * 100, 2)

        # Save to database
        conn = sqlite3.connect('/tmp/database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, emotion, image_path) VALUES (?, ?, ?, ?)",
                  (name, email, emotion, filepath))
        conn.commit()
        conn.close()

        # Render same page with result
        return render_template(
            'index.html',
            emotion=emotion,
            confidence=confidence,
            image_file=filename,
            name=name
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
