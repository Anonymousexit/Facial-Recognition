# app.py
from flask import Flask, render_template, request
import sqlite3, os, shutil, logging
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Use /tmp for writes on Render; still create static/uploads for display
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)

# Model loading (absolute path)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'face_emotionModel.h5')
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully from %s", MODEL_PATH)
    else:
        logger.error("Model file not found at %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model: %s", e)

classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

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
        logger.info("Database initialized.")
    except Exception as e:
        logger.exception("DB init error: %s", e)

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            logger.error("Prediction attempted but model is not loaded.")
            return "Model not loaded on server.", 500

        name = request.form.get('name', '')
        email = request.form.get('email', '')
        file = request.files.get('image', None)
        if file is None:
            return "No image uploaded", 400

        filename = secure_filename(file.filename)
        tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(tmp_path)
        logger.info("Saved upload to %s", tmp_path)

        # Preprocess and predict
        img = load_img(tmp_path, target_size=(48,48), color_mode='grayscale')
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0
        pred = model.predict(img)
        emotion = classes[int(np.argmax(pred))]

        # Copy to static for display
        public_path = os.path.join('static', 'uploads', filename)
        shutil.copy(tmp_path, public_path)
        logger.info("Copied to public path %s", public_path)

        # Save record
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, emotion, image_path) VALUES (?, ?, ?, ?)",
                  (name, email, emotion, public_path))
        conn.commit()
        conn.close()

        return render_template('index.html', image_file=filename, emotion=emotion, name=name)
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return f"Server error: {str(e)}", 500

if __name__ == '__main__':
    # local debug (not used on Render)
    app.run(host='0.0.0.0', port=5000, debug=True)
