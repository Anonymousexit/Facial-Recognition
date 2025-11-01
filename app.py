from flask import Flask, render_template, request, url_for
import sqlite3
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from werkzeug.utils import secure_filename

# Initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained model
model = load_model('face_emotionModel.h5')

# Emotion classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Database setup
def init_db():
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

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    file = request.files['image']

    if not file:
        return "No image uploaded", 400

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

    # Save to database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (name, email, emotion, image_path) VALUES (?, ?, ?, ?)",
              (name, email, emotion, filepath))
    conn.commit()
    conn.close()

    # Show result page
    return render_template('result.html', name=name, emotion=emotion, image_file=filename)

if __name__ == '__main__':
    app.run(debug=True)
