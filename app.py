import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Safe temp folder for Render
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once
# NOTE: Make sure your model path is correct (e.g., "model.h5")
MODEL_PATH = "model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print("⚠️ Error loading model:", e)
    model = None

# Define emotion classes (adjust to match your model)
CLASS_NAMES = ['Happy', 'Sad', 'Angry', 'Neutral']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"✅ Saved file to {filepath}")

    # Preprocess image for model
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize((48, 48))  # adjust to your model’s expected size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = float(np.max(preds))

        result = {
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2)
        }
        return jsonify(result)

    except Exception as e:
        print("⚠️ Prediction error:", e)
        return jsonify({"error": "Failed to process image"}), 500


if __name__ == '__main__':
    # Use PORT from environment (Render uses it)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
