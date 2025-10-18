from flask import Flask, render_template, request
import os
import uuid
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model("model.h5")

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
# Reverse mapping: index -> label
class_labels = {v: k.capitalize() for k, v in class_indices.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    pred = model.predict(x)[0]  # shape (3,)
    pred_class = np.argmax(pred)
    confidence = round(pred[pred_class] * 100, 2)
    prediction = class_labels[pred_class]

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True)
