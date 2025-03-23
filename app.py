import os
import traceback
import logging
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, static_folder='./static', template_folder='./templates')
CORS(app)

UPLOAD_FOLDER = 'uploads'
MODEL_DIR = os.path.join("SORT", "model.savedmodel")
LABELS_FILE = os.path.join("SORT", "labels.txt")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = None
class_names = []

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the TensorFlow model and class labels."""
    global model, class_names
    try:
        logging.info(f"🔄 Loading model from {MODEL_DIR}...")
        model = tf.saved_model.load(MODEL_DIR)

        if not os.path.exists(LABELS_FILE):
            raise FileNotFoundError(f"Labels file not found at {LABELS_FILE}")

        with open(LABELS_FILE, "r") as file:
            class_names[:] = [line.strip() for line in file.readlines()]

        logging.info("✅ Model and labels loaded successfully!")
        return True

    except Exception as e:
        logging.error(f"❌ Error loading model: {e}")
        return False

def preprocess_image(image_path):
    """Preprocess an image for TensorFlow model prediction."""
    try:
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image, dtype=np.float32) / 127.5 - 1
        return np.expand_dims(image_array, axis=0)

    except Exception as e:
        logging.error(f"❌ Error processing image: {e}")
        return None

@app.route('/', methods=['GET'])
def serve_html():
    return render_template('sell.html')

@app.route('/', methods=['POST'])
def predict():
    """Handle image uploads and run predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400

    selected_category = request.form.get('category', None)
    
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the image
        input_data = preprocess_image(file_path)
        if input_data is None:
            return jsonify({'error': 'Image processing failed'}), 500

        # Perform inference
        infer = model.signatures.get("serving_default")
        if infer is None:
            return jsonify({'error': 'Model signature "serving_default" not found'}), 500

        output = infer(tf.constant(input_data))
        prediction = list(output.values())[0].numpy()
        index = np.argmax(prediction)
        class_name = class_names[index] if index < len(class_names) else "Unknown"
        confidence_score = float(prediction[0][index])
        
        is_match = bool(selected_category and class_name.lower() == selected_category.lower())

        return jsonify({
            'class': class_name,
            'confidence': confidence_score,
            'selected_category': selected_category,
            'is_match': is_match
        })

    except Exception as e:
        logging.error(f"❌ Exception in /predict: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if load_model():
        app.run(debug=True, port=3000)
    else:
        logging.error("❌ Failed to load model. Exiting...")
