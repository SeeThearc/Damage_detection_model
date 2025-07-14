from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

#final code
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Parameters - MATCH YOUR COMMAND LINE CODE
IMG_SIZE = 128

# Load the model at startup
try:
    model = load_model("parcel_damage_model_final.h5")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None

# Function to prepare the image for prediction - EXACTLY LIKE COMMAND LINE
def prepare_image(img_path):
    # 🔍 Method 1: Using OpenCV (EXACTLY like command line)
    frame = cv2.imread(img_path)
    print(f"Original image shape: {frame.shape}")
    
    # Resize and normalize image - EXACTLY like command line
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img / 255.0, axis=0)
    
    # 🔍 Add debugging to match command line output
    print(f"After resize shape: {img.shape}")
    print(f"Final array shape: {img_array.shape}")
    print(f"Image range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    print(f"Image mean: {img_array.mean():.3f}")
    print(f"Image std: {img_array.std():.3f}")
    
    return img_array

# Alternative function using Keras (for comparison)
def prepare_image_keras(img_path):
    from tensorflow.keras.preprocessing import image
    # Load and resize the image to (128, 128)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    print(f"KERAS - Image shape: {img_array.shape}")
    print(f"KERAS - Image range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    print(f"KERAS - Image mean: {img_array.mean():.3f}")
    print(f"KERAS - Image std: {img_array.std():.3f}")
    
    return img_array

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        print(f"\n🔍 Processing file: {file.filename}")
        
        # Test both methods
        print("\n--- Using OpenCV (like command line) ---")
        img_array_cv = prepare_image(filepath)
        prediction_cv = model.predict(img_array_cv)
        prediction_value_cv = prediction_cv[0][0]
        
        print("\n--- Using Keras preprocessing ---")
        img_array_keras = prepare_image_keras(filepath)
        prediction_keras = model.predict(img_array_keras)
        prediction_value_keras = prediction_keras[0][0]
        
        print(f"\nCV2 Prediction: {prediction_value_cv:.4f}")
        print(f"Keras Prediction: {prediction_value_keras:.4f}")
        
        # Use OpenCV method (matches command line)
        prediction_value = prediction_value_cv
        label = "Damaged" if prediction_value < 0.5 else "Intact"
        
        print(f"Final Result: {label}")
        
        return jsonify({
            "prediction": prediction_cv.tolist(),
            "prediction_value": float(prediction_value),
            "label": label,
            "debug": {
                "cv2_prediction": float(prediction_value_cv),
                "keras_prediction": float(prediction_value_keras),
                "difference": abs(float(prediction_value_cv) - float(prediction_value_keras))
            }
        })
    except Exception as e:
        print(f"Error details: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# Root route to test the server
@app.route("/", methods=["GET"])
def home():
    return "✅ Flask backend is running!"

if __name__ == "__main__":
    app.run(debug=True)