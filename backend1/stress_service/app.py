import os
import numpy as np
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# --- Model Loading ---
stress_model = tf.keras.models.load_model("stress_model.h5")
scaler = joblib.load("scaler.pkl")
INTERNAL_SERVICE_KEY = os.environ.get("INTERNAL_SERVICE_KEY")

@app.route("/predict", methods=["POST"])
def predict_stress():
    # Security check
    if request.headers.get("X-Internal-Service-Key") != INTERNAL_SERVICE_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    features = np.array([[float(data["heart_rate"]), float(data["steps"]), float(data["sleep"]), float(data["age"])]])
    scaled_features = scaler.transform(features)
    prediction = stress_model.predict(scaled_features)
    stress_level = int(np.round(prediction[0][0]))
    stress_level = max(0, min(10, stress_level))
    
    # (Suggestion logic would also be here)

    return jsonify({"stress_level": stress_level})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)
