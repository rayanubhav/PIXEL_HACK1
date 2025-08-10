import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# --- Model Loading ---
facial_emotion_model = None
try:
    facial_emotion_model = joblib.load("emotion_classifier_rf_TUNED.joblib")
    print("✅ Facial emotion model loaded.")
except Exception as e:
    print(f"❌ Error loading facial emotion model: {e}")

facial_emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
INTERNAL_SERVICE_KEY = os.environ.get("INTERNAL_SERVICE_KEY")

@app.route("/predict", methods=["POST"])
def predict_emotion():
    if request.headers.get("X-Internal-Service-Key") != INTERNAL_SERVICE_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    if not facial_emotion_model:
        return jsonify({"error": "Facial emotion model is not available."}), 503
        
    try:
        data = request.json
        feature_vector = np.array([[data['avg_ear'], data['mar'], data['eyebrow_dist'], data['jaw_drop']]])
        predicted_class = facial_emotion_model.predict(feature_vector)[0]
        emotion = facial_emotion_labels[predicted_class]
        prediction_proba = facial_emotion_model.predict_proba(feature_vector)[0]
        confidence = round(max(prediction_proba) * 100, 2)

        return jsonify({"emotion": emotion, "confidence": confidence})
    except Exception as e:
        print(f"Error in /predict-emotion: {e}")
        return jsonify({"error": "An error occurred during emotion prediction."}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5004))
    app.run(host="0.0.0.0", port=port)
