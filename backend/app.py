import os
import random
import threading
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
import numpy as np
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Lazy globals (models + DB) ---
stress_model = None
stress_scaler = None

chatbot_model = None
chatbot_tokenizer = None
_chatbot_device = None

facial_emotion_model = None
facial_emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# DB globals
_client = None
users_collection = None
stress_logs_collection = None
chat_logs_collection = None

# -------------------------
# Database (lazy) helpers
# -------------------------
def ensure_db_connected():
    global _client, users_collection, stress_logs_collection, chat_logs_collection
    if users_collection is not None:
        return True

    MONGO_URI = os.environ.get("MONGO_URI")
    if not MONGO_URI:
        logger.warning("MONGO_URI not set; DB features will be unavailable.")
        return False

    try:
        from pymongo import MongoClient
        # short timeout to avoid long blocking during import/startup
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = _client["mindcare_db"]
        users_collection = db["users"]
        stress_logs_collection = db["stress_logs"]
        chat_logs_collection = db["chat_logs"]
        # Try server selection quickly (will raise if cannot reach)
        _client.server_info()
        logger.info("✅ MongoDB connection established.")
        return True
    except Exception as e:
        logger.error(f"❌ Error connecting to MongoDB: {e}")
        users_collection = None
        return False

# -------------------------
# Model loader functions (lazy)
# -------------------------
def load_stress_model():
    global stress_model, stress_scaler
    if stress_model is not None:
        return
    try:
        logger.info("🌀 Loading stress model (lazy)...")
        # import heavy libs here
        import tensorflow as tf
        import joblib
        stress_model = tf.keras.models.load_model("stress_model.h5")
        stress_scaler = joblib.load("scaler.pkl")
        logger.info("✅ Stress model & scaler loaded.")
    except Exception as e:
        logger.error(f"❌ Failed to load stress model: {e}")
        stress_model = None
        stress_scaler = None

def load_chatbot_model():
    global chatbot_model, chatbot_tokenizer, _chatbot_device
    if chatbot_model is not None:
        return
    try:
        logger.info("🌀 Loading chatbot model (lazy)...")
        # heavy imports inside loader
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
        import torch
        chatbot_model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
        chatbot_tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")
        _chatbot_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        chatbot_model.to(_chatbot_device)
        chatbot_model.eval()
        logger.info(f"✅ Chatbot model loaded on {_chatbot_device}.")
    except Exception as e:
        logger.error(f"❌ Failed to load chatbot model: {e}")
        chatbot_model = None
        chatbot_tokenizer = None
        _chatbot_device = None

def load_facial_emotion_model():
    global facial_emotion_model
    if facial_emotion_model is not None:
        return
    try:
        logger.info("🌀 Loading facial emotion model (lazy)...")
        import joblib
        facial_emotion_model = joblib.load("emotion_classifier_rf_TUNED.joblib")
        logger.info("✅ Facial emotion model loaded.")
    except Exception as e:
        logger.error(f"❌ Failed to load facial emotion model: {e}")
        facial_emotion_model = None

# Optionally: start background threads to pre-load models (non-blocking)
def preload_models_in_background():
    # Only pre-load models if you actually want them warmed up
    threading.Thread(target=load_stress_model, daemon=True).start()
    threading.Thread(target=load_chatbot_model, daemon=True).start()
    threading.Thread(target=load_facial_emotion_model, daemon=True).start()

# If you want eager background loading uncomment the next line
# preload_models_in_background()

# -------------------------
# Your routes (unchanged logic, calling loaders lazily)
# -------------------------
@app.route("/")
def home():
    return "MindCare backend is running 🚀"

@app.route("/api/auth/register", methods=["POST"])
def register():
    if not ensure_db_connected():
        return jsonify({"msg": "Database unavailable"}), 503

    data = request.get_json()
    name = data.get('name'); email = data.get('email'); password = data.get('password')
    if not name or not email or not password:
        return jsonify({"msg": "Missing name, email, or password"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"msg": "Email already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_id = users_collection.insert_one({
        "name": name, "email": email, "password": hashed_password, "created_at": datetime.utcnow()
    }).inserted_id

    access_token = create_access_token(identity=str(user_id))
    logger.info(f"New user registered: {email}")
    return jsonify(access_token=access_token, user={"id": str(user_id), "name": name, "email": email}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    if not ensure_db_connected():
        return jsonify({"msg": "Database unavailable"}), 503

    data = request.get_json()
    email = data.get('email'); password = data.get('password')
    if not email or not password:
        return jsonify({"msg": "Missing email or password"}), 400

    user = users_collection.find_one({"email": email})
    if user and bcrypt.check_password_hash(user['password'], password):
        access_token = create_access_token(identity=str(user['_id']))
        logger.info(f"User logged in: {email}")
        return jsonify(access_token=access_token, user={"id": str(user['_id']), "name": user['name'], "email": user['email']})
    return jsonify({"msg": "Invalid credentials"}), 401

@app.route("/api/predict-stress", methods=["POST"])
@jwt_required()
def predict_stress_route():
    # Lazy-load model on demand
    load_stress_model()
    if not stress_model or not stress_scaler:
        return jsonify({"error": "Stress model unavailable"}), 503

    if not ensure_db_connected():
        return jsonify({"error": "Database unavailable"}), 503

    current_user_id = get_jwt_identity()
    data = request.json
    features = np.array([[float(data["heart_rate"]), float(data["steps"]), float(data["sleep"]), float(data["age"])]])
    scaled_features = stress_scaler.transform(features)
    prediction = stress_model.predict(scaled_features)
    stress_level = int(np.round(prediction[0][0])); stress_level = max(0, min(10, stress_level))

    stress_logs_collection.insert_one({
        "user_id": ObjectId(current_user_id),
        "stress_level": stress_level,
        "inputs": data,
        "timestamp": datetime.utcnow()
    })

    # categorize & suggestions (kept same)
    category = 'low'
    if 4 <= stress_level <= 6: category = 'medium'
    elif stress_level > 6: category = 'high'

    suggestions = [
        {"type": "Breathing", **suggestion_library[category]["Breathing"]},
        {"type": "Yoga", **suggestion_library[category]["Yoga"]},
        {"type": "Music", **suggestion_library[category]["Music"]},
    ]
    return jsonify({"stress_level": stress_level, "suggestions": suggestions})

@app.route("/api/chat", methods=["POST"])
@jwt_required()
def chat_route():
    load_chatbot_model()
    if not chatbot_model or not chatbot_tokenizer:
        return jsonify({"error": "Chatbot model unavailable"}), 503
    if not ensure_db_connected():
        return jsonify({"error": "Database unavailable"}), 503

    current_user_id = get_jwt_identity()
    data = request.json
    message_text = data["message"]

    # use local device variable
    from torch import no_grad, softmax
    inputs = chatbot_tokenizer(message_text, return_tensors="pt", truncation=True, padding=True).to(_chatbot_device)
    with no_grad():
        outputs = chatbot_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class_id = int(np.argmax(probabilities))
    emotion = {0: "positive", 1: "negative"}.get(predicted_class_id, "unknown")
    response_text = random.choice(coping_strategies.get(emotion, ["I'm here to listen. Tell me more."]))

    chat_logs_collection.insert_one({
        "user_id": ObjectId(current_user_id), "user_message": message_text, "ai_response": response_text,
        "detected_emotion": emotion, "timestamp": datetime.utcnow()
    })
    return jsonify({"response": response_text})

@app.route("/api/predict-emotion", methods=["POST"])
@jwt_required()
def predict_emotion_route():
    load_facial_emotion_model()
    if not facial_emotion_model:
        return jsonify({"error": "Facial emotion model unavailable"}), 503
    data = request.json
    feature_vector = np.array([[data['avg_ear'], data['mar'], data['eyebrow_dist'], data['jaw_drop']]])
    predicted_class = facial_emotion_model.predict(feature_vector)[0]
    emotion = facial_emotion_labels[predicted_class]
    prediction_proba = facial_emotion_model.predict_proba(feature_vector)[0]
    confidence = round(max(prediction_proba) * 100, 2)
    return jsonify({"emotion": emotion, "confidence": confidence})

# ... (keep your other routes like therapists, history, resources, using the same lazy helpers)

if __name__ == "__main__":
    # Local dev only. On Render use: gunicorn app:app --bind 0.0.0.0:$PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
