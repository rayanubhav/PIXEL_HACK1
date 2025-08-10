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

# ===================================================================================
# --- MODEL LOADING ---
# ===================================================================================

# --- 1. Load Stress Prediction Model & Scaler ---
try:
    stress_model = tf.keras.models.load_model("stress_model.h5")
    stress_scaler = joblib.load("scaler.pkl")
    logger.info("✅ Stress prediction model and scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading stress model or scaler: {e}")
    stress_model = None
    stress_scaler = None

# --- 2. Load Chatbot Emotion Model & Tokenizer ---
try:
    # Ensure you have the fine-tuned model files in a directory named 'fine_tuned_model'
    chatbot_model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
    chatbot_tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chatbot_model.to(device)
    chatbot_model.eval()
    logger.info(f"✅ Chatbot model loaded successfully on {device}.")
except Exception as e:
    logger.error(f"❌ Error loading chatbot model: {e}")
    chatbot_model = None
    chatbot_tokenizer = None
    
# --- 3. Load Facial Emotion Detection Model ---
try:
    emotion_model_filename = 'emotion_classifier_rf_TUNED.joblib'
    facial_emotion_model = joblib.load(emotion_model_filename)
    # This dictionary maps the model's integer output to a human-readable emotion
    facial_emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    logger.info(f"✅ Facial emotion detection model loaded from: {emotion_model_filename}")
except Exception as e:
    logger.error(f"❌ Error loading facial emotion model: {e}")
    facial_emotion_model = None


# --- MongoDB Setup for Chat Logs ---
try:
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
    current_user_id = get_jwt_identity()
    if not stress_model or not stress_scaler:
        return jsonify({"error": "Stress model is not available."}), 500

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
    
    # --- UPDATED RESPONSE ---
    category = 'low'
    if 4 <= stress_level <= 6: category = 'medium'
    elif stress_level > 6: category = 'high'

    suggestions = [
        {"type": "Breathing", **suggestion_library[category]["Breathing"]},
        {"type": "Yoga", **suggestion_library[category]["Yoga"]},
        {"type": "Music", **suggestion_library[category]["Music"]},
    ]

    logger.info(f"Stress prediction for user {current_user_id} saved. Level: {stress_level}")
    return jsonify({
        "stress_level": stress_level,
        "suggestions": suggestions
    })


@app.route("/api/chat", methods=["POST"])
@jwt_required()
def chat_route():
    current_user_id = get_jwt_identity()
    if not chatbot_model or not chatbot_tokenizer:
        return jsonify({"error": "Chatbot model not available."}), 500
    
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
    if not facial_emotion_model:
        return jsonify({"error": "Facial emotion model is not available. Check server logs."}), 500
        
    try:
        data = request.json
        # The frontend will calculate these features and send them in the request body
        feature_vector = np.array([[
            data['avg_ear'],
            data['mar'],
            data['eyebrow_dist'],
            data['jaw_drop']
        ]])

        predicted_class = facial_emotion_model.predict(feature_vector)[0]
        emotion = facial_emotion_labels[predicted_class]
        
        prediction_proba = facial_emotion_model.predict_proba(feature_vector)[0]
        confidence = round(max(prediction_proba) * 100, 2)

        logger.info(f"Facial emotion prediction: {emotion} ({confidence}%)")
        return jsonify({"emotion": emotion, "confidence": confidence})

    except Exception as e:
        logger.error(f"Error in /api/predict-emotion: {e}")
        return jsonify({"error": "An error occurred during emotion prediction."}), 400

# ===================================================================================
# --- NEW: THERAPIST FINDER ROUTE ---
# ===================================================================================

@app.route("/api/therapists", methods=["GET"])
@jwt_required()
def find_therapists_route():
    lat = request.args.get("lat")
    lng = request.args.get("lng")
    query = request.args.get("query", "mental health therapist")

    if not lat or not lng:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    fallback_locations = [
        {"name": "Mumbai", "lat": 19.076, "lng": 72.8777},
        {"name": "Delhi", "lat": 28.6139, "lng": 77.209},
        {"name": "Bangalore", "lat": 12.9716, "lng": 77.5946},
    ]

    def fetch_therapists(fs_lat, fs_lng, fs_query):
        api_url = "https://places-api.foursquare.com/places/search"
        api_key = os.environ.get("FOURSQUARE_SERVICE_KEY")

        # Debug: print API key status (mask the real value)
        print(f"🔑 Using API key: {'SET' if api_key else 'NOT SET'}")

        headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Accept": "application/json",
            "X-Places-API-Version": "2025-06-17",
        }
        params = {
            "ll": f"{fs_lat},{fs_lng}",
            "query": fs_query,
            "radius": 10000,
            "limit": 20,
        }

        # Debug: print the request details
        print(f"🌍 Requesting: {api_url}")
        print(f"📍 Params: {params}")

        try:
            response = requests.get(api_url, headers=headers, params=params)
            print(f"HTTP Status: {response.status_code}")
            print(f"Raw Response: {response.text[:500]}...")  # Limit length

            response.raise_for_status()
            data = response.json()

            results = []
            for place in data.get("results", []):
                lat_val = (
                    place.get("geocodes", {}).get("main", {}).get("latitude")
                    or place.get("latitude")
                )
                lng_val = (
                    place.get("geocodes", {}).get("main", {}).get("longitude")
                    or place.get("longitude")
                )

                if not lat_val or not lng_val:
                    continue

                results.append({
                    "id": place.get("fsq_id") or place.get("fsq_place_id"),
                    "name": place.get("name"),
                    "address": ", ".join(
                        filter(
                            None,
                            [
                                place.get("location", {}).get("address"),
                                place.get("location", {}).get("locality"),
                                place.get("location", {}).get("region"),
                            ],
                        )
                    ) or "Address not available",
                    "latitude": lat_val,
                    "longitude": lng_val,
                    "phone": place.get("tel"),
                })

            return results

        except requests.exceptions.RequestException as e:
            print(f"❌ API Request failed: {e}")
            return []

    # Try user location
    results = fetch_therapists(lat, lng, query)

    # If no results, try fallback cities
    if not results:
        print("⚠️ No results at user location, trying fallback cities...")
        for loc in fallback_locations:
            results = fetch_therapists(loc["lat"], loc["lng"], query)
            if results:
                print(f"✅ Fallback location used: {loc['name']}")
                break

    return jsonify(results)

# ===================================================================================
# --- NEW: DASHBOARD HISTORY ROUTE ---
# ===================================================================================

@app.route("/api/history", methods=["GET"])
@jwt_required()
def get_history():
    current_user_id = get_jwt_identity()
    try:
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        
        stress_logs_cursor = stress_logs_collection.find({
            "user_id": ObjectId(current_user_id),
            "timestamp": {"$gte": seven_days_ago}
        }).sort("timestamp", 1)

        stress_logs = list(stress_logs_cursor)

        # --- NEW: Aggregate data for Pie Chart ---
        stress_summary = {"Low": 0, "Medium": 0, "High": 0}
        for log in stress_logs:
            level = log['stress_level']
            if level <= 3:
                stress_summary["Low"] += 1
            elif 4 <= level <= 6:
                stress_summary["Medium"] += 1
            else:
                stress_summary["High"] += 1
        
        pie_chart_data = [{"name": key, "value": value} for key, value in stress_summary.items()]

        # (Existing history logic remains)
        dates_last_7_days = [(seven_days_ago + timedelta(days=i)).strftime("%b %d") for i in range(8)]
        stress_map = {date: None for date in dates_last_7_days}
        for log in stress_logs:
            date_str = log['timestamp'].strftime("%b %d")
            stress_map[date_str] = log['stress_level']
        stress_history = [{"date": date, "level": level} for date, level in stress_map.items()]

        last_stress_score = stress_logs[-1]['stress_level'] if stress_logs else "N/A"
        # --- 3. Get Last Chat Insight ---
        last_chat_log = chat_logs_collection.find_one(
            {"user_id": ObjectId(current_user_id)},
            sort=[("timestamp", -1)]
        )
        last_chat_insight = last_chat_log['ai_response'] if last_chat_log else "No recent chats."

        # --- 4. Combine and Return Data ---
      
        dashboard_data = {
            "stress_history": stress_history,
            "stress_summary_pie": pie_chart_data, # NEW
            "last_stress_score": last_stress_score,
            "last_chat_insight": last_chat_insight
        }
        
        return jsonify(dashboard_data), 200
    except Exception as e:
        logger.error(f"Error fetching history for user {current_user_id}: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route("/api/resources", methods=["GET"])
@jwt_required()
def get_resources():
    # This data is hardcoded for easy management, but could be moved to a database.
    resources_data = [
        {
            "category": "Immediate Help & Helplines",
            "items": [
                {
                    "title": "iCall Psychosocial Helpline (India)",
                    "description": "Free telephone and email-based counseling services provided by trained mental health professionals.",
                    "link": "https://icallhelpline.org/"
                },
                {
                    "title": "Samaritans Mumbai (India)",
                    "description": "Provides emotional support to anyone in distress, struggling to cope, or at risk of suicide.",
                    "link": "http://www.samaritansmumbai.com/"
                },
                {
                    "title": "National Suicide Prevention Lifeline (US)",
                    "description": "A national network of local crisis centers that provides free and confidential emotional support.",
                    "link": "https://suicidepreventionlifeline.org/"
                }
            ]
        },
        {
            "category": "Guided Meditations & Mindfulness",
            "items": [
                {
                    "title": "10-Minute Meditation for Beginners",
                    "description": "A simple, guided meditation to help you start your mindfulness practice.",
                    "link": "https://www.youtube.com/watch?v=O-6f5wQXSu8"
                },
                {
                    "title": "Mindful Breathing Exercise",
                    "description": "A short exercise focusing on the breath to calm anxiety and center your thoughts.",
                    "link": "https://youtu.be/watch?v=r6Vynwn_q-U"
                }
            ]
        },
        {
            "category": "Understanding Anxiety",
            "items": [
                {
                    "title": "What Is Anxiety?",
                    "description": "An informative article from the American Psychiatric Association explaining anxiety disorders.",
                    "link": "https://www.psychiatry.org/patients-families/anxiety-disorders/what-are-anxiety-disorders"
                },
                {
                    "title": "How to Cope with Anxiety",
                    "description": "Practical tips and strategies for managing anxiety symptoms in your daily life from Mind UK.",
                    "link": "https://www.mind.org.uk/information-support/types-of-mental-health-problems/anxiety-and-panic-attacks/self-care/"
                }
            ]
        }
    ]
    return jsonify(resources_data), 200

if __name__ == "__main__":
    # Use 0.0.0.0 to make it accessible on your local network, and set port to 5001
    app.run(host='0.0.0.0', port=5001, debug=True)

