import os
import random
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
    JWTManager,
)

import numpy as np
import requests
from dotenv import load_dotenv


load_dotenv()

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "changeme")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mindcare")

# --- Globals for models and DB ---
stress_model = None
stress_scaler = None
chatbot_model = None
chatbot_tokenizer = None
_chatbot_device = None
facial_emotion_model = None
facial_emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# Mongo globals
_mongo_client = None
db = None
users_collection = None
stress_logs_collection = None
chat_logs_collection = None

# ------------------------------------------------------------------
# Small in-memory defaults to avoid NameErrors in routes. Replace with real
# content or load from DB/config as needed.
# ------------------------------------------------------------------
text_emotion_label_map = {0: "positive", 1: "negative"}
coping_strategies = {
    "positive": [
        "I'm so glad you're feeling good today! What's got you in such a great mood?",
        "That's awesome to hear! Keep shining, and maybe try journaling to capture this positivity.",
        "You're doing great! How about celebrating with a small self-care activity, like a walk?",
        "Love the positive vibes! What's something fun you're looking forward to?",
        "It's amazing to see you thriving! Maybe share this energy with a friend today.",
        "Fantastic to hear! Try setting a small goal to keep this momentum going.",
        "You sound so upbeat! How about some music to keep the good vibes flowing?",
        "Today's energy is infectious! How about spreading some kindness to someone else?",
        "Maybe snap a picture of something beautiful you notice today to remember this feeling.",
        "Consider making a gratitude list—what's one thing you're especially thankful for right now?",
        "Share your good news with someone close. Joy is even sweeter when it's shared.",
        "Want to try a creative hobby while your motivation is high—like doodling or baking?",
        "Your positivity might inspire someone else today. Who do you want to cheer up?",
        "You deserve to celebrate your wins, big or small! What's one thing you're proud of?",
        "Keep this streak going—set a fun goal for tomorrow!",
        "If you feel extra energized, maybe try organizing or cleaning a small space."
    ],
    "stressed": [
        "It sounds really tough right now. Try deep breathing: inhale for 4, hold for 4, exhale for 4.",
        "Stress can be overwhelming. How about a quick 5-minute stretch to ease some tension?",
        "I hear you're stressed. Writing down what's on your mind might help organize your thoughts.",
        "That sounds heavy. Maybe step away for a moment and sip some water to reset.",
        "Stress is hard. Try grounding yourself: name 5 things you see, 4 you feel, 3 you hear.",
        "You're carrying a lot. A short mindfulness exercise could help calm things down.",
        "I'm here with you. How about prioritizing one small task to feel more in control?",
        "Try a 'brain dump': jot down every worry on paper, no matter how small.",
        "Play a calming song and just listen for a few minutes.",
        "Check in with your body: are you hungry, tired, or thirsty? Simple self-care matters.",
        "Visualize a relaxing place—even for 60 seconds.",
        "Try massaging your temples or your hands for a mini self-soothe break.",
        "Step outside if you can, and take a few breaths of fresh air.",
        "Say something kind to yourself. Self-talk can make a difference.",
        "Stretch your arms above your head—big movements can help release tension.",
        "Take a short walk around your home or office. Changing locations can shift perspective."
    ],
    "anxious": [
        "Anxiety can feel so intense. Try slow, deep breaths to calm your mind a bit.",
        "I'm here for you. Naming your worries out loud or on paper might make them feel smaller.",
        "That sounds really unsettling. A quick body scan meditation could help you feel grounded.",
        "Anxiety is tough. Try focusing on one thing you can control right now, like tidying a space.",
        "I hear how worried you are. Maybe a soothing tea and a quiet moment could help?",
        "You're not alone in this. Try the 3-3-3 rule: name 3 things you see, hear, and touch.",
        "It's okay to feel anxious. How about a short walk to shift your focus?",
        "Muscle tension is common with anxiety—try clenching then releasing your fists a few times.",
        "Close your eyes and picture a safe place for one minute.",
        "Write yourself a reassuring note; keep it for anxious moments.",
        "Try holding an ice cube for a minute. Cold stimulus can break anxious spirals.",
        "Remind yourself: feelings are not permanent—they will pass.",
        "Count backwards from 100 by sevens to interrupt anxious thought loops.",
        "List your strengths or past successes. Remind yourself you've coped before.",
        "Reach out to someone—a friend, family member, or support line.",
        "Distract yourself with a small, engaging task: puzzle, coloring, or sorting something nearby."
    ],
    "sad": [
        "I'm so sorry you're feeling down. It's okay to feel this way sometimes—want to share more?",
        "Sadness can be heavy. Maybe listen to a favorite song or watch a comforting show?",
        "I hear you're feeling low. A warm drink or cozy blanket might bring a little comfort.",
        "It's tough to feel sad. Writing a letter to yourself about what you're grateful for could help.",
        "You're not alone in this. Try reaching out to a friend or loved one for a chat.",
        "I'm here with you. Maybe a short cry or journaling could let some of those feelings out?",
        "Feeling sad is hard. How about a small act of kindness for yourself, like a treat?",
        "Draw or doodle your feelings—art can be a gentle way to process sadness.",
        "Step outside and feel the sun or breeze, even for a moment.",
        "Try moving your body—slow stretching or gentle movement counts.",
        "Cuddle up with a pet, pillow, or stuffed animal for a little comfort.",
        "Let yourself cry if you need—you don't have to hold it in.",
        "Write down 3 things you wish you could tell someone, even if you don't send them.",
        "Allow yourself to rest with no guilt; self-kindness is key.",
        "Reread a favorite book, story, or poem for familiar comfort.",
        "Light a scented candle or focus on a calming sensory detail."
    ],
    "angry": [
        "Anger is valid, but it can feel overwhelming. Try taking 10 deep breaths to cool off.",
        "Write out what's making you angry—sometimes seeing it on paper helps.",
        "Channel your anger into movement: jumping jacks, dancing, or a fast walk.",
        "If you can, talk it out with a trusted person.",
        "Try squeezing a stress ball or tearing a piece of paper for safe release.",
        "Splash your face with cold water or hold something cool.",
        "Put on intense music and let yourself feel your emotions for a few moments.",
        "Remind yourself, 'This feeling will pass. I can choose my next step.'",
        "Step outside and name everything you see in your environment.",
        "Draw or color with bold strokes—externalizing can help process anger.",
        "Count to 100 slowly, then check in with yourself again.",
        "Try writing a letter you never send to the person or situation."
    ],
    "lonely": [
        "Loneliness can be tough. Reach out with a quick text or call to someone you trust.",
        "Try a virtual hangout or join an online community around a shared interest.",
        "Write yourself a letter as if from a friend—what would you want to hear?",
        "Sometimes small public interactions—a wave to a neighbor or chatting with a store clerk—help.",
        "Listen to a podcast or audiobook while you do something comforting.",
        "Sit with a pet or stuffed animal for a sense of connection.",
        "Do something for someone else—a small act of kindness can lighten loneliness."
    ],
    "overwhelmed": [
        "It can help to break tasks into tiny steps—what's the next right thing?",
        "Write down everything on your mind, then cross off anything that isn't urgent.",
        "Set a timer for five minutes, and do just one thing.",
        "Ask for help if you can—delegating lightens the load.",
        "Remind yourself: Perfection isn't required—even a small effort counts.",
        "Take a five-minute sensory break—focus on touch, sight, sound, taste, and smell."
    ],
    "hopeless": [
        "When everything feels hopeless, remember: this feeling is temporary.",
        "Reach out to a helpline or someone you trust—you're not alone.",
        "Focus on surviving the next hour, not fixing everything at once.",
        "What's one small comfort you can offer yourself right now?",
        "Try grounding exercises: feeling feet on the floor, hands on your lap.",
        "Remind yourself: You've survived hard things before."
    ]
}

helplines = {
    "US": "1-800-273-8255 (National Suicide Prevention Lifeline)",
    "India": "9152987821 (iCall, India), +91-22-25521111 (Samaritans Mumbai)",
    "Global": "Find local helplines at www.iasp.info/resources/Crisis_Centres/"
}

suggestion_library = {
    "low": {
        "Breathing": {"title": "Mindful Sigh", "description": "Inhale deeply through your nose and exhale with an audible sigh. A simple way to release tension and reset.", "link": "https://youtu.be/watch?v=r6Vynwn_q-U"},
        "Yoga": {"title": "Cat-Cow Stretch", "description": "A gentle, accessible stretch to increase spinal flexibility and calm the mind. Great for any time of day.", "link": "https://youtu.be/watch?v=LIVJZZyZ2qM"},
        "Music": {"title": "Lofi Hip Hop Radio", "description": "Relaxing beats perfect for studying, relaxing, or focusing without distraction.", "link": "https://youtu.be/watch?v=lTRiuFIWV54"}
    },
    "medium": {
        "Breathing": {"title": "Box Breathing", "description": "Inhale for 4s, hold for 4s, exhale for 4s, hold for 4s. A powerful technique to calm the nervous system.", "link": "https://www.youtube.com/watch?v=tEmt1Znux58"},
        "Yoga": {"title": "Child's Pose", "description": "A resting pose that can help relieve stress and fatigue. It gently stretches your back, hips, and ankles.", "link": "https://youtu.be/watch?v=kH12QrSGedM"},
        "Music": {"title": "Calm Piano Music", "description": "Beautiful, light piano music that can help reduce anxiety and promote a sense of peace.", "link": "https://youtu.be/watch?v=5OIeIaAhQOg"}
    },
    "high": {
        "Breathing": {"title": "4-7-8 Breathing", "description": "Inhale for 4s, hold your breath for 7s, and exhale slowly for 8s. Excellent for reducing anxiety quickly.", "link": "https://youtu.be/watch?v=LiUnFJ8P4gM"},
        "Yoga": {"title": "Legs-Up-The-Wall Pose", "description": "A restorative pose that helps calm the nervous system and reduce stress and anxiety.", "link": "https://youtu.be/watch?v=do_1LisFah0"},
        "Music": {"title": "Weightless by Marconi Union", "description": "Specifically designed in collaboration with sound therapists to reduce anxiety, heart rate, and blood pressure.", "link": "https://youtu.be/watch?v=UfcAVejslrU"}
    }
}

# ------------------------------------------------------------------
# MongoDB initialization & helpers
# ------------------------------------------------------------------

def init_mongo() -> bool:
    """Attempt to initialise a MongoDB connection using MONGO_URI.
    Returns True if connection and collections are available.
    """
    global _mongo_client, db, users_collection, stress_logs_collection, chat_logs_collection

    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        logger.warning("MONGO_URI not set; DB features will be unavailable.")
        return False

    try:
        from pymongo import MongoClient

        _mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # trigger connection check
        _mongo_client.server_info()

        db = _mongo_client.get_database("mindcare_db")
        users_collection = db.get_collection("users")
        stress_logs_collection = db.get_collection("stress_logs")
        chat_logs_collection = db.get_collection("chat_logs")

        logger.info("✅ MongoDB connection established.")
        return True
    except Exception as e:
        logger.error(f"❌ Error connecting to MongoDB: {e}")
        _mongo_client = None
        db = None
        users_collection = None
        stress_logs_collection = None
        chat_logs_collection = None
        return False


def ensure_db_connected() -> bool:
    """Utility used inside routes to verify DB availability.
    Avoids throwing at import time.
    """
    global _mongo_client
    if not _mongo_client:
        return init_mongo()

    try:
        # quick server check
        _mongo_client.server_info()
        return True
    except Exception:
        return False


# Try to initialize at startup (optional)
init_mongo()

# ------------------------------------------------------------------
# Lazy model loaders (heavy libraries imported inside functions)
# ------------------------------------------------------------------

def load_stress_model():
    global stress_model, stress_scaler
    if stress_model is not None and stress_scaler is not None:
        return

    try:
        logger.info("🌀 Loading stress model (lazy)...")
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
    if chatbot_model is not None and chatbot_tokenizer is not None:
        return

    try:
        logger.info("🌀 Loading chatbot model (lazy)...")
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


def preload_models_in_background():
    threading.Thread(target=load_stress_model, daemon=True).start()
    threading.Thread(target=load_chatbot_model, daemon=True).start()
    threading.Thread(target=load_facial_emotion_model, daemon=True).start()


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def home():
    return "MindCare backend is running 🚀"


# --- Auth: register/login ---
@app.route("/api/auth/register", methods=["POST"]) 
def register():
    if not ensure_db_connected():
        return jsonify({"msg": "Database unavailable"}), 503

    data = request.get_json(force=True)
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    if not name or not email or not password:
        return jsonify({"msg": "Missing name, email, or password"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"msg": "Email already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    inserted = users_collection.insert_one(
        {"name": name, "email": email, "password": hashed_password, "created_at": datetime.utcnow()}
    )
    user_id = str(inserted.inserted_id)

    access_token = create_access_token(identity=user_id)
    logger.info(f"New user registered: {email}")
    return jsonify(access_token=access_token, user={"id": user_id, "name": name, "email": email}), 201


@app.route("/api/auth/login", methods=["POST"]) 
def login():
    if not ensure_db_connected():
        return jsonify({"msg": "Database unavailable"}), 503

    data = request.get_json(force=True)
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"msg": "Missing email or password"}), 400

    user = users_collection.find_one({"email": email})
    if user and bcrypt.check_password_hash(user["password"], password):
        access_token = create_access_token(identity=str(user["_id"]))
        logger.info(f"User logged in: {email}")
        return jsonify(access_token=access_token, user={"id": str(user["_id"]), "name": user["name"], "email": user["email"]})

    return jsonify({"msg": "Invalid credentials"}), 401


# --- Stress prediction ---
@app.route("/api/predict-stress", methods=["POST"]) 
@jwt_required()
def predict_stress_route():
    current_user_id = get_jwt_identity()
    # Lazy-load model if needed
    if stress_model is None or stress_scaler is None:
        load_stress_model()

    if not stress_model or not stress_scaler:
        return jsonify({"error": "Stress model is not available."}), 500

    data = request.get_json(force=True)
    try:
        features = np.array([[float(data["heart_rate"]), float(data["steps"]), float(data["sleep"]), float(data["age"]) ]])
        scaled_features = stress_scaler.transform(features)
        prediction = stress_model.predict(scaled_features)
        stress_level = int(np.round(prediction[0][0]))
        stress_level = max(0, min(10, stress_level))

        if ensure_db_connected():
            stress_logs_collection.insert_one({
                "user_id": ObjectId(current_user_id),
                "stress_level": stress_level,
                "inputs": data,
                "timestamp": datetime.utcnow(),
            })

        # categorize
        category = "low"
        if 4 <= stress_level <= 6:
            category = "medium"
        elif stress_level > 6:
            category = "high"

        suggestions = [
            {"type": "Breathing", **suggestion_library[category]["Breathing"]},
            {"type": "Yoga", **suggestion_library[category]["Yoga"]},
            {"type": "Music", **suggestion_library[category]["Music"]},
        ]

        logger.info(f"Stress prediction for user {current_user_id} saved. Level: {stress_level}")
        return jsonify({"stress_level": stress_level, "suggestions": suggestions})

    except Exception as e:
        logger.error(f"Error in /api/predict-stress: {e}")
        return jsonify({"error": "Invalid input or internal error."}), 400


# --- Chat route ---
@app.route("/api/chat", methods=["POST"]) 
@jwt_required()
def chat_route():
    current_user_id = get_jwt_identity()
    if chatbot_model is None or chatbot_tokenizer is None:
        load_chatbot_model()

    if not chatbot_model or not chatbot_tokenizer:
        return jsonify({"error": "Chatbot model not available."}), 500

    data = request.get_json(force=True)
    message_text = data.get("message", "")

    try:
        # imports local to avoid heavy deps at import time
        import torch
        from torch import no_grad

        inputs = chatbot_tokenizer(message_text, return_tensors="pt", truncation=True, padding=True).to(_chatbot_device)
        with no_grad():
            outputs = chatbot_model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_id = int(np.argmax(probabilities))
        emotion = {0: "positive", 1: "negative"}.get(predicted_class_id, "unknown")
        response_text = random.choice(coping_strategies.get(emotion, ["I'm here to listen. Tell me more."]))

        if ensure_db_connected():
            chat_logs_collection.insert_one({
                "user_id": ObjectId(current_user_id),
                "user_message": message_text,
                "ai_response": response_text,
                "detected_emotion": emotion,
                "timestamp": datetime.utcnow(),
            })

        return jsonify({"response": response_text})

    except Exception as e:
        logger.error(f"Error in /api/chat: {e}")
        return jsonify({"error": "An error occurred while generating a response."}), 500


# --- Facial emotion prediction ---
@app.route("/api/predict-emotion", methods=["POST"]) 
@jwt_required()
def predict_emotion_route():
    if facial_emotion_model is None:
        load_facial_emotion_model()

    if not facial_emotion_model:
        return jsonify({"error": "Facial emotion model is not available. Check server logs."}), 500

    try:
        data = request.get_json(force=True)
        feature_vector = np.array([[data["avg_ear"], data["mar"], data["eyebrow_dist"], data["jaw_drop"]]])
        predicted_class = facial_emotion_model.predict(feature_vector)[0]
        emotion = facial_emotion_labels.get(predicted_class, "Unknown")
        prediction_proba = facial_emotion_model.predict_proba(feature_vector)[0]
        confidence = round(max(prediction_proba) * 100, 2)

        logger.info(f"Facial emotion prediction: {emotion} ({confidence}%)")
        return jsonify({"emotion": emotion, "confidence": confidence})

    except Exception as e:
        logger.error(f"Error in /api/predict-emotion: {e}")
        return jsonify({"error": "An error occurred during emotion prediction."}), 400


# --- Therapist finder (Foursquare Places API) ---
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

    def fetch_therapists(fs_lat, fs_lng, fs_query) -> List[Dict[str, Any]]:
        api_url = "https://places-api.foursquare.com/places/search"
        api_key = os.environ.get("FOURSQUARE_SERVICE_KEY")

        headers = {
            "Authorization": f"Bearer {api_key}" if api_key else "",
            "Accept": "application/json",
            "X-Places-API-Version": "2025-06-17",
        }
        params = {"ll": f"{fs_lat},{fs_lng}", "query": fs_query, "radius": 10000, "limit": 20}

        try:
            response = requests.get(api_url, headers=headers, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()

            results = []
            for place in data.get("results", []):
                lat_val = place.get("geocodes", {}).get("main", {}).get("latitude") or place.get("latitude")
                lng_val = place.get("geocodes", {}).get("main", {}).get("longitude") or place.get("longitude")
                if not lat_val or not lng_val:
                    continue

                results.append({
                    "id": place.get("fsq_id") or place.get("fsq_place_id"),
                    "name": place.get("name"),
                    "address": ", ".join(filter(None, [
                        place.get("location", {}).get("address"),
                        place.get("location", {}).get("locality"),
                        place.get("location", {}).get("region"),
                    ])) or "Address not available",
                    "latitude": lat_val,
                    "longitude": lng_val,
                    "phone": place.get("tel"),
                })

            return results

        except requests.exceptions.RequestException as e:
            logger.warning(f"❌ Foursquare request failed: {e}")
            return []

    results = fetch_therapists(lat, lng, query)
    if not results:
        for loc in fallback_locations:
            results = fetch_therapists(loc["lat"], loc["lng"], query)
            if results:
                logger.info(f"✅ Fallback location used: {loc['name']}")
                break

    return jsonify(results)


# --- Dashboard/history ---
@app.route("/api/history", methods=["GET"]) 
@jwt_required()
def get_history():
    current_user_id = get_jwt_identity()
    try:
        seven_days_ago = datetime.utcnow() - timedelta(days=7)

        if not ensure_db_connected():
            return jsonify({"error": "Database unavailable"}), 503

        stress_logs_cursor = stress_logs_collection.find({
            "user_id": ObjectId(current_user_id),
            "timestamp": {"$gte": seven_days_ago},
        }).sort("timestamp", 1)

        stress_logs = list(stress_logs_cursor)

        stress_summary = {"Low": 0, "Medium": 0, "High": 0}
        for log in stress_logs:
            level = log.get("stress_level", 0)
            if level <= 3:
                stress_summary["Low"] += 1
            elif 4 <= level <= 6:
                stress_summary["Medium"] += 1
            else:
                stress_summary["High"] += 1

        pie_chart_data = [{"name": k, "value": v} for k, v in stress_summary.items()]

        dates_last_7_days = [(seven_days_ago + timedelta(days=i)).strftime("%b %d") for i in range(8)]
        stress_map = {date: None for date in dates_last_7_days}
        for log in stress_logs:
            date_str = log.get("timestamp").strftime("%b %d")
            stress_map[date_str] = log.get("stress_level")

        stress_history = [{"date": d, "level": l} for d, l in stress_map.items()]

        last_stress_score = stress_logs[-1]["stress_level"] if stress_logs else "N/A"
        last_chat_log = chat_logs_collection.find_one({"user_id": ObjectId(current_user_id)}, sort=[("timestamp", -1)])
        last_chat_insight = last_chat_log["ai_response"] if last_chat_log else "No recent chats."

        dashboard_data = {
            "stress_history": stress_history,
            "stress_summary_pie": pie_chart_data,
            "last_stress_score": last_stress_score,
            "last_chat_insight": last_chat_insight,
        }

        return jsonify(dashboard_data), 200

    except Exception as e:
        logger.error(f"Error fetching history for user {current_user_id}: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


# --- Static resources ---
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
# ------------------------------------------------------------------
# CLI: run the app
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Optional: uncomment to warm models at startup
    # preload_models_in_background()

    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=(os.environ.get("FLASK_ENV") == "development"))
