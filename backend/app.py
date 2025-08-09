import os
import random
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity, JWTManager
import numpy as np
import tensorflow as tf
import joblib
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from pymongo import MongoClient
from bson import ObjectId
import torch
import logging
import requests
from dotenv import load_dotenv
load_dotenv()
# --- App Initialization ---
app = Flask(__name__)
CORS(app) # Allow requests from your React frontend

# --- Configuration ---
# It's crucial to set a secret key for JWT.
# In production, use a long, random string stored in an environment variable.
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
    client = MongoClient(MONGO_URI)
    db = client["mindcare_db"] # Using a new database name for clarity
    users_collection = db["users"]
    stress_logs_collection = db["stress_logs"]
    chat_logs_collection = db["chat_logs"] # You already had this
    logger.info("✅ MongoDB connection established.")
except Exception as e:
    logger.error(f"❌ Error connecting to MongoDB: {e}")
    users_collection = None

# --- Chatbot Helper Data ---
# This label map is for the text-based emotion detection in the chatbot
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


@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')

    if not name or not email or not password:
        return jsonify({"msg": "Missing name, email, or password"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"msg": "Email already exists"}), 409

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    user_id = users_collection.insert_one({
        "name": name,
        "email": email,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    }).inserted_id

    access_token = create_access_token(identity=str(user_id))
    logger.info(f"New user registered: {email}")
    return jsonify(access_token=access_token, user={"id": str(user_id), "name": name, "email": email}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"msg": "Missing email or password"}), 400

    user = users_collection.find_one({"email": email})

    if user and bcrypt.check_password_hash(user['password'], password):
        access_token = create_access_token(identity=str(user['_id']))
        logger.info(f"User logged in: {email}")
        return jsonify(access_token=access_token, user={"id": str(user['_id']), "name": user['name'], "email": user['email']})
    
    return jsonify({"msg": "Invalid credentials"}), 401

# ===================================================================================
# --- API ROUTES ---
# ===================================================================================

@app.route("/api/predict-stress", methods=["POST"])
@jwt_required()
def predict_stress_route():
    current_user_id = get_jwt_identity()
    if not stress_model or not stress_scaler:
        return jsonify({"error": "Stress model is not available."}), 500

    data = request.json
    features = np.array([[float(data["heart_rate"]), float(data["steps"]), float(data["sleep"]), float(data["age"])]])
    scaled_features = stress_scaler.transform(features)
    prediction = stress_model.predict(scaled_features)
    stress_level = int(np.round(prediction[0][0]))
    stress_level = max(0, min(10, stress_level))

    stress_logs_collection.insert_one({
        "user_id": ObjectId(current_user_id),
        "stress_level": stress_level,
        "inputs": data,
        "timestamp": datetime.utcnow()
    })
    
    # --- UPDATED RESPONSE ---
    category = 'low'
    if 4 <= stress_level <= 6:
        category = 'medium'
    elif stress_level > 6:
        category = 'high'
        
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
    
    inputs = chatbot_tokenizer(message_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = chatbot_model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
    predicted_class_id = np.argmax(probabilities)
    emotion = {0: "positive", 1: "negative"}.get(predicted_class_id, "unknown")
    response_text = random.choice({ "positive": ["That's wonderful to hear!"], "negative": ["I'm sorry to hear that."] }.get(emotion, ["I'm here to listen."]))

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

