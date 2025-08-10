import os
import random
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import numpy as np

load_dotenv()
app = Flask(__name__)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
chatbot_model = None
chatbot_tokenizer = None
device = None
try:
    chatbot_model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_model")
    chatbot_tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chatbot_model.to(device)
    chatbot_model.eval()
    logger.info(f"✅ Chatbot model loaded on {device}.")
except Exception as e:
    logger.error(f"❌ Error loading chatbot model: {e}")

# --- Security ---
INTERNAL_SERVICE_KEY = os.environ.get("INTERNAL_SERVICE_KEY")

# --- Helper Data ---
text_emotion_label_map = {0: "positive", 1: "negative"} 
coping_strategies = {
    "positive": ["That's wonderful to hear! What's been the best part of your day?", "I'm so glad you're feeling positive. Keep that energy going!"],
    "negative": ["I'm sorry to hear that. It's okay to feel this way. Would you like to talk more about it?", "That sounds really tough. Remember to be kind to yourself."]
}

@app.route("/chat", methods=["POST"])
def chat_route():
    # --- Security Check ---
    if request.headers.get("X-Internal-Service-Key") != INTERNAL_SERVICE_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    if not chatbot_model or not chatbot_tokenizer:
        return jsonify({"error": "Chatbot model is not available."}), 503

    try:
        data = request.json
        message_text = data["message"]
        
        inputs = chatbot_tokenizer(message_text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = chatbot_model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class_id = np.argmax(probabilities)
        
        emotion = text_emotion_label_map.get(predicted_class_id, "unknown")
        confidence = float(probabilities[predicted_class_id])
        
        response_text = random.choice(coping_strategies.get(emotion, ["I'm here to listen. Tell me more."]))

        return jsonify({
            "response": response_text,
            "emotion": emotion,
            "confidence": confidence
        })
    except Exception as e:
        logger.error(f"Error in /chat: {e}")
        return jsonify({"error": "An error occurred in the chat service."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port)
