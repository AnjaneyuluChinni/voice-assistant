from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import time
from datetime import datetime
import requests
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gtts import gTTS
import google.generativeai as genai
import base64
from io import BytesIO

app = Flask(__name__)

# --- Configuration ---
OPENWEATHERMAP_API_KEY = "7dc6e4149828f811e558584da35a264d"
WEATHER_BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"
CITY_NAME = "London"

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyB9u_DjHIUNRrMH0lLEhK4iSb0ZQRPcjyo"
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
gemini = genai.GenerativeModel("models/gemini-1.5-flash")

# Initialize the transformer model and tokenizer
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)

# Define intent labels
INTENT_LABELS = {
    0: "greeting",
    1: "get_time",
    2: "get_date",
    3: "get_weather",
    4: "search_wikipedia",
    5: "exit"
}

def predict_intent(text):
    """Predicts the intent of the user's text."""
    if not text:
        return "none"

    text_lower = text.lower()

    # First check for Wikipedia search intent as it's most specific
    search_phrases = [
        "explain", "tell me about", "who is", "what is", 
        "search for", "look up", "find information about",
        "can you explain", "can you tell me about", "about"
    ]
    
    if any(phrase in text_lower for phrase in search_phrases):
        return "search_wikipedia"
    
    if text_lower.endswith("?") or any(word in text_lower.split() for word in ["what", "who", "when", "where", "why", "how"]):
        return "search_wikipedia"

    if any(word in text_lower for word in ["weather", "temperature", "forecast", "climate"]):
        return "get_weather"
    elif any(word in text_lower for word in ["time", "clock"]):
        return "get_time"
    elif any(word in text_lower for word in ["date", "day", "today"]):
        return "get_date"
    elif any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
        return "greeting"
    elif any(word in text_lower for word in ["exit", "quit", "bye", "goodbye"]):
        return "exit"

    try:
        inputs = tokenizer(text_lower, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        return INTENT_LABELS[predicted_class]
    except Exception as e:
        print(f"Error in transformer model prediction: {e}")
        return "none"

def get_response(intent, text=""):
    """Generates a response based on the detected intent."""
    if intent == "greeting":
        return "Hello! How can I help you today?"
    elif intent == "get_time":
        now = datetime.now()
        current_time = now.strftime("%I:%M %p")
        return f"The current time is {current_time}"
    elif intent == "get_date":
        now = datetime.now()
        current_date = now.strftime("%B %d, %Y")
        return f"Today's date is {current_date}"
    elif intent == "get_weather":
        if OPENWEATHERMAP_API_KEY == "YOUR_API_KEY":
            return "Please add your OpenWeatherMap API key to the script to get weather information."
        try:
            city = CITY_NAME
            text_lower = text.lower()
            
            if "weather in" in text_lower:
                parts = text_lower.split("weather in", 1)
                if len(parts) > 1:
                    city = parts[1].strip().split()[0].capitalize()
            elif "temperature in" in text_lower:
                parts = text_lower.split("temperature in", 1)
                if len(parts) > 1:
                    city = parts[1].strip().split()[0].capitalize()
            elif "forecast in" in text_lower:
                parts = text_lower.split("forecast in", 1)
                if len(parts) > 1:
                    city = parts[1].strip().split()[0].capitalize()

            print(f"Fetching weather for: {city}")
            complete_url = WEATHER_BASE_URL + "appid=" + OPENWEATHERMAP_API_KEY + "&q=" + city
            response = requests.get(complete_url)
            data = response.json()

            if data["cod"] != "404":
                main_data = data["main"]
                current_temperature_k = main_data["temp"]
                current_temperature_c = round(current_temperature_k - 273.15, 1)
                weather_description = data["weather"][0]["description"]
                return f"The weather in {city} is {weather_description} with a temperature of {current_temperature_c} degrees Celsius."
            else:
                return f"City '{city}' not found."
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return "Sorry, I couldn't retrieve the weather information."
    elif intent == "search_wikipedia":
        query = text.lower()
        
        search_phrases = [
            "do you explain me about", "do you explain about", "explain me about",
            "explain about", "explain", "tell me about", "who is", "what is", 
            "search for", "look up", "find information about",
            "can you explain", "can you tell me about", "about"
        ]
        
        search_phrases.sort(key=len, reverse=True)
        
        for phrase in search_phrases:
            if phrase in query:
                query = query.replace(phrase, "", 1).strip()
                break
        
        query = query.rstrip('?')
        
        common_words = ["the", "a", "an", "in", "on", "at", "to", "for", "with", "by", 
                       "do", "you", "me", "can", "could", "would", "should", "will", "shall",
                       "please", "tell", "give", "show", "find", "search", "look", "get"]
        
        query_words = query.split()
        query = " ".join(word for word in query_words if word.lower() not in common_words)
        
        query = query.lstrip("about").rstrip("about").strip()
        
        print(f"Original query: {text}")
        print(f"Cleaned query: {query}")
        
        if not query:
            return "Sorry, I didn't catch what you want to know about."

        try:
            print(f"Searching for: {query}")
            prompt = f"Please provide a concise and informative response about {query}. Keep it to 2-3 sentences."
            response = gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"An error occurred while generating response: {e}")
            return "Sorry, I encountered an error while processing your request."

def text_to_speech(text):
    """Converts text to speech and returns base64 audio data."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        return audio_base64
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    text = data.get('text', '')
    
    intent = predict_intent(text)
    response = get_response(intent, text)
    audio_data = text_to_speech(response)
    
    return jsonify({
        'response': response,
        'audio': audio_data
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 
