# Voice Assistant Project

## Description
This project is a voice assistant built using Flask, which allows users to interact with it through text input. It can perform various tasks such as greeting users, providing the current time and date, fetching weather information, and searching for information on Wikipedia

## Features
- Greeting users
- Providing the current time and date
- Fetching weather information for a specified city
- Searching for information on Wikipedia
- Text-to-speech conversion for responses

## Technologies Used
- **Flask**: Web framework for building the application
- **Speech Recognition**: For processing voice input
- **OpenWeatherMap API**: For fetching weather data
- **Google Generative AI (Gemini)**: For generating responses to queries
- **gTTS (Google Text-to-Speech)**: For converting text responses to speech
- **PyTorch & Transformers**: For intent prediction using a pre-trained model

## Setup Instructions
1. **Clone the Repository**: 
   ```bash
   git clone <repository-url>
   cd Voice-Assistant
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Keys**:
   - Obtain an API key from [OpenWeatherMap](https://openweathermap.org/) and replace `YOUR_API_KEY` in `app.py`.
   - Configure the Gemini API key in `app.py`.

4. **Run the Application**:
   ```bash
   python app.py
   ```

5. **Access the Application**:
   Open your web browser and go to `http://127.0.0.1:5000/`.

## Usage
- Enter your query in the text input field on the web interface.
- The assistant will process your input and provide a response, which can be heard as speech. 
