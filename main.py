import cv2
import base64
import numpy as np
import requests
import json
import threading
import pyttsx3

engine = pyttsx3.init()

def speak(message):
    engine.say(message)
    engine.runAndWait()

url = "https://api.fireworks.ai/inference/v1/chat/completions"
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "YOUR_API_KEY"
}

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

cap = cv2.VideoCapture(0)

def camera_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

camera_thread = threading.Thread(target=camera_loop)
camera_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    jpg_as_text = frame_to_base64(frame)
    payload = {
        "model": "accounts/fireworks/models/firellava-13b",
        "max_tokens": 512,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.6,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Only say if im happy, neutral Or sad"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{jpg_as_text}"
                        }
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    try:
        response_json = response.json()
        content = response_json['choices'][0]['message']['content']
        print("Content:", content)
        if content.lower() == "sad.":
            speak("Stay Happy")
        elif content.lower() == "neutral.":
            speak("Neutral")
        elif content.lower() == "happy.":
            speak("You're Happy")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        print("Error parsing response:", e)
    cv2.imshow('Camera Feed', frame)

cap.release()
cv2.destroyAllWindows()
