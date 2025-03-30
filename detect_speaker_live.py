from flask import Flask, Response, jsonify
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

app = Flask(__name__)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes (Speaker 1 vs. Speaker 2)
model.eval()

# OpenCV setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_speaker(frame):
    """Detect faces and determine the active speaker."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    max_confidence = -1
    speaker_status = "No speaker detected"
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        input_face = cv2.resize(face, (224, 224))
        input_face = input_face.astype(np.float32) / 255.0
        input_face = input_face.transpose((2, 0, 1))
        input_tensor = torch.tensor(input_face).float().unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
            confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

        if confidence > max_confidence:
            max_confidence = confidence
            speaker_status = f"Speaker {predicted_class.item() + 1} ({confidence:.2f})"

    return speaker_status

@app.route('/video_feed')
def video_feed():
    """Stream video frames to the frontend."""
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/speaker_status')
def speaker_status():
    """Provide real-time speaker detection status."""
    ret, frame = cap.read()
    if not ret:
        return jsonify({"status": "Camera error"})
    
    status = detect_speaker(frame)
    return jsonify({"status": status})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)