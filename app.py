import os
import time
from flask import Flask, request, jsonify, send_file
import torch
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from torchvision import models, transforms

app = Flask(__name__)

# Ensure necessary directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('processed_videos', exist_ok=True)

# Initialize the pre-trained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# Load OpenCV face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize recognizer for speech-to-text
recognizer = sr.Recognizer()

@app.route('/process-video', methods=['POST'])
def process_video():
    # Get the uploaded video
    video_file = request.files['video']
    time_offset = float(request.form.get('timeOffset', 0))
    
    # Save video to a temporary file
    video_path = os.path.join('uploads', 'video.mp4')
    video_file.save(video_path)
    
    # Process video for face detection, transcription, and subtitle generation
    transcription, timestamps = process_video_with_transcription(video_path, time_offset)

    # Generate a unique output video filename
    output_video_path = os.path.join('processed_videos', f'video_{int(time.time())}.mp4')

    # Add subtitles to the video
    output_video_path = add_subtitles_to_video(video_path, transcription, timestamps, output_video_path)

    return jsonify({
        'success': True,
        'videoURL': output_video_path
    })

def process_video_with_transcription(video_path, time_offset):
    cap = cv2.VideoCapture(video_path)
    
    transcription = []
    timestamps = []
    last_transcription_time = time.time() - time_offset

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        max_confidence = -1
        active_face = None

        # Detect the face with the highest confidence
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            input_face = cv2.resize(face, (224, 224))
            input_face = input_face.astype(np.float32) / 255.0
            input_face = input_face.transpose((2, 0, 1))
            input_tensor = torch.tensor(input_face).float()
            input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_class = torch.max(output, 1)
                confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

            if confidence > max_confidence:
                max_confidence = confidence
                active_face = (x, y, w, h, predicted_class, confidence)

        # Capture the transcription using speech recognition
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=5)
        try:
            transcription_text = recognizer.recognize_google(audio)
            timestamps.append(time.time() - last_transcription_time)
            transcription.append(transcription_text)
        except Exception as e:
            print(f"Error: {e}")
    
    cap.release()
    return transcription, timestamps

def add_subtitles_to_video(video_path, transcription, timestamps, output_video_path):
    clip = VideoFileClip(video_path)
    # Create a subtitles clip or overlay and add it to the video
    # (You can use the moviepy library to overlay text as subtitles)
    # For simplicity, let's assume we just add the transcription text as subtitles in the video
    # Add subtitles processing here
    clip.write_videofile(output_video_path, codec="libx264")
    return output_video_path

if __name__ == '__main__':
    app.run(debug=True)