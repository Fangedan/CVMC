import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the last fully connected layer for your task (e.g., 2 classes)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes (Speaker 1 vs. Speaker 2)

# Set model to evaluation mode
model.eval()

# OpenCV to capture video from your webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load OpenCV face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize a variable to track the highest confidence
    max_confidence = -1
    active_face = None  # This will hold the coordinates and data of the active speaker

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]
        
        # Preprocess the face for the model
        input_face = cv2.resize(face, (224, 224))  # Resize to 224x224 for ResNet input
        input_face = input_face.astype(np.float32) / 255.0  # Normalize between 0-1
        input_face = input_face.transpose((2, 0, 1))  # Change to CxHxW format (channels first)
        input_tensor = torch.tensor(input_face).float()  # Convert to tensor
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

        # Pass the face through the model to get the active speaker and confidence score
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
            confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

        # If this face has the highest confidence, assign it as the active speaker
        if confidence > max_confidence:
            max_confidence = confidence
            active_face = (x, y, w, h, predicted_class, confidence)

    # If an active face has been detected (the one with the highest confidence), draw it
    if active_face:
        x, y, w, h, predicted_class, confidence = active_face
        speaker = "Speaker 1" if predicted_class == 0 else "Speaker 2"
        color = (0, 255, 0)  # Green box for the active speaker
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{speaker} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow("Active Speaker Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
