import cv2
import torch
from torchvision import transforms
import numpy as np
import time

# Assuming you have a pre-trained model (or the model is loaded)
class TalkNetASDModel:
    def __init__(self, model_path):
        # Initialize your model (Load the pre-trained weights)
        self.model = torch.load(model_path)
        self.model.eval()

    def detect_speaker(self, frame):
        # This is where you would preprocess the frame and run it through the model
        # For now, we will just return a dummy prediction
        # You should adapt this to the model you are using
        # Convert frame to a tensor and preprocess it
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(frame)
        input_tensor = input_tensor.unsqueeze(0).cuda()  # Assuming you're using CUDA

        with torch.no_grad():
            output = self.model(input_tensor)
            # Here you can extract the active speaker or confidence score from the output
            return output.argmax(dim=1)  # Dummy output: replace with actual logic

def main():
    # Load your model
    model_path = 'path_to_your_model.pth'  # Replace with your actual model path
    model = TalkNetASDModel(model_path)

    # Open webcam (0 is default webcam, change it if necessary)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Detect active speaker in the frame
        prediction = model.detect_speaker(frame)

        # Display the prediction (active speaker or confidence score)
        cv2.putText(frame, f'Active Speaker: {prediction.item()}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Active Speaker Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()