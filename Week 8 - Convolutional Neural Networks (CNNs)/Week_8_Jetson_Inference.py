import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

# ---------------------------------------------------------
# 1. Define the LeNet Architecture (Must match training!)
# ---------------------------------------------------------
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Feature Extractor
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Classifier
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------------------------------------------------
# 2. Setup Device and Load Model
# ---------------------------------------------------------
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LeNet().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Could not find '{model_path}'. Make sure you copied it to this folder.")
        exit(1)
        
    model.eval() # Set to evaluation mode
    return model, device

# ---------------------------------------------------------
# 3. Preprocessing Function
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def preprocess_frame(frame, device):
    # Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Invert colors (MNIST is white text on black background, webcam is usually opposite)
    # We use a simple threshold to make it "digit-like"
    # Adaptive thresholding helps with varying lighting
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Apply PyTorch Transforms
    tensor = transform(resized)
    
    # Add batch dimension (1, 1, 28, 28) and send to GPU
    tensor = tensor.unsqueeze(0).to(device)
    
    return tensor, resized

# ---------------------------------------------------------
# 4. Main Loop
# ---------------------------------------------------------
def main():
    MODEL_PATH = 'lenet_mnist.pth'
    net, device = load_model(MODEL_PATH)
    
    # Open Webcam (0 is usually the default USB camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Inference... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Define a Region of Interest (ROI) box in the center of the screen
        # We only want to classify what's inside this box
        height, width, _ = frame.shape
        box_size = 200
        x1 = int(width / 2 - box_size / 2)
        y1 = int(height / 2 - box_size / 2)
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Preprocess and Infer
        input_tensor, debug_image = preprocess_frame(roi, device)
        
        with torch.no_grad():
            outputs = net(input_tensor)
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        prediction = predicted.item()
        conf_score = confidence.item()
        
        # -----------------------------------------------------
        # Visualization
        # -----------------------------------------------------
        # Draw the box on the original frame
        color = (0, 255, 0) # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Display Prediction text
        text = f"Digit: {prediction} ({conf_score*100:.1f}%)"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Show the main window
        cv2.imshow('Jetson Edge AI - LeNet', frame)
        
        # Show what the network actually sees (the 28x28 input) - blown up for visibility
        debug_view = cv2.resize(debug_image, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Network Input', debug_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
