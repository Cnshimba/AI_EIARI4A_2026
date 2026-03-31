import cv2
import numpy as np

# Use tflite_runtime if on Edge, otherwise fallback to tf.lite
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ---------------------------------------------------------
# 1. Setup Device and Load TFLite Model
# ---------------------------------------------------------
def load_tflite_model(model_path):
    print("Loading TFLite model...")
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully!")
    except ValueError:
        print(f"ERROR: Could not find '{model_path}'. Make sure you copied it to this folder.")
        exit(1)
        
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    return interpreter, input_details[0]['index'], output_details[0]['index']

# ---------------------------------------------------------
# 2. Preprocessing Function
# ---------------------------------------------------------
def preprocess_frame(frame):
    # Convert to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Invert colors (MNIST is white text on black background, webcam is usually opposite)
    # We use adaptive thresholding to help with varying lighting
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] matching our training data
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions for TFLite (1, 28, 28, 1)
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    
    return input_tensor, resized

# ---------------------------------------------------------
# 3. Main Loop
# ---------------------------------------------------------
def main():
    MODEL_PATH = 'lenet_mnist.tflite'
    interpreter, input_index, output_index = load_tflite_model(MODEL_PATH)
    
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
        input_tensor, debug_image = preprocess_frame(roi)
        
        # Set tensor to point to input data to be inferred
        interpreter.set_tensor(input_index, input_tensor)
        
        # Run inference
        interpreter.invoke()
        
        # Extract output data
        outputs = interpreter.get_tensor(output_index)[0]
        
        prediction = np.argmax(outputs)
        conf_score = outputs[prediction]
        
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
        cv2.imshow('Jetson Edge AI - LeNet (TFLite)', frame)
        
        # Show what the network actually sees (the 28x28 input) - blown up for visibility
        debug_view = cv2.resize(debug_image, (200, 200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Network Input', debug_view)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
