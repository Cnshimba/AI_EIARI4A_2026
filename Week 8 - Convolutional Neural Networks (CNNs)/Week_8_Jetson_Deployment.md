# Week 8: Deploying LeNet-5 to NVIDIA Jetson Orin Nano

> [!NOTE]
> This guide is a "Part 2" extension of Lab 1. It assumes you have a trained PyTorch model and an NVIDIA Jetson device (Nano, Orin Nano, or AGX) with a USB webcam.

## Goal
Take the digit classification model you trained in Jupyter Notebook on your PC and deploy it to an edge device (Jetson Orin Nano) to classify handwritten digits in the real world using a webcam.

## Step 1: Save Your Trained Model
In your `Week_8_Lab_1_LeNet.ipynb` notebook on your PC, you trained a model called `net`. We need to save its learned parameters (weights) to a file so we can move it.

Add the following code to a new cell at the end of your notebook and run it:

```python
# Save the trained model weights
PATH = './lenet_mnist.pth'
torch.save(net.state_dict(), PATH)
print(f"Model saved to {PATH}")
```

This will create a file named `lenet_mnist.pth` in your working directory.

## Step 2: Transfer Files to the Jetson
You need to move two files to your Jetson device:
1.  The model weights file: `lenet_mnist.pth`
2.  The inference script: `Week_8_Jetson_Inference.py` (provided in the course materials)

You can use a USB drive or `scp` (Secure Copy) over the network.

**Using SCP (Example):**
Open your terminal (PowerShell or Command Prompt) and run:
```powershell
# Syntax: scp <source_files> <username>@<jetson_ip_address>:<destination_folder>
scp lenet_mnist.pth Week_8_Jetson_Inference.py jetson@192.168.1.50:/home/jetson/Desktop/
```
*(Replace `jetson`, `192.168.1.50`, and the paths with your actual details).*

## Step 3: Run Inference on Jetson
1.  Connect a USB Webcam to the Jetson.
2.  Open a terminal on the Jetson.
3.  Navigate to the folder where you copied the files.
    ```bash
    cd ~/Desktop
    ```
4.  Run the inference script using Python 3.
    ```bash
    python3 Week_8_Jetson_Inference.py
    ```

## How It Works
*   The script loads your trained `lenet_mnist.pth`.
*   It opens the webcam feed using OpenCV.
*   It grabs a frame, converts it to grayscale, creating a **Region of Interest (ROI)** in the center (the green box).
*   It resizes that box to 28x28 pixels (what LeNet expects).
*   It runs the image through the Neural Network on the Jetson's GPU (CUDA).
*   It displays the predicted number on the screen.

## Troubleshooting
*   **"No Camera Found"**: Ensure your camera is plugged in. Try changing `cap = cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the python script if you have multiple cameras.
*   **"CUDA error"**: Typically happens if the Jetson PyTorch version is incompatible. The script attempts to fall back to CPU if CUDA is unavailable, but on a Jetson, you really want CUDA!
