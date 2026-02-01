# Week 8 Lab: CNNs with PyTorch (LeNet-5)

## Foreword
In this lab, we build our first **Convolutional Neural Network (CNN)** using **PyTorch**.
We will implement the classic **LeNet-5** architecture to classify handwritten digits (MNIST).

PyTorch simplifies the process:
*   No more manual backprop (Autograd handles it).
*   Layers are pre-defined (`nn.Conv2d`, `nn.Linear`).

---

## Part 1: Training LeNet on PC

### Step 1: Import Dependencies
We import `torch` and `torchvision`. The latter contains popular datasets (MNIST) and transforms.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device configuration (Use GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### Step 2: Prepare the Dataset
We use `torchvision` to download and transform the MNIST data.
*   We transform images to Tensors.
*   We normalize them (Mean 0.5, Std 0.5) to help training.

```python
# Hyper-parameters
batch_size = 64

# Transformations: Convert to Tensor and Normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to range [-1, 1]
])

# Download and Load Data
# This will create a 'data' folder
try:
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training Batch: {len(train_loader)} batches")
except Exception as e:
    print(f"Error loading data: {e}")
```

### Step 3: Define the LeNet-5 Architecture
We define a class that inherits from `nn.Module`.

**LeNet-5 Structure:**
1.  **Conv1**: 1 input channel (grayscale) -> 6 output channels (5x5 kernel).
2.  **Pool**: 2x2 Max Pooling.
3.  **Conv2**: 6 input channels -> 16 output channels.
4.  **Pool**: 2x2 Max Pooling.
5.  **FC1** (Fully Connected): Flatten -> 120 neurons.
6.  **FC2**: 84 neurons.
7.  **Output**: 10 neurons (Digits 0-9).

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Feature Extractor (Convolutional Layers)
        self.conv1 = nn.Conv2d(1, 6, 5) # In: 1, Out: 6, Kernel: 5x5
        self.pool = nn.MaxPool2d(2, 2)  # Kernel: 2x2, Stride: 2
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Classifier (Fully Connected Layers)
        # Input to FC1 depends on image size. MNIST is 28x28.
        # Conv1(5x5) -> 24x24 -> Pool(2x2) -> 12x12
        # Conv2(5x5) -> 8x8 -> Pool(2x2) -> 4x4
        # So final feature map is 16 channels * 4 * 4 pixels = 256 inputs.
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = LeNet().to(device)
print(net)
```

### Step 4: Define Loss and Optimizer
We use **CrossEntropyLoss** (standard for classification) and **SGD** (Stochastic Gradient Descent) with momentum.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

### Step 5: Training Loop
We iterate through the dataset multiple times (epochs).

```python
epochs = 5
loss_history = []

print("Starting Training...")
for epoch in range(epochs):  
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # Print every 200 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            loss_history.append(running_loss / 200)
            running_loss = 0.0

print('Finished Training')
```

### Step 6: Visualize Loss Curve
Let's see how the error decreased over time.

```python
plt.plot(loss_history)
plt.xlabel('Batch Intervals (x200)')
plt.ylabel('Loss')
plt.show()
```

### Step 7: Evaluate on Test Data
We check the accuracy on the 10,000 test images.

```python
correct = 0
total = 0
# No gradient needed for evaluation
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # The class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f} %')
```

> [!GOAL]
> **Expected Result**: You should achieve **>98% accuracy** with this simple network!

---

## Part 2: Deploying LeNet-5 to NVIDIA Jetson Orin Nano

> [!NOTE]
> This guide is a "Part 2" extension of Lab 1. It assumes you have a trained PyTorch model and an NVIDIA Jetson device (Nano, Orin Nano, or AGX) with a USB webcam.

### Goal
Take the digit classification model you trained in Jupyter Notebook on your PC and deploy it to an edge device (Jetson Orin Nano) to classify handwritten digits in the real world using a webcam.

### Step 1: Save Your Trained Model
In your `Week_8_Lab_1_LeNet.ipynb` notebook on your PC, you trained a model called `net`. We need to save its learned parameters (weights) to a file so we can move it.

Add the following code to a new cell at the end of your notebook and run it:

```python
# Save the trained model weights
PATH = './lenet_mnist.pth'
torch.save(net.state_dict(), PATH)
print(f"Model saved to {PATH}")
```

This will create a file named `lenet_mnist.pth` in your working directory.

### Step 2: Transfer Files to the Jetson
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

### Step 3: Run Inference on Jetson
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

### How It Works
*   The script loads your trained `lenet_mnist.pth`.
*   It opens the webcam feed using OpenCV.
*   It grabs a frame, converts it to grayscale, creating a **Region of Interest (ROI)** in the center (the green box).
*   It resizes that box to 28x28 pixels (what LeNet expects).
*   It runs the image through the Neural Network on the Jetson's GPU (CUDA).
*   It displays the predicted number on the screen.

### Troubleshooting
*   **"No Camera Found"**: Ensure your camera is plugged in. Try changing `cap = cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the python script if you have multiple cameras.
*   **"CUDA error"**: Typically happens if the Jetson PyTorch version is incompatible. The script attempts to fall back to CPU if CUDA is unavailable, but on a Jetson, you really want CUDA!
