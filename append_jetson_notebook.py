import json
import os

nb_path = r"d:\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 8 - Convolutional Neural Networks (CNNs)\Week_8_Lab_1_LeNet.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

new_cells = [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2: Deploying LeNet-5 to NVIDIA Jetson Orin Nano\n",
    "\n",
    "> **Note:** This section assumes you have an NVIDIA Jetson device available.\n",
    "\n",
    "### Step 1: Save Your Trained Model\n",
    "We need to save the model weights to transfer them to the Jetson."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the trained model weights\n",
    "PATH = './lenet_mnist.pth'\n",
    "torch.save(net.state_dict(), PATH)\n",
    "print(f\"Model saved to {PATH}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Step 2: Transfer and Run\n",
    "1. Transfer `lenet_mnist.pth` and `Week_8_Jetson_Inference.py` to your Jetson.\n",
    "2. On the Jetson, run: `python3 Week_8_Jetson_Inference.py`"
   ]
  }
]

# Check if already added
last_source = data['cells'][-1]['source']
is_already_there = any("Deploying LeNet-5 to NVIDIA Jetson" in line for line in last_source)

if not is_already_there:
    data['cells'].extend(new_cells)
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1) # Using indent=1 based on file inspection
    print("Notebook updated.")
else:
    print("Notebook already contains Jetson section.")
