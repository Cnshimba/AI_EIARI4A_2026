import os
import re

base_path = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 8 - Convolutional Neural Networks (CNNs)"
file_path = os.path.join(base_path, "Week_8_Student_Notes.html")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update Topic & Goal
text = text.replace("CNNs) with PyTorch", "CNNs) with TensorFlow & Keras")
text = text.replace("architecture in PyTorch", "architecture in Keras")

goal_old = 'We will introduce <strong>PyTorch</strong>, a framework that handles the calculus automatically ("Autograd"), allowing us to focus on architecture rather than implementation details.'
goal_new = 'We will build upon the <strong>TensorFlow and Keras</strong> skills you learned last week. By leveraging the high-level Keras API, we can focus entirely on architecture rather than low-level calculus.'
text = text.replace(goal_old, goal_new)

# 2. Rewrite Section 1 completely
sec1_start = text.find("<h2>1. Introduction to PyTorch</h2>")
sec2_start = text.find("<h2>2. Convolutional Neural Networks (CNNs)</h2>")

new_sec1 = """<h2>1. Deep Computer Vision with Keras</h2>
<p>Last week we used Keras to build dense neural networks. This week, we will introduce the specific Layers needed for Computer Vision. Keras abstracts away the difficult tensor math so you can snap layers together like Lego blocks.</p>

<h3>1.1 Image Data Dimensions (Channel-Last)</h3>
<p>In standard Artificial Neural Networks, we deal with 1D vectors of data. In Computer Vision, we deal with images which are inherently 3D. TensorFlow and Keras use the <strong>Channel-Last</strong> convention by default:</p>
<ul>
<li><strong>Grayscale Image</strong>: <code>[Height, Width, 1]</code> (The 1 represents the single grayscale channel)</li>
<li><strong>Color Image</strong>: <code>[Height, Width, 3]</code> (The 3 represents Red, Green, and Blue channels)</li>
<li><strong>Batch of Images</strong>: <code>[BatchSize, Height, Width, Channels]</code></li>
</ul>
<p>When loading image data using TensorFlow datasets or Keras preprocessing utilities, the data is automatically formatted this way.</p>

<h3>1.2 The Conv2D Layer</h3>
<p>The workhorse of any modern Computer Vision model is the Convolutional Layer. In Keras, this is called <code>layers.Conv2D</code>.</p>
<pre><code class="language-python">from tensorflow.keras import layers

# Creating a Convolutional Layer
conv_layer = layers.Conv2D(
    filters=32,            # Number of feature maps to output
    kernel_size=(3, 3),    # Size of the sliding window
    activation='relu',     # Apply ReLU directly inside the layer!
    input_shape=(28, 28, 1) # Only required for the very first layer
)
</code></pre>

<h3>1.3 The MaxPooling2D Layer</h3>
<p>To reduce computational load and provide translation invariance, we pair our Convolutional layers with Pooling layers.</p>
<pre><code class="language-python"># Creating a Max Pooling Layer
pool_layer = layers.MaxPooling2D(
    pool_size=(2, 2)       # Reduces spatial dimensions by half
)
</code></pre>

<h3>1.4 Flatten & Dense Layers</h3>
<p>At the end of a CNN, the feature maps must be collapsed into a 1D vector so they can be fed into standard Dense layers for final classification.</p>
<pre><code class="language-python">flatten_layer = layers.Flatten()
dense_layer = layers.Dense(10, activation='softmax')
</code></pre>
<hr />
"""

if sec1_start != -1 and sec2_start != -1:
    text = text[:sec1_start] + new_sec1 + text[sec2_start:]

# 3. Shape Flow adjustments
text = text.replace("Input: <code>[1, 28, 28]</code> (1 Channel, 28 Height", "Input: <code>[28, 28, 1]</code> (28 Height")
text = text.replace("Conv Layer (k=5, p=2)</strong>: The number of channels increases (we find multiple features), but the height/width stays roughly the same. -&gt; <code>[6, 28, 28]</code>", "Conv Layer (k=5, padding=\"same\")</strong>: The number of channels increases (we find multiple features), but the height/width stays roughly the same. -&gt; <code>[28, 28, 6]</code>")
text = text.replace("Pool Layer (2x2)</strong>: The spatial dimensions are cut in half. -&gt; <code>[6, 14, 14]</code>", "Pool Layer (2x2)</strong>: The spatial dimensions are cut in half. -&gt; <code>[14, 14, 6]</code>")
text = text.replace("take us to -&gt; <code>[16, 5, 5]</code>", "take us to -&gt; <code>[5, 5, 16]</code>")
text = text.replace("<code>16 channels x 5 height x 5 width</code>", "<code>5 height x 5 width x 16 channels</code>")
text = text.replace("<code>x.view(-1, 400)</code>", "<code>layers.Flatten()</code>")

# 4. Lab Preview adjustments
text = text.replace("<h2>5. Lab Preview: LeNet in PyTorch</h2>", "<h2>5. Lab Preview: LeNet in Keras</h2>")
text = text.replace("In this week's lab (Experiment 1), we will implement the LeNet-5 architecture using PyTorch. This lab moves away from raw math and focuses on the high-level API <code>torch.nn</code>.", "In this week's lab, we will implement the LeNet-5 architecture using TensorFlow/Keras.")
text = text.replace("<code>nn.Module</code>, defining the layers (<code>Conv2d</code>, <code>Linear</code>)", "<code>models.Sequential()</code>, stacking layers like <code>Conv2D</code> and <code>Dense</code>")
text = text.replace("You will implement the <code>forward()</code> method to define how data flows through these layers.", "Keras handles the forward pass automatically based on layer order.")
text = text.replace("You will use the <code>torch.optim</code> module to handle the weight updates, replacing our manual Gradient Descent loops.", "You will use <code>model.compile()</code> to define an optimizer and <code>model.fit()</code> to train the network.")

# 5. Summary adjustments
text = text.replace("We have transitioned from the manual mechanics of backpropagation to the automated efficiency of PyTorch.", "We expanded our framework toolkit to include Convolutional layers in TensorFlow/Keras.")
text = text.replace("PyTorch: Automates gradient calculation, enables GPU acceleration", "TensorFlow/Keras: Rapid CNN architecture deployment and GPU acceleration")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

# Do the same for Lecture Notes
lecture_file = os.path.join(base_path, "Week_8_Lecture_Notes.html")
with open(lecture_file, "r", encoding="utf-8") as f:
    text2 = f.read()

text2 = text2.replace("CNNs with PyTorch", "CNNs with TensorFlow/Keras")
text2 = text2.replace("<h3>Part 1: Introduction to PyTorch (20 mins)</h3>", "<h3>Part 1: Intro to Keras for CNNs (20 mins)</h3>")
text2 = text2.replace("Tensors", "Channel-Last Data Dimensions")
text2 = text2.replace("Autograd", "The Keras API")
text2 = text2.replace("<code>[1, 28, 28]</code>", "<code>[28, 28, 1]</code>")
text2 = text2.replace("<code>[6, 28, 28]</code>", "<code>[28, 28, 6]</code>")
text2 = text2.replace("<code>[6, 14, 14]</code>", "<code>[14, 14, 6]</code>")
text2 = text2.replace("<code>[16, 5, 5]</code>", "<code>[5, 5, 16]</code>")
text2 = text2.replace("<code>[400]          # 16×5×5 = 400</code>", "<code>[400]          # 5×5×16 = 400</code>")
text2 = text2.replace("RuntimeError: size mismatch", "ValueError: Invalid input shape for Flatten")
text2 = text2.replace("nn.Conv2d()", "layers.Conv2D()")
text2 = text2.replace("nn.MaxPool2d()", "layers.MaxPooling2D()")
text2 = text2.replace("nn.Linear()", "layers.Dense()")
text2 = text2.replace("nn.ReLU()", "activation='relu'")
text2 = text2.replace("optimizer.zero_grad(), loss.backward(), optimizer.step()", "model.compile(), model.fit()")
text2 = text2.replace("PyTorch", "TensorFlow/Keras")
text2 = text2.replace("pytorch", "keras")

with open(lecture_file, "w", encoding="utf-8") as f:
    f.write(text2)

print("Notes updated.")
