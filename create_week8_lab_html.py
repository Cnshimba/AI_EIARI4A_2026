import os

base_path = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 8 - Convolutional Neural Networks (CNNs)"
notes_path = os.path.join(base_path, "Week_8_Student_Notes.html")
out_path = os.path.join(base_path, "Week_8_Lab_1_LeNet_Keras.html")

with open(notes_path, "r", encoding="utf-8") as f:
    text = f.read()

# Extract header (up until the <article ...> tag is closed and content begins)
# In VUT template, content usually starts after a div with <h1 class="text-3xl font-bold ...">
header_end_str = '<div class="flex-1 flex flex-col items-center justify-center p-6 text-center">'
header_end = text.find(header_end_str)
if header_end == -1: print("Header not found"); exit(1)

# we need to skip past the h1 and span to the </div></div>
header_split = text.find("</div>\n\n            \n            \n\n<div class=\"flex items-start", header_end)
if header_split == -1: 
    # Fallback, just find the end of the hero container
    header_split = text.find("</div>\n\n            \n            \n\n", header_end)

if header_split == -1: 
    # Hard fallback
    header_split = text.find("<article", 0) + 150 # hacky

# Extract footer
footer_start_str = "</article>"
footer_start = text.find(footer_start_str)

header = text[:header_end + len(header_end_str)]
header += '''\n        <h1 class="text-3xl font-bold text-secondary m-0">Convolutional Neural Networks (CNNs)</h1>
        <span class="text-lg font-medium text-gray-500 dark:text-gray-400 mt-1">Lab 1: LeNet-5 Architecture (Keras)</span>
    </div>
</div>

<div class="mb-8 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-700 flex items-center justify-between">
    <div>
        <h4 class="font-bold text-slate-800 dark:text-slate-200">Jupyter Notebook Available</h4>
        <p class="text-sm text-slate-600 dark:text-slate-400 m-0">Download the interactive coding lab to run on your local machine.</p>
    </div>
    <a href="Week_8_Lab_1_LeNet.ipynb" download target="_blank" class="ml-4 px-4 py-2 bg-secondary hover:bg-yellow-600 text-white font-medium rounded-md transition-colors shadow-sm no-underline whitespace-nowrap">
        Download .ipynb
    </a>
</div>

<div class="flex items-start p-4 my-4 rounded-lg border-l-4 shadow-sm bg-[#c9984a]/10 dark:bg-[#c9984a]/20 border-[#c9984a] text-slate-700 dark:text-slate-300"><svg class="w-6 h-6 mr-3 flex-shrink-0 text-[#c9984a]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg><div class="flex-1 text-sm md:text-base"><strong>Lab Goal</strong><br />
Implement the foundational LeNet-5 Convolutional Neural Network (CNN) to recognize handwritten digits using TensorFlow and Keras, and export the model for Edge deployment.</div></div>

<hr />
<h2>1. Setup and Load Data</h2>
<p>Unlike raw algorithms where we used 1D rows for features, a CNN requires spatial dimensions. We load the MNIST dataset and reshape the 28x28 arrays into 3D tensors: <code>[Height=28, Width=28, Channels=1]</code>.</p>
<pre><code class="language-python">import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape to include the Channel dimension (Channel-Last format for TF)
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0

print(f"Train Data Shape: {train_images.shape}")
</code></pre>

<hr />
<h2>2. Building LeNet-5</h2>
<p>The LeNet-5 architecture alternates between <strong>Convolutional Layers</strong> (to extract features) and <strong>Pooling Layers</strong> (to compress spatial resolution), followed by <strong>Dense Layers</strong> for the final classification.</p>
<pre><code class="language-python">model = models.Sequential([
    # Block 1: Feature Extraction
    layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Block 2: Feature Extraction
    layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Flatten and Classify
    layers.Flatten(),
    layers.Dense(120, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
</code></pre>
<div class="flex items-start p-4 my-4 rounded-lg border-l-4 shadow-sm bg-[#002F6E]/10 dark:bg-[#002F6E]/20 border-[#002F6E] text-slate-700 dark:text-slate-300"><svg class="w-6 h-6 mr-3 flex-shrink-0 text-[#002F6E]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg><div class="flex-1 text-sm md:text-base"><strong>Observation:</strong> Look at the summary output. Notice how the height and width decrease after Pooling, but the number of filters increases after Convolution. The Flatten layer reshapes the 3D tensor back into a 1D vector for the Dense layer.</div></div>

<hr />
<h2>3. Compiling and Training</h2>
<p>Thanks to the high-level Keras API, training this complex neural network requires only two functional calls.</p>
<pre><code class="language-python">model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
</code></pre>

<hr />
<h2>4. Edge Export for Jetson</h2>
<p>Embedded systems like the Nvidia Jetson Orin Nano require optimized models. We use <code>TFLiteConverter</code> to convert our weighty floating-point CNN into a lean TensorFlow Lite model.</p>
<pre><code class="language-python"># Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
with open('lenet_mnist.tflite', 'wb') as f:
    f.write(tflite_model)

print("Export Complete! Transfer lenet_mnist.tflite to your Jetson.")
</code></pre>

<div class="mt-8 flex justify-center">
    <a href="Week_8_Student_Notes.html" class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-primary hover:bg-primary/90 shadow-sm transition-colors">
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m7 7l-7-7 7-7"></path></svg>
        Back to Student Notes
    </a>
</div>
'''

footer = text[footer_start:]

with open(out_path, "w", encoding="utf-8") as f:
    f.write(header + footer)

print(f"Created HTML lab at {out_path}")
