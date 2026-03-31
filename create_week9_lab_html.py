import os

base_path = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 9 - Natural Language Processing (NLP)"
notes_path = os.path.join(base_path, "Week_9_Student_Notes.html")
out_path = os.path.join(base_path, "Week_9_Lab_1_Text_Classification.html")

with open(notes_path, "r", encoding="utf-8") as f:
    text = f.read()

# Extract header
header_end_str = '<div class="flex-1 flex flex-col items-center justify-center p-6 text-center">'
header_end = text.find(header_end_str)
if header_end == -1: print("Header not found"); exit(1)

# Extract footer
footer_start_str = "</article>"
footer_start = text.find(footer_start_str)

header = text[:header_end + len(header_end_str)]
header += '''\n        <h1 class="text-3xl font-bold text-secondary m-0">Natural Language Processing (NLP)</h1>
        <span class="text-lg font-medium text-gray-500 dark:text-gray-400 mt-1">Lab 1: Text Classification via Embeddings (Keras)</span>
    </div>
</div>

<div class="mb-8 p-4 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-700 flex items-center justify-between">
    <div>
        <h4 class="font-bold text-slate-800 dark:text-slate-200">Jupyter Notebook Available</h4>
        <p class="text-sm text-slate-600 dark:text-slate-400 m-0">Download the interactive coding lab to run on your local machine.</p>
    </div>
    <a href="Week_9_Lab_1_Text_Classification.ipynb" download target="_blank" class="ml-4 px-4 py-2 bg-secondary hover:bg-yellow-600 text-white font-medium rounded-md transition-colors shadow-sm no-underline whitespace-nowrap">
        Download .ipynb
    </a>
</div>

<div class="flex items-start p-4 my-4 rounded-lg border-l-4 shadow-sm bg-[#c9984a]/10 dark:bg-[#c9984a]/20 border-[#c9984a] text-slate-700 dark:text-slate-300"><svg class="w-6 h-6 mr-3 flex-shrink-0 text-[#c9984a]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg><div class="flex-1 text-sm md:text-base"><strong>Lab Goal</strong><br />
Implement a complete NLP pipeline—Tokenization, Word Embeddings, and CNN processing—to classify movie reviews as Positive or Negative using TensorFlow/Keras. Finally, export the model for Edge deployment.</div></div>

<hr />
<h2>1. Setup Data</h2>
<p>Because text must be translated to numeric form, mapping dictionaries take massive amounts of RAM and disk space. Instead of raw text, Keras supplies the IMDb dataset already pre-tokenized into Integer IDs.</p>
<pre><code class="language-python">import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load IMDb data. Restrict vocabulary to the top 10,000 most common words.
vocab_size = 10000
(train_data, train_labels), (test_data, test_labels) = datasets.imdb.load_data(num_words=vocab_size)

print(f"Training reviews: {len(train_data)}")
</code></pre>

<hr />
<h2>2. Padding Sequences</h2>
<p>Neural networks process batches of arrays, requiring all arrays in a single batch to have exactly the same geometric shape. Reviews, unfortunately, have highly variable lengths. We apply <code>pad_sequences</code> to standardize length.</p>
<pre><code class="language-python">maxlen = 200

# Truncate text longer than 200, pad zeroes to text shorter than 200
train_padded = pad_sequences(train_data, maxlen=maxlen, padding='post', truncating='post')
test_padded = pad_sequences(test_data, maxlen=maxlen, padding='post', truncating='post')
</code></pre>

<hr />
<h2>3. Building the TextCNN Architecture</h2>
<p>The <code>Embedding</code> layer is trained simultaneously with the classification layer, creating dense word representations mapping to 64 dimensions.</p>
<pre><code class="language-python">model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=maxlen),
    
    # 1D Convolution sliding across sequence to detect N-Grams
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(), # Shrink variable length mapping down to a summary
    
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Binary Output (0 or 1)
])

model.summary()
</code></pre>
<div class="flex items-start p-4 my-4 rounded-lg border-l-4 shadow-sm bg-[#002F6E]/10 dark:bg-[#002F6E]/20 border-[#002F6E] text-slate-700 dark:text-slate-300"><svg class="w-6 h-6 mr-3 flex-shrink-0 text-[#002F6E]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg><div class="flex-1 text-sm md:text-base"><strong>Observation:</strong> Look at the summary output. The Embedding layer contains 640,000 trainable parameters (10,000 vocab * 64 weights each), drastically overshadowing the rest of the 1D Convolution mapping layers!</div></div>

<hr />
<h2>4. Compiling and Exfiltrating to Jetson Edge AI</h2>
<p>After a quick training cycle leveraging GPU speed, we convert our robust NLP <code>model</code> structure into a deployment-ready <code>TFLite</code> model. This file strips away massive overhead APIs designed for training, producing an inference-only engine fit for low-power robotics or Jetson Nano hardware.</p>
<pre><code class="language-python">model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_padded, train_labels, epochs=3, batch_size=64, validation_data=(test_padded, test_labels))

# Convert the model for Edge Devices
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Export .tflite to disk
with open('sentiment_model.tflite', 'wb') as f:
    f.write(tflite_model)
    
print("Export Complete! Transfer sentiment_model.tflite to your Jetson.")
</code></pre>

<div class="mt-8 flex justify-center">
    <a href="Week_9_Student_Notes.html" class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-primary hover:bg-primary/90 shadow-sm transition-colors">
        <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m7 7l-7-7 7-7"></path></svg>
        Back to Student Notes
    </a>
</div>
'''

footer = text[footer_start:]

with open(out_path, "w", encoding="utf-8") as f:
    f.write(header + footer)

print(f"Created HTML lab at {out_path}")
