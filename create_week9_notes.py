import os

base_path = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 9 - Natural Language Processing (NLP)"
notes_path = os.path.join(base_path, "Week_9_Student_Notes.html")

with open(notes_path, "r", encoding="utf-8") as f:
    text = f.read()

header_end = text.find('<h2>1. From Words to Numbers</h2>')
if header_end == -1: print("Header not found"); exit(1)

footer_start = text.find('<div class="mt-12 p-8 bg-[#002F6E]/10')
if footer_start == -1:
    footer_start = text.find('</article>')

header = text[:header_end]
footer = text[footer_start:]

new_content = """<h2>1. From Raw Text to Numerical Data</h2>
<p>The fundamental challenge of Natural Language Processing (NLP) is representation. Computers execute mathematical operations; they cannot natively ingest sentences like "I love AI." We must translate discrete text into continuous mathematical vectors.</p>

<h3>1.1 Traditional Preprocessing: Cleaning the Mess</h3>
<p>Before advanced Deep Learning models ruled the field, raw text was cleaned heavily. A traditional pipeline involved:</p>
<ul>
<li><strong>Lowercasing</strong>: "The" and "the" become the identical string.</li>
<li><strong>Stopword Removal</strong>: Eliminating high-frequency, low-meaning words ("and", "the", "is") to reduce noise.</li>
<li><strong>Stemming/Lemmatization</strong>: Chopping prefixes/suffixes to find the root word. "Running", "Ran", and "Run" all map to "run".</li>
</ul>

<h3>1.2 Tokenization: Word, Character, and Subword (BPE)</h3>
<p>We cannot parse an entire document at once. We break it into atomic units called <strong>Tokens</strong>.</p>
<ul>
<li><strong>Word-level Tokenization</strong>: Splitting by spaces (["I", "love", "AI"]). Limitation: massive vocabularies and inability to handle typos or completely new words.</li>
<li><strong>Character-level Tokenization</strong>: ["I", " ", "l", "o", "v", "e", ...]. Limitation: characters carry almost no semantic meaning alone, making patterns very hard for the network to find.</li>
<li><strong>Subword Tokenization (BPE)</strong>: The modern standard used by models like GPT. Byte-Pair Encoding merges the most frequent pairs of characters. Common words stay whole ("love"), while rare words split into logical subwords ("un", "believ", "able"). This gracefully balances vocabulary size and semantic clarity.</li>
</ul>

<h3>1.3 Vocabulary and Classical Vectorization</h3>
<p>Once tokens are defined, we map them to a Vocabulary (a dictionary lookup, e.g., "Apple" = ID 150). Next, how do we represent a full sentence?</p>
<ul>
<li><strong>Bag of Words (BoW)</strong>: Represents a sentence purely by the count of each token. It destroys word order completely. "Dog bites man" and "Man bites dog" look mathematically identical in BoW!</li>
<li><strong>TF-IDF (Term Frequency - Inverse Document Frequency)</strong>: An upgrade to BoW. It mathematically penalizes words that appear too frequently across <em>all</em> documents (like "the") to highlight rare, meaningful words. Very popular in classic Machine Learning (Scikit-Learn).</li>
<li><strong>One-Hot Encoding</strong>: A vector of zeros with a single '1' at the word's ID. Extremely inefficient due to sparsity (a 50,000-word vocabulary means a vector of 50,000 dimensions for just one word!).</li>
</ul>

<hr />
<h2>2. The Rise of Word Embeddings</h2>
<p>One-Hot Encoding was computationally brutal and contained zero semantic understanding (the vector for "King" was mathematically equidistant to "Queen" and "Potato").</p>
<p><strong>Word Embeddings</strong> (like Word2Vec, GloVe, or Keras's <code>layers.Embedding</code>) changed everything. Instead of sparse arrays, every word is mapped to a <strong>Dense Vector</strong> (e.g., of size 300) containing floating-point numbers.</p>
<p>These vectors are learned during training. The core intuition relies on distribution: <em>"You shall know a word by the company it keeps."</em> If "Dog" and "Cat" frequently appear next to "Pet", "Leash", and "Cute", their embedding vectors will shift closer together in mathematical space!</p>

<div class="flex items-start p-4 my-4 rounded-lg border-l-4 shadow-sm bg-[#002F6E]/10 dark:bg-[#002F6E]/20 border-[#002F6E] text-slate-700 dark:text-slate-300"><svg class="w-6 h-6 mr-3 flex-shrink-0 text-[#002F6E]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg><div class="flex-1 text-sm md:text-base"><strong>Semantic Arithmetic</strong>: A famous consequence of well-trained embeddings is that mathematical operations mirror linguistic analogies: <br/> <code>Vector(King) - Vector(Man) + Vector(Woman) ≈ Vector(Queen)</code></p></div></div>

<hr />
<h2>3. The Architecture for Text</h2>
<h3>3.1 The Sequence Problem: RNNs and LSTMs</h3>
<p>To fix the "Bag of Words" problem of ignoring sequences, researchers developed <strong>Recurrent Neural Networks (RNNs)</strong>. These models process text word-by-word sequentially, maintaining a "hidden state" (memory) of everything read so far.</p>
<p><strong>The Vanishing Gradient Problem</strong>: Basic RNNs suffer catastrophically on long paragraphs. When updating weights (backpropagation through time), gradients diminish exponentially. By word 50, the network completely forgets word 1. <strong>LSTMs (Long Short-Term Memory)</strong> introduced complex gates to "choose" what to remember and what to forget, fixing this problem for medium-length text.</p>

<h3>3.2 1D Convolutions (TextCNN)</h3>
<p>Surprisingly, Convolutional Neural Networks (CNNs) from last week also work exceptionally well on text! If an image is a 2D grid, a sentence is a 1D sequence of word embeddings. We can slide a 1D filter across 3 words at a time. This allows the network to detect <strong>N-Grams</strong> (local word patterns) like "Not Good" which instantly flip sentiment.</p>

<h3>3.3 A Glimpse of the Future: Attention</h3>
<p>LSTMs are unparallelizable (you must process word 3 before word 4), making them painfully slow to train on GPUs. This sparked the quest for a new architecture. Modern NLP replaces sequences with the <strong>Attention Mechanism</strong>: algorithms that allow the network to look at <em>all</em> words in a sentence simultaneously and calculate which words relate strictly to each other. This is the foundation of GenAI and Transformers, which we cover next week!</p>

<hr />
<h2>4. Lab Preview: Text Classification in Keras</h2>
<p>In this week's lab, we will use TensorFlow and Keras to build a robust NLP classifier targeting the IMDB Movie Review dataset.</p>
<ol>
<li><strong>TextVectorization</strong>: We'll use the powerful Keras <code>TextVectorization</code> layer to automatically tokenize strings and build a vocabulary map under the hood.</li>
<li><strong>Embedding Layer</strong>: We will dynamically train custom dense embeddings via <code>layers.Embedding</code>.</li>
<li><strong>Modeling</strong>: We will use a fast 1D Pooling or Convolution approach to map sequences to sentiment prediction.</li>
<li><strong>Jetson Export</strong>: You will convert this NLP model to TensorFlow Lite and run localized Edge NLP on your Jetson Orin Nano!</li>
</ol>
"""

with open(notes_path, "w", encoding="utf-8") as f:
    f.write(header + new_content + footer)

# Do the same concept for Lecture Notes
lecture_path = os.path.join(base_path, "Week_9_Lecture_Notes.html")
with open(lecture_path, "w", encoding="utf-8") as f:
    f.write(header + new_content + footer)
    
print("Updated Week 9 html files via python script")
