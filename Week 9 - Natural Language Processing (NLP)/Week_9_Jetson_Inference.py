"""
Week 9 - Jetson Edge AI: Sentiment Analysis (TFLite)
======================================================
Run on your Jetson Orin Nano AFTER exporting sentiment_model.tflite from the lab notebook.

Usage:
    python3 Week_9_Jetson_Inference.py

The script loads the TFLite sentiment model and runs a simple interactive
loop where you can type sentences and get a POSITIVE/NEGATIVE prediction.
"""

import numpy as np
import json

# Use tflite_runtime on edge devices (lightweight), fallback to full TF
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime (Edge mode)")
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    print("Using full TensorFlow (Development mode)")

# ---------------------------------------------------------
# CONFIG — Match these to your training parameters!
# ---------------------------------------------------------
MODEL_PATH  = "sentiment_model.tflite"
VOCAB_SIZE  = 10000
MAXLEN      = 200

# ---------------------------------------------------------
# 1. Recreate a simple word-index lookup from IMDb dataset
#    (on Jetson we can't import keras easily, so we use
#    the pre-saved JSON lookup table)
# ---------------------------------------------------------
def load_vocab():
    """Try to load saved vocab.json, otherwise fetch from Keras."""
    try:
        with open("vocab.json", "r") as f:
            word_index = json.load(f)
        print(f"Loaded vocabulary ({len(word_index)} words) from vocab.json")
        return word_index
    except FileNotFoundError:
        import tensorflow as tf
        print("vocab.json not found — downloading from Keras (requires internet)...")
        word_index = tf.keras.datasets.imdb.get_word_index()
        # Shift by 3 to match the IMDb convention
        word_index = {k: (v + 3) for k, v in word_index.items()}
        word_index["<PAD>"]   = 0
        word_index["<START>"] = 1
        word_index["<UNK>"]   = 2
        word_index["<UNUSED>"]= 3
        with open("vocab.json", "w") as f:
            json.dump(word_index, f)
        print("Saved vocab.json for future runs.")
        return word_index

# ---------------------------------------------------------
# 2. Load TFLite model
# ---------------------------------------------------------
def load_model():
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_index  = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    print("TFLite model loaded successfully!")
    return interpreter, input_index, output_index

# ---------------------------------------------------------
# 3. Encode a sentence to a padded integer sequence
# ---------------------------------------------------------
def encode_sentence(sentence, word_index):
    words  = sentence.lower().strip().split()
    ids    = [word_index.get(w, word_index["<UNK>"]) for w in words]
    ids    = ids[:MAXLEN]                       # truncate
    padded = ids + [0] * (MAXLEN - len(ids))    # post-pad
    return np.array(padded, dtype=np.int32).reshape(1, MAXLEN)

# ---------------------------------------------------------
# 4. Run Inference
# ---------------------------------------------------------
def predict(interpreter, input_index, output_index, sentence, word_index):
    input_tensor = encode_sentence(sentence, word_index)
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()
    score = interpreter.get_tensor(output_index)[0][0]
    label = "POSITIVE 😊" if score >= 0.5 else "NEGATIVE 😞"
    return label, score

# ---------------------------------------------------------
# 5. Interactive Loop
# ---------------------------------------------------------
def main():
    print("\n=== Jetson Edge AI — Sentiment Analyser (TFLite) ===")
    print("Loading resources...\n")

    word_index = load_vocab()
    interpreter, in_idx, out_idx = load_model()

    print("\nType a sentence and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Input > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if text.lower() in ("quit", "exit", "q"):
            break

        if not text:
            continue

        label, score = predict(interpreter, in_idx, out_idx, text, word_index)
        print(f"  Sentiment: {label} ({score * 100:.1f}% confidence)\n")

    print("Goodbye!")

if __name__ == "__main__":
    main()
