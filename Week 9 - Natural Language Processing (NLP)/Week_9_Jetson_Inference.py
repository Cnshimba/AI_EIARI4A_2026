import torch
import torch.nn as nn
import json
import argparse
import sys

# ---------------------------------------------------------
# 1. Define Model Architecture (Must match training!)
# ---------------------------------------------------------
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def prepare_sequence(seq, to_ix):
    # Split by space and map to Index. Use <UNK> (ID 1) if word not found.
    idxs = [to_ix.get(w, to_ix.get("<UNK>", 1)) for w in seq.lower().split()]
    return torch.tensor(idxs, dtype=torch.long)

def load_data(model_path, vocab_path, device):
    print(f"Loading vocab from {vocab_path}...")
    try:
        with open(vocab_path, 'r') as f:
            word_to_ix = json.load(f)
    except FileNotFoundError:
        print("Error: Vocab file not found.")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    try:
        # Config (Must match training)
        VOCAB_SIZE = len(word_to_ix)
        EMBED_DIM = 10
        NUM_CLASS = 2
        
        model = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, word_to_ix
    except FileNotFoundError:
        print("Error: Model file not found.")
        sys.exit(1)

# ---------------------------------------------------------
# 3. Main Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Jetson Text Classification Inference')
    parser.add_argument('--model', type=str, default='text_classifier.pth', help='Path to .pth model file')
    parser.add_argument('--vocab', type=str, default='vocab.json', help='Path to .json vocab file')
    args = parser.parse_args()

    # Device handling
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, word_to_ix = load_data(args.model, args.vocab, device)
    
    print("\n" + "="*40)
    print("      Jetson NLP Sentiment Analyzer      ")
    print("="*40)
    print("Type a sentence to analyze sentiment.")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nInput > ")
            if user_input.lower() in ['quit', 'exit']:
                break
            
            if not user_input.strip():
                continue

            # Preprocess
            text_tensor = prepare_sequence(user_input, word_to_ix).to(device)
            offsets = torch.tensor([0], dtype=torch.long).to(device)
            
            # Infer
            with torch.no_grad():
                output = model(text_tensor, offsets)
                probabilities = torch.softmax(output, dim=1)
                predicted_cls = output.argmax(1).item()
                confidence = probabilities[0][predicted_cls].item()

            sentiment = "POSITIVE :)" if predicted_cls == 1 else "NEGATIVE :("
            color_code = "\033[92m" if predicted_cls == 1 else "\033[91m" # Green or Red
            reset_code = "\033[0m"
            
            print(f"Sentiment: {color_code}{sentiment}{reset_code} ({confidence*100:.1f}%)")

        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
