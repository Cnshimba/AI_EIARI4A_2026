import requests
import json
import argparse

# Configuration
DEFAULT_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3.2"

def ask_local_llm(prompt, model=DEFAULT_MODEL, url=DEFAULT_URL):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Sending request to {model}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get('response', 'No response field in JSON')
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Is it running? (run 'ollama serve')"
    except Exception as e:
        return f"Error: {e}"

def main():
    parser = argparse.ArgumentParser(description='Jetson Edge LLM Client')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Ollama model name')
    args = parser.parse_args()

    print("========================================")
    print(f"   Jetson Edge AI Client ({args.model})   ")
    print("========================================")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            if not user_input.strip():
                continue

            answer = ask_local_llm(user_input, model=args.model)
            print(f"AI: {answer}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
