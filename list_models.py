# list_models.py
import os
from groq import Groq
from dotenv import load_dotenv

# Load .env if you have it (for local use)
load_dotenv()

def list_groq_models():
    """
    Fetches and prints all available models from Groq.
    Useful for checking if 'llama-3.3-70b' is online or if new models exist.
    """
    try:
        # Tries to get key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ Error: GROQ_API_KEY not found in environment.")
            return

        client = Groq(api_key=api_key)
        models = client.models.list()

        print("\n=== AVAILABLE GROQ MODELS ===")
        for model in models.data:
            print(f"• {model.id}")
        print("=============================\n")

    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    list_groq_models()