import joblib
import os

# --- Configuration ---
MODEL_FILE = "intent_classifier.pkl"
CONFIDENCE_THRESHOLD = 0.4 # If confidence is lower than 40%, we say "fallback"

def load_model():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: '{MODEL_FILE}' not found. Run train_classifier.py first.")
        return None
    print(f"Loading model from {MODEL_FILE}...")
    return joblib.load(MODEL_FILE)

def get_prediction(model, user_text):
    # The pipeline handles TF-IDF automatically. We just pass raw text.
    # predict_proba gives us the percentage scores for all intents
    probabilities = model.predict_proba([user_text])[0]
    
    # Find the highest score
    max_index = probabilities.argmax()
    confidence = probabilities[max_index]
    predicted_intent = model.classes_[max_index]
    
    return predicted_intent, confidence

def main():
    model = load_model()
    if not model:
        return

    print("\n" + "="*50)
    print("VERITAS AGENT - INTENT CLASSIFIER DEMO")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if not user_input.strip():
            continue

        intent, confidence = get_prediction(model, user_input)
        
        # --- THE LOGIC YOUR CHATBOT WILL USE ---
        print("-" * 30)
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"🤖 System: I'm not sure what you mean (Confidence: {confidence:.2f})")
            print(f"   -> Suggested Fallback: 'I didn't catch that. Can you rephrase?'")
        else:
            print(f"🤖 Predicted Intent:  [{intent.upper()}]")
            print(f"📊 Confidence Score:  {confidence*100:.1f}%")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    main()