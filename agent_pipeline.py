from report_generator import generate_veritas_response

def run_agent_pipeline(intent, user_text, chat_history, evidence=None):
    """
    Main entry point. Routes the intent to the correct response logic using Groq.
    """
    
    # Case 1: The user wants a forensic report (Action: Analyze)
    # Triggered by the "Run Analysis" button
    if evidence: 
        # We generate a specialized report based on the evidence dictionary
        return generate_veritas_response([], evidence=evidence)

    # Case 2: The user is asking a complex question (Action: Chat)
    # Triggered by the Chat Input bar
    if intent in ["ask_question", "learn_deepfake", "fallback"]:
        # We pass the conversation history so the LLM has context
        return generate_veritas_response(chat_history, evidence=None)

    # Case 3: Simple intents (Action: Static Response)
    # These return None so that app.py can handle them with fast, static text
    return None
