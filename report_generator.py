# report_generator.py

# 👇 CHANGE: We now import from the updated llm_utils
from llm_utils import generate_text
from prompt_templates import SYSTEM_IDENTITY, FORENSIC_REPORT_TEMPLATE

def generate_veritas_response(messages, evidence: dict | None = None):
    """
    Generates a response using Groq (via llm_utils).
    """
    
    # --- 1. DETERMINE SYSTEM PROMPT ---
    if evidence:
        # Report Mode
        score = evidence.get('score', 0.0)
        anomalies = ", ".join(evidence.get('anomalies', [])) or "None detected"
        verdict = "HIGHLY SUSPICIOUS" if score > 60 else "LIKELY AUTHENTIC"
        
        system_content = FORENSIC_REPORT_TEMPLATE.format(
            system_identity=SYSTEM_IDENTITY,
            verdict=verdict,
            score=score,
            anomalies=anomalies
        )
    else:
        # Chat Mode
        system_content = SYSTEM_IDENTITY

    # --- 2. PREPARE MESSAGES ---
    formatted_messages = [{"role": "system", "content": system_content}] + messages

    # --- 3. CALL GROQ ---
    # 👇 CHANGE: We use the function name from llm_utils
    return generate_text(formatted_messages)