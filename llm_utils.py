# llm_utils.py
import os
import streamlit as st
from groq import Groq

def get_groq_client():
    """
    Securely creates a Groq client.
    - Checks Streamlit Cloud Secrets first (Production)
    - Checks local .env second (Development)
    """
    api_key = None
    
    # 1. Try Streamlit Secrets
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        pass

    # 2. Try Local Environment
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    # 3. Handle Missing Key
    if not api_key:
        # We don't stop the app here to allow UI to handle the error gracefully
        raise ValueError("GROQ_API_KEY not found. Please set it in Secrets or .env")

    return Groq(api_key=api_key)

def generate_text(messages, model="llama-3.3-70b-versatile", temperature=0.3):
    """
    Simple wrapper to generate text from Groq.
    """
    try:
        client = get_groq_client()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ API Connection Error: {str(e)}"  