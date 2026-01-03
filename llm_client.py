# llm_client.py
import os
import streamlit as st
from groq import Groq

def get_groq_client():
    """
    Securely returns a Groq client instance.
    Checks Streamlit Secrets first (for cloud), then local environment.
    """
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        # Stop the app if no key is found to prevent crashing
        st.error("⚠️ GROQ_API_KEY is missing. Please check your secrets.")
        st.stop()

    return Groq(api_key=api_key)

def call_llm(messages, model="llama-3.3-70b-versatile", temperature=0.3):
    """
    Wrapper to call Groq API easily.
    """
    client = get_groq_client()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API Error: {str(e)}"