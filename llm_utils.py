# llm_utils.py
import streamlit as st
from groq import Groq

# Load Groq API key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# Load model name from secrets, fallback to a valid Groq model
LLM_MODEL = st.secrets.get("LLM_MODEL", "llama-3.1-8b-instant")


def get_llm_client():
    """Initialize Groq client."""
    if GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)
    return None


def generate_answer(client, messages):
    """Send cleaned messages to Groq LLM."""
    if client is None:
        return "LLM is not configured. Missing GROQ_API_KEY."

    # --- CLEAN messages before sending to Groq ---
    cleaned_messages = []
    for m in messages:
        # Streamlit sometimes sends system objects, ignore them
        if isinstance(m, dict) and "role" in m and "content" in m:
            cleaned_messages.append({
                "role": m["role"],
                "content": str(m["content"])
            })

    # If empty, avoid Groq error
    if not cleaned_messages:
        cleaned_messages = [{"role": "user", "content": "Hello"}]

    # --- SEND TO GROQ ---
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=cleaned_messages
        )
        return resp.choices[0].message["content"]

    except Exception as e:
        # Print error to Streamlit logs
        print("Groq LLM error:", e)
        return "I'm having trouble generating a response."
