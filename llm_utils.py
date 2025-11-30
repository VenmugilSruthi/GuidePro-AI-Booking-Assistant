# llm_utils.py
import streamlit as st
from groq import Groq

# Load Groq key from Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

def get_llm_client():
    if GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)
    return None


def generate_answer(client, messages):
    if client is None:
        return "LLM is not configured. Missing GROQ_API_KEY."

    # ✅ FIX: Clean messages for Groq API
    cleaned_messages = []
    for m in messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            cleaned_messages.append({
                "role": m["role"],
                "content": str(m["content"])  # force clean string
            })

    # If no valid messages, add dummy system prompt
    if not cleaned_messages:
        cleaned_messages = [{"role": "user", "content": "Hello"}]

    try:
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=cleaned_messages
        )
        return resp.choices[0].message["content"]

    except Exception as e:
        print("Groq LLM error:", e)
        return "I'm having trouble generating a response."
