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

    try:
        resp = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        print("Groq LLM error:", e)
        return "I'm having trouble generating a response."
