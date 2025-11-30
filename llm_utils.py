import streamlit as st
from groq import Groq

# Load API key & model
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
LLM_MODEL = st.secrets.get("LLM_MODEL", "llama-3.1-8b-instant")


def get_llm_client():
    """Initialize Groq LLM client."""
    if not GROQ_API_KEY:
        print("❌ Missing GROQ_API_KEY in Secrets!")
        return None
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print("Groq init error:", e)
        return None


def generate_answer(client, messages):
    """Generate answer using Groq LLM."""
    if client is None:
        return "LLM is not configured. Missing GROQ_API_KEY."

    # Clean messages for Groq
    cleaned_messages = []
    for m in messages:
        try:
            cleaned_messages.append({
                "role": m["role"],
                "content": str(m["content"])
            })
        except:
            pass

    if not cleaned_messages:
        cleaned_messages = [{"role": "user", "content": "Hello"}]

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=cleaned_messages
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        print("Groq LLM error:", e)
        return "I'm having trouble generating a response."
