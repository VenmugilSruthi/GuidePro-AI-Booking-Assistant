import streamlit as st
from groq import Groq

# Load Groq LLM API key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

# Load LLM model
LLM_MODEL = st.secrets.get("LLM_MODEL", "llama3-8b-8192")


def get_llm_client():
    """Initialize and return the Groq LLM client."""
    if not GROQ_API_KEY:
        print("❌ ERROR: GROQ_API_KEY missing in Streamlit Secrets")
        return None
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print("❌ Failed to initialize Groq client:", e)
        return None


def generate_answer(client, messages):
    """Send messages to the Groq LLM and return a response."""
    if client is None:
        return "❌ LLM not configured. Missing GROQ_API_KEY."

    # --- Prepare cleaned messages ---
    cleaned_messages = []
    for m in messages:
        if isinstance(m, dict) and "role" in m and "content" in m:
            cleaned_messages.append({
                "role": m["role"],
                "content": str(m["content"])
            })

    if not cleaned_messages:
        cleaned_messages = [{"role": "user", "content": "Hello"}]

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=cleaned_messages
        )

        return resp.choices[0].message["content"]

    except Exception as e:
        print("🔥 Groq Error:", e)
        return f"🔥 ERROR FROM LLM: {str(e)}"
