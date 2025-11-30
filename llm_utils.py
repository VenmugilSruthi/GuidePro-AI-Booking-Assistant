# llm_utils.py

import streamlit as st
from groq import Groq
import re

# Load API key & model from Streamlit Secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
LLM_MODEL = st.secrets.get("LLM_MODEL", "llama-3.1-8b-instant")


def get_llm_client():
    """Initialize Groq client."""
    if not GROQ_API_KEY:
        print("❌ Missing GROQ_API_KEY in Secrets!")
        return None

    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        print("🔥 Groq init error:", e)
        return None


def generate_answer(client, messages):
    """Generate answer using Groq LLM with cleaned messages."""

    if client is None:
        return "LLM is not configured. Missing GROQ_API_KEY."

    cleaned_messages = []

    for m in messages:

        # Ignore invalid messages
        if "role" not in m or "content" not in m:
            continue

        content = m["content"]

        # Convert non-string content to text
        if not isinstance(content, str):
            try:
                content = str(content)
            except:
                continue

        # Remove HTML tags, UI components, emojis, and markup
        content = re.sub(r"<.*?>", "", content)  
        content = content.replace("\n", " ").strip()

        # Skip empty strings after cleaning
        if content == "":
            continue

        cleaned_messages.append({
            "role": m["role"],
            "content": content
        })

    # If nothing valid left, add simple fallback
    if not cleaned_messages:
        cleaned_messages = [{"role": "user", "content": "Hello"}]

    # ---- CALL GROQ ----
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=cleaned_messages
        )
        return resp.choices[0].message["content"]

    except Exception as e:
        print("🔥 Groq LLM error:", e)
        return "I'm having trouble generating a response."
