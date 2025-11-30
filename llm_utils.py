# llm_utils.py
import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Always use Groq cleanly
def get_llm_client():
    if GROQ_API_KEY:
        return Groq(api_key=GROQ_API_KEY)
    return None


def generate_answer(client, messages):
    if client is None:
        return "LLM client not configured."

    # Always use stable working model
    try:
        resp = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        return resp.choices[0].message["content"]
    except Exception as e:
        print("Groq LLM error:", e)
        return "I'm having trouble generating a response."
