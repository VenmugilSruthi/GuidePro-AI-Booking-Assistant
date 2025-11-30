import streamlit as st
import html

def render_chat_bubble(message, extra_css=""):
    role = message["role"]
    content = html.escape(message["content"])

    if role == "user":
        bubble_class = "chat-user"
        align = "flex-end"
    else:
        bubble_class = "chat-bot"
        align = "flex-start"

    # Add animation css class if provided
    if extra_css:
        bubble_class += " " + extra_css

    st.markdown(f"""
    <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
        <div class="{bubble_class}">
            {content}
        </div>
    </div>
    """, unsafe_allow_html=True)
