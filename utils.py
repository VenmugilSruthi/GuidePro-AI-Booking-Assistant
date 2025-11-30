import streamlit as st

# NOTE: html import is removed as we are no longer using html.escape()
# Removing html.escape allows <b> tags in the message content for bolding.

def render_chat_bubble(message, extra_css=""):
    """
    Renders a styled chat bubble (user or bot) with appropriate alignment and an emoji.
    
    Args:
        message (dict): A dictionary with "role" ("user" or "assistant") and "content".
        extra_css (str): Optional string for adding animation or other temporary CSS classes.
    """
    role = message["role"]
    
    # Use raw content to allow HTML formatting (like <b> for bolding)
    content = message["content"] 

    if role == "user":
        bubble_class = "chat-user"
        align = "flex-end"
        # User message format: Content Only (Optional: add emoji if desired, but typically only bot has one)
        # We will keep the content clean for the user side.
        rendered_content = content
    else:
        bubble_class = "chat-bot"
        align = "flex-start"
        
        # Bot message format: Bot Emoji + Content (including <b> for bolded prompts)
        emoji = "🤖" 
        rendered_content = f"{emoji} {content}"

    # Add animation css class if provided
    if extra_css:
        bubble_class += " " + extra_css

    st.markdown(f"""
    <div style="display: flex; justify-content: {align}; margin-bottom: 10px;">
        <div class="{bubble_class}">
            {rendered_content}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Important Note: To see the bolding (e.g., **full name**), 
# the source that generates the message must now use HTML tags:
# e.g., "What is your **<b>full name</b>**?"
