import streamlit as st

def render_chat_bubble(message, extra_css=""):
    """
    Renders a styled chat bubble (user or bot) with appropriate alignment, 
    an emoji, and converts Markdown bold (**) to HTML bold (<b>).
    
    Args:
        message (dict): A dictionary with "role" ("user" or "assistant") and "content".
        extra_css (str): Optional string for adding animation or other temporary CSS classes.
    """
    role = message["role"]
    
    # Use raw content
    content = message["content"] 
    
    # 🌟 FIX: Convert Markdown bold (**) to HTML bold (<b>)
    # This is a simple replacement that works for one pair of **...** in a string.
    content = content.replace('**', '<b>', 1).replace('**', '</b>', 1) 

    if role == "user":
        bubble_class = "chat-user"
        align = "flex-end"
        # 👤 User message format: Content Only
        rendered_content = content
    else:
        bubble_class = "chat-bot"
        align = "flex-start"
        
        # 🤖 Bot message format: Bot Emoji + Content (now correctly bolded)
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
