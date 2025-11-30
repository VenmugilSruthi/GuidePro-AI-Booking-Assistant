import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# modules
from utils import render_chat_bubble
from rag import RAGStore
from booking_flow import start_booking_flow, handle_booking_turn
from email_utils import send_confirmation_email
from db import init_db, add_booking, get_bookings, delete_booking, export_bookings_csv
from hotel_data import hotels
from llm_utils import get_llm_client, generate_answer

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="GuidePro AI", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------
# AUTO-SCROLL + TYPING ANIMATION + FLOAT BUTTON JS
# ----------------------------------------------------------
def inject_js():
    js_code = """
    <script>

    // Auto-scroll function
    function scrollToBottom() {
        const mainDiv = window.parent.document.querySelector('.main');
        if (mainDiv) { mainDiv.scrollTop = mainDiv.scrollHeight; }
    }

    // Call scroll on load
    scrollToBottom();

    // Mutation observer = trigger scroll when new content added
    const observer = new MutationObserver(scrollToBottom);
    observer.observe(window.parent.document.querySelector('.main'), {
        childList: true,
        subtree: true
    });

    // Floating button
    let btn = document.getElementById("floatScrollBtn");
    if (!btn) {
        const b = document.createElement("button");
        b.id = "floatScrollBtn";
        b.innerHTML = "⬇️";
        b.style.position = "fixed";
        b.style.bottom = "20px";
        b.style.right = "20px";
        b.style.padding = "12px 15px";
        b.style.borderRadius = "50%";
        b.style.border = "none";
        b.style.background = "#3A7AFE";
        b.style.color = "white";
        b.style.fontSize = "20px";
        b.style.cursor = "pointer";
        b.style.zIndex = "9999";
        b.onclick = scrollToBottom;
        window.parent.document.body.appendChild(b);
    }

    </script>
    """
    st.markdown(js_code, unsafe_allow_html=True)

# ----------------------------------------------------------
# CSS FOR TYPING + SMOOTH ANIMATION
# ----------------------------------------------------------
st.markdown("""
<style>
.chat-bubble-animate {
    animation: fadeIn 0.4s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to   {opacity: 1; transform: translateY(0);}
}

.typing-indicator {
    display: inline-block;
    padding: 8px 12px;
    background: #e8f0ff;
    border-radius: 10px;
    margin-top: 8px;
}
.typing-indicator span {
    height: 8px;
    width: 8px;
    margin: 0 2px;
    background: #4a86ff;
    display: inline-block;
    border-radius: 50%;
    animation: blink 1.4s infinite both;
}
.typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes blink {
    0% { opacity: .2; }
    20% { opacity: 1; }
    100% { opacity: .2; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# INITIALIZE STATE
# ----------------------------------------------------------
init_db()

if "chat" not in st.session_state:
    st.session_state.chat = [{
        "role": "assistant",
        "content": "Hello! 👋 I’m GuidePro AI. How can I assist your travel today?"
    }]

if "rag" not in st.session_state:
    st.session_state.rag = RAGStore()

if "llm_client" not in st.session_state:
    st.session_state.llm_client = get_llm_client()

if "booking_in_progress" not in st.session_state:
    st.session_state.booking_in_progress = False


# ----------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>GuidePro AI</div>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["Chat Assistant", "Trip Planner", "Hotels Browser", "Admin", "About"])

    st.markdown("<div class='sidebar-section'>Upload PDF for RAG</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF file", type=["pdf"], accept_multiple_files=False)

    if uploaded:
        st.session_state.rag = RAGStore()
        st.session_state.rag.add_pdf(uploaded)
        st.success("PDF uploaded successfully!")


# ----------------------------------------------------------
# CHAT PAGE
# ----------------------------------------------------------
if page == "Chat Assistant":

    inject_js()

    for msg in st.session_state.chat:
        css = "chat-bubble-animate"
        render_chat_bubble(msg, extra_css=css)

    st.markdown("---")

    new_input = st.chat_input("Type your message…")

    if new_input:
        user_text = new_input.lower()
        st.session_state.chat.append({"role": "user", "content": new_input})

        inject_js()
        st.rerun()

        # RAG check
        if len(st.session_state.rag.chunks) > 0:
            rag_match = st.session_state.rag.query(new_input)
            st.session_state.chat.append({"role": "assistant", "content": rag_match})
            inject_js()
            st.rerun()

        # Booking flow
        if start_booking_flow(new_input) or st.session_state.booking_in_progress:
            bot = handle_booking_turn(new_input)
            st.session_state.chat.append({"role": "assistant", "content": bot})
            inject_js()
            st.rerun()

        # Normal LLM reply
        reply = generate_answer(st.session_state.llm_client, st.session_state.chat)
        st.session_state.chat.append({"role": "assistant", "content": reply})
        inject_js()
        st.rerun()


# ----------------------------------------------------------
# TRIP PLANNER
# ----------------------------------------------------------
elif page == "Trip Planner":
    st.header("Trip Planner")
    trip_type = st.selectbox("Trip Type", ["Beach", "City", "Mountain", "International"])
    guests = st.number_input("Guests", 1, 20, 2)
    destination = st.text_input("Destination")
    if st.button("Generate Itinerary"):
        q = f"Create a detailed 3-day {trip_type} trip itinerary for {guests} guests to {destination}."
        st.write(generate_answer(st.session_state.llm_client, [{"role": "user", "content": q}]))


# ----------------------------------------------------------
# HOTELS BROWSER
# ----------------------------------------------------------
elif page == "Hotels Browser":
    st.header("Hotels Browser")
    for h in hotels:
        st.markdown("---")
        st.subheader(h["name"])
        st.write(h["location"])
        try:
            st.image(h["images"][0], width=250)
        except:
            pass


# ----------------------------------------------------------
# ADMIN
# ----------------------------------------------------------
elif page == "Admin":
    st.header("Admin Panel")
    data = get_bookings()
    st.table(data)

    if st.button("Export CSV"):
        export_bookings_csv()
        st.success("Exported!")

    delete_id = st.text_input("Delete booking ID")
    if st.button("Delete"):
        delete_booking(delete_id)
        st.success("Deleted!")


# ----------------------------------------------------------
# ABOUT
# ----------------------------------------------------------
elif page == "About":
    st.header("About GuidePro AI")
    st.write("Your smart AI trip and hotel assistant.")
