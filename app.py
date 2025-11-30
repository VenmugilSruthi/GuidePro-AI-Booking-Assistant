import os
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# local modules
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
st.set_page_config(
    page_title="GuidePro AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------------------------------------
# PREMIUM UI
# ----------------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background-color: #c5d5c5;
    padding-top: 25px;
}

.hero-bg {
    width: 100%; height: 430px;
    background-image: url('https://images.unsplash.com/photo-1506744038136-46273834b3fb');
    background-size: cover;
    background-position: center;
}

.hero-card {
    position: absolute;
    top: 62%; left: 50%;
    transform: translate(-50%, -50%);
    width: 70%;
    background: rgba(255,255,255,0.45);
    padding: 30px 50px;
    border-radius: 20px;
    text-align: center;
    backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)



# ----------------------------------------------------------
# INITIALIZE SESSION STATE
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

    page = st.radio("Navigate",
                    ["Chat Assistant", "Trip Planner", "Hotels Browser", "Admin", "About"])

    st.markdown("<div class='sidebar-section'>Upload PDFs for RAG</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        st.session_state.rag.add_documents(uploaded)
        st.success("PDF(s) indexed successfully!")



# ----------------------------------------------------------
# CHAT PAGE
# ----------------------------------------------------------
if page == "Chat Assistant":

    st.markdown("""
        <div class="hero-bg">
            <div class="hero-card">
                <h1 class="hero-title">GuidePro AI</h1>
                <p class="hero-sub">Your personal AI for trips, hotels & smart planning.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.chat:
        render_chat_bubble(msg)

    st.markdown("---")

    user_input = st.chat_input("Type your message…")

    if user_input:

        st.session_state.chat.append({"role": "user", "content": user_input})

        text = user_input.lower()

        # ------------------------------------------------------
        # 1️⃣ RAG SIMILARITY CHECK (IMPROVED)
        # Always runs if PDF exists
        # ------------------------------------------------------
        rag_used = False
        if len(st.session_state.rag.embeddings) > 0:
            rag_answer = st.session_state.rag.query(user_input)
            # If answer comes from RAG and has similarity
            if rag_answer and len(rag_answer.strip()) > 0:
                rag_used = True
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": rag_answer
                })
        
        # ------------------------------------------------------
        # 2️⃣ BOOKING FLOW
        # ------------------------------------------------------
        booking_keywords = ["book", "reservation", "reserve"]

        if not rag_used and (any(k in text for k in booking_keywords) or st.session_state.booking_in_progress):
            st.session_state.booking_in_progress = True
            resp = handle_booking_turn(user_input)
            st.session_state.chat.append({"role": "assistant", "content": resp})
            st.rerun()

        # ------------------------------------------------------
        # 3️⃣ LLM FALLBACK
        # ------------------------------------------------------
        if not rag_used:
            llm_reply = generate_answer(st.session_state.llm_client, st.session_state.chat)
            st.session_state.chat.append({
                "role": "assistant",
                "content": llm_reply
            })

        st.rerun()



# ----------------------------------------------------------
# OTHER PAGES
# ----------------------------------------------------------
elif page == "Trip Planner":
    st.header("Trip Planner")
    trip_type = st.selectbox("Trip Type", ["Beach", "City", "Mountain", "International"])
    guests = st.number_input("Guests", 1, 20, 2)
    destination = st.text_input("Destination")
    if st.button("Generate Itinerary"):
        query = f"Create a detailed 3-day {trip_type} trip itinerary for {guests} guests to {destination}."
        reply = generate_answer(st.session_state.llm_client, [{"role": "user", "content": query}])
        st.write(reply)

elif page == "Hotels Browser":
    st.header("Hotels Browser")
    for h in hotels:
        st.markdown("---")
        st.subheader(h["name"])
        st.write(h["location"])
        st.image(h["images"][0], width=250)

elif page == "Admin":
    st.header("Admin Panel")
    data = get_bookings()
    st.table(data)

    if st.button("Export as CSV"):
        export_bookings_csv()
        st.success("Exported bookings.csv!")

    del_id = st.text_input("Delete booking by ID")
    if st.button("Delete Booking"):
        delete_booking(del_id)
        st.success("Booking Deleted")

elif page == "About":
    st.header("About GuidePro AI")
    st.write("Your smart AI travelling assistant.")
