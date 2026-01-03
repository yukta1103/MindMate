# app/app.py
import streamlit as st
from pathlib import Path
import sys

# Allow "src" imports
root = Path(__file__).resolve().parents[1]
if str(root / "src") not in sys.path:
    sys.path.append(str(root / "src"))

from chatbot import MindMateBot

# --- Page Config ---
st.set_page_config(
    page_title="MindMate",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Chat Bubbles & Styling ---
st.markdown("""
<style>
    /* Gradient Title */
    .title-text {
        background: linear-gradient(45deg, #A06CD5, #6c5ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        padding-bottom: 20px;
    }
    
    /* Clean up default Streamlit padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Bot & Session State ---
if "bot" not in st.session_state:
    with st.spinner("💜 Awakening MindMate..."):
        st.session_state.bot = MindMateBot(use_openai=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there. I'm MindMate. I'm here to listen and support you. How are you feeling today?"}
    ]

# --- Sidebar: Technical Details ---
with st.sidebar:
    st.title("🧠 MindMate Internals")
    st.markdown("---")
    
    # Placeholder for the latest technical info
    if "latest_analysis" in st.session_state:
        analysis = st.session_state.latest_analysis
        
        st.subheader("🎭 Detected Emotions")
        for name, score in analysis["emotions"]:
            st.progress(score, text=f"{name} ({int(score*100)}%)")
            
        st.markdown("---")
        st.subheader("📚 Knowledge & Advice (RAG)")
        for i, hit in enumerate(analysis["retrieval"], 1):
            with st.expander(f"Snippet #{i} (Confidence: {hit['score']:.2f})"):
                st.info(hit['doc']['title'])
                st.caption(hit['doc']['content'])
    else:
        st.info("Start chatting to see emotion analysis and RAG retrieval in real-time!")

    st.markdown("---")
    st.caption("🔒 Your conversation is private and runs locally (except when using OpenAI).")

# --- Main Chat Interface ---
st.markdown("<h1 class='title-text'>💜 MindMate</h1>", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="💜" if msg["role"] == "assistant" else "👤"):
        st.write(msg["content"])

# Handle User Input
if prompt := st.chat_input("Share what's on your mind..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # 2. Generate Response
    with st.chat_message("assistant", avatar="💜"):
        with st.spinner("Listening compassionately..."):
            result = st.session_state.bot.respond(prompt)
            response = result["answer"]
            
            # Store analysis for sidebar
            st.session_state.latest_analysis = result
            
            st.write(response)
            
    # 3. Append to History
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

