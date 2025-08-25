# app/app.py
import streamlit as st
from pathlib import Path
import sys

# Allow "src" imports when running `streamlit run app/app.py`
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

from chatbot import MindMateBot

st.set_page_config(page_title="MindMate", page_icon="💜", layout="centered")
st.title("💜 MindMate — Empathetic AI for Mental Wellbeing")

if "bot" not in st.session_state:
    st.session_state.bot = MindMateBot(use_openai=True)  # set False to avoid OpenAI

user_msg = st.text_area("Share what's on your mind:", height=120, placeholder="e.g., I'm anxious about my upcoming exam...")

if st.button("Send") and user_msg.strip():
    with st.spinner("Thinking compassionately..."):
        result = st.session_state.bot.respond(user_msg)

    st.subheader("Response")
    st.write(result["answer"])

    st.subheader("Detected Emotions")
    emo_str = ", ".join([f"{name} ({score:.2f})" for name, score in result["emotions"]])
    st.write(emo_str if emo_str else "neutral")

    st.subheader("Helpful Snippets (RAG)")
    for hit in result["retrieval"]:
        st.markdown(f"**{hit['doc']['title']}** — {hit['doc']['content']}  \n*score:* {hit['score']:.3f}")
