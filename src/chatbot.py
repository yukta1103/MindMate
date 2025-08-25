# src/chatbot.py
import os
from typing import List, Dict
from dotenv import load_dotenv

from rag_module import RagRetriever
from utils import load_emotion_pipeline, predict_emotions, map_emotion_names, summarize_emotions

# If you plan to use OpenAI for generation:
# pip install openai>=1.0.0  (or your preferred LLM client)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

SYSTEM_PROMPT = (
    "You are MindMate, an empathetic assistant for mental wellbeing. "
    "You must be supportive, respectful, and never provide medical diagnoses. "
    "Base your advice on the provided context snippets. Keep responses concise, practical, and compassionate. "
    "If a user indicates crisis or self-harm, recommend contacting local emergency services or trusted people."
)

def build_prompt(user_msg: str, emotion_labels: List[str], kb_snippets: List[str]) -> List[Dict]:
    context_block = "\n\n".join([f"- {s}" for s in kb_snippets])
    emo = ", ".join(emotion_labels) if emotion_labels else "neutral"
    user_context = (
        f"[Detected emotions: {emo}]\n"
        f"[Helpful snippets:]\n{context_block}\n\n"
        f"User: {user_msg}\n"
        f"Assistant:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_context}
    ]

class MindMateBot:
    def __init__(self, use_openai=True):
        self.tokenizer, self.emotion_model, self.id2label, self.device = load_emotion_pipeline()
        self.retriever = RagRetriever()
        self.use_openai = use_openai and (OpenAI is not None)

        if self.use_openai:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def respond(self, message: str) -> Dict:
        # 1) Emotion detection
        idx_scores = predict_emotions(message, self.tokenizer, self.emotion_model, self.device)
        named = map_emotion_names(idx_scores, self.id2label)
        top_emotions = [n for n, _ in named][:3]

        # 2) RAG
        hits = self.retriever.search(message, k=3)
        kb_snippets = [h["doc"]["content"] for h in hits]

        # 3) LLM generation
        msgs = build_prompt(message, top_emotions, kb_snippets)

        if self.use_openai:
            resp = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=msgs,
                temperature=0.7,
            )
            answer = resp.choices[0].message.content.strip()
        else:
            # Fallback simple template (no external LLM)
            answer = (
                "I hear you. Based on what you shared, here are a couple of practical steps:\n"
                f"- {kb_snippets[0] if kb_snippets else 'Try a brief breathing exercise.'}\n"
                f"- {kb_snippets[1] if len(kb_snippets) > 1 else 'Consider reframing the thought compassionately.'}\n"
                "If this feels overwhelming, it can help to talk with a trusted friend or professional."
            )

        return {
            "answer": answer,
            "emotions": named,
            "retrieval": hits
        }
