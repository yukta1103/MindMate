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

        api_key = os.getenv("OPENAI_API_KEY")
        if self.use_openai and not api_key:
            print("⚠️ REMINDER: OPENAI_API_KEY not found in .env file. Falling back to template mode.")
            self.use_openai = False
            
        if self.use_openai:
            self.client = OpenAI(api_key=api_key)

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
            # Differentiate response based on top emotion
            primary_emotion = top_emotions[0] if top_emotions else "neutral"
            
            positive_emotions = {"joy", "love", "admiration", "approval", "caring", "excitement", "gratitude", "optimism", "pride", "relief"}
            negative_emotions = {"sadness", "anger", "fear", "annoyance", "disappointment", "grief", "remorse", "nervousness"}
            
            if primary_emotion in positive_emotions:
                answer = (
                    f"That sounds meaningful! It seems like you're feeling {primary_emotion}.\n"
                    "It's great to acknowledge these moments. Is there anything specific that made you feel this way?"
                )
            elif primary_emotion in negative_emotions:
                answer = (
                    f"I hear you. It sounds like you might be dealing with {primary_emotion}. "
                    "Based on what you shared, here are a couple of practical steps that might help:\n"
                    f"- **{kb_snippets[0] if kb_snippets else 'Take a mindful pause'}**: {kb_snippets[0] if kb_snippets else 'Focus on your breath for a few moments.'}\n"
                    f"- **{kb_snippets[1] if len(kb_snippets) > 1 else 'Reflect'}**: {kb_snippets[1] if len(kb_snippets) > 1 else 'Be gentle with yourself right now.'}\n"
                    "\nRemember, I'm here to listen."
                )
            else:
                answer = (
                    "Thank you for sharing that with me. "
                    "How has this been affecting you lately? I'm here to listen if you'd like to explore it further."
                )

        return {
            "answer": answer,
            "emotions": named,
            "retrieval": hits
        }
