# src/utils.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_EMOTION_MODEL_DIR = "models/emotion_model"  # after training
# Optionally, start with a pretrained GoEmotions head:
# PRETRAINED = "joeddav/distilbert-base-uncased-go-emotions"

def load_emotion_pipeline(model_path=DEFAULT_EMOTION_MODEL_DIR, device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    id2label = getattr(model.config, "id2label", {i: str(i) for i in range(model.config.num_labels)})
    return tokenizer, model, id2label, device

def predict_emotions(text, tokenizer, model, device, top_k=3, threshold=0.5):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = model(**enc).logits  # (1, num_labels)
    probs = torch.sigmoid(logits).cpu().numpy()[0]  # multi-label
    # top emotions above threshold; fallback to top_k if none
    idxs = np.where(probs >= threshold)[0]
    if len(idxs) == 0:
        idxs = probs.argsort()[-top_k:][::-1]
    scores = probs[idxs]
    return list(zip(idxs.tolist(), scores.tolist()))

def map_emotion_names(idx_scores, id2label):
    return [(id2label.get(i, str(i)), s) for i, s in idx_scores]

def summarize_emotions(named_scores):
    """Return a compact string like: 'anxiety (0.81), sadness (0.64)'"""
    return ", ".join([f"{name} ({score:.2f})" for name, score in named_scores])
