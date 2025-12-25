# src/utils.py
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_EMOTION_MODEL_DIR = "models/emotion_model"  # after training
# Pre-trained model as fallback
PRETRAINED_MODEL = "joeddav/distilbert-base-uncased-go-emotions"

def load_emotion_pipeline(model_path=DEFAULT_EMOTION_MODEL_DIR, device=None):
    """Load emotion detection model. Uses pre-trained model if local model doesn't exist."""
    
    # Check if local trained model exists
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
        print(f"✓ Loading local emotion model from {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"✗ Error loading local model: {e}")
            print(f"Falling back to pre-trained model...")
            tokenizer = None
            model = None
    else:
        tokenizer = None
        model = None
    
    # Load pre-trained model if local model failed or doesn't exist
    if tokenizer is None or model is None:
        print(f"📥 Downloading pre-trained model: {PRETRAINED_MODEL}")
        print("   (This may take a few minutes on first run...)")
        try:
            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
            print("✓ Pre-trained model loaded successfully!")
            print("   (You can train a custom model later using: python src/train_emotion.py)")
        except Exception as e:
            print(f"\n✗ ERROR: Failed to load emotion detection model")
            print(f"   Details: {e}")
            print(f"\n   Troubleshooting:")
            print(f"   1. Check your internet connection")
            print(f"   2. Try: pip install --upgrade transformers")
            print(f"   3. Visit: https://huggingface.co/{PRETRAINED_MODEL}")
            raise
    
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✓ Model running on: {device}")
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
