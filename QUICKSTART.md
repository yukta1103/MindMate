# MindMate - Quick Start Guide

Welcome to MindMate! This guide will get you up and running in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- Internet connection (for downloading the pre-trained model on first run)
- ~2GB free disk space

## Installation Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: This will install all required packages including PyTorch, Transformers, Streamlit, and more. It may take a few minutes.

### Step 2: Run the Application

```bash
streamlit run app/app.py
```

### Step 3: Wait for Initialization

On first run, you'll see:
```
📥 Downloading pre-trained model: joeddav/distilbert-base-uncased-go-emotions
   (This may take a few minutes on first run...)
```

This is normal! The app is downloading the emotion detection model (~500MB). This only happens once.

You'll also see:
```
📚 Initializing RAG retriever...
   Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
✓ Loaded 18 knowledge base entries
   Building new FAISS index...
✓ RAG retriever ready!
```

### Step 4: Open in Browser

The app will automatically open at `http://localhost:8501`

If it doesn't open automatically, manually navigate to that URL in your browser.

### Step 5: Try It Out!

1. Type a message in the text area, for example:
   - "I'm feeling anxious about my upcoming exam"
   - "I can't sleep at night"
   - "I'm feeling really down lately"

2. Click **Send**

3. View the results:
   - **Response**: Supportive advice
   - **Detected Emotions**: What emotions were detected in your message
   - **Helpful Snippets**: Relevant mental health techniques

## Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "Port 8501 is already in use"
```bash
streamlit run app/app.py --server.port 8502
```

### Slow first run
This is expected! The model download takes 2-5 minutes. Subsequent runs will be much faster.

### Model download fails
- Check your internet connection
- Try: `pip install --upgrade transformers`
- If issues persist, check https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions

## Optional: Better Responses with OpenAI

For higher quality responses:

1. Get an API key from https://platform.openai.com/api-keys
2. Create a `.env` file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your key:
   ```
   OPENAI_API_KEY=sk-your-actual-key-here
   ```
4. Restart the app

## What's Next?

- Read the full [README.md](README.md) for more details
- Explore the knowledge base in `data/mental_health_knowledge.json`
- Consider training a custom emotion model (see README)
- Add more knowledge base entries for your specific needs

## Important Reminder

⚠️ **MindMate is a supportive tool, not a replacement for professional mental health care.**

If you're in crisis:
- **US**: Call 988 (Suicide & Crisis Lifeline)
- **Text**: "HELLO" to 741741 (Crisis Text Line)
- **Emergency**: Call 911 or your local emergency services

---

**Enjoy using MindMate! 💜**
