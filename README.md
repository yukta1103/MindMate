# MindMate 💜
**Empathetic Conversational AI for Mental Wellbeing**

MindMate is an AI-powered mental health support chatbot that combines emotion detection, knowledge retrieval (RAG), and empathetic responses to provide compassionate support for mental wellbeing.

## ⚠️ Important Disclaimer

**MindMate is NOT a replacement for professional mental health care.** If you're experiencing a mental health crisis, please contact:
- **US**: 988 (Suicide & Crisis Lifeline) or text "HELLO" to 741741
- **Emergency**: Call 911 or your local emergency services
- **Professional Help**: Consult a licensed mental health professional

## ✨ Features

- 🎭 **Emotion Detection**: Identifies emotions in user messages using fine-tuned NLP models
- 📚 **Knowledge Retrieval (RAG)**: Retrieves relevant mental health tips and techniques
- 💬 **Empathetic Responses**: Generates compassionate, supportive responses
- 🔄 **Dual Mode**: Works with or without OpenAI API (fallback mode available)
- 🎨 **User-Friendly Interface**: Clean Streamlit web interface

## 🚀 Quick Start (Minimum Working Model)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MindMate.git
cd MindMate
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app/app.py
```

The app will automatically:
- Use a pre-trained emotion detection model (no training required!)
- Build the knowledge base index
- Run in fallback mode (template-based responses)

### 5. Open in Browser
The app should automatically open at `http://localhost:8501`

## 🎯 Usage

1. Type your thoughts or feelings in the text area
2. Click "Send"
3. View:
   - **Response**: Empathetic, supportive advice
   - **Detected Emotions**: Top emotions identified in your message
   - **Helpful Snippets**: Relevant mental health techniques from the knowledge base

### Example Queries
- "I'm feeling anxious about my upcoming exam"
- "I can't sleep at night and I'm exhausted"
- "I'm feeling really down and unmotivated lately"
- "I'm overwhelmed with work and don't know where to start"

## 🔧 Configuration

### Optional: OpenAI Integration (Better Responses)

For higher quality, more personalized responses:

1. Get an API key from [OpenAI](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root:
```bash
cp .env.example .env
```
3. Edit `.env` and add your API key:
```
OPENAI_API_KEY=your_actual_api_key_here
OPENAI_MODEL=gpt-4o-mini
```
4. Restart the application

### Optional: Train Custom Emotion Model

To train your own emotion detection model instead of using the pre-trained one:

```bash
# 1. Preprocess the GoEmotions dataset
python src/preprocess.py

# 2. Train the model (takes 30-60 minutes)
python src/train_emotion.py
```

The trained model will be saved to `models/emotion_model/` and automatically used on next run.

## 📁 Project Structure

```
MindMate/
├── app/
│   └── app.py                 # Streamlit web interface
├── src/
│   ├── chatbot.py            # Main chatbot logic
│   ├── rag_module.py         # RAG retrieval system
│   ├── utils.py              # Emotion detection utilities
│   ├── preprocess.py         # Dataset preprocessing
│   └── train_emotion.py      # Model training script
├── data/
│   ├── mental_health_knowledge.json  # Knowledge base (18 entries)
│   └── tokenized_goemotions/         # Preprocessed data (created by preprocess.py)
├── models/
│   └── emotion_model/        # Trained model (created by train_emotion.py)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 🛠️ Troubleshooting

### "No module named 'openai'"
```bash
pip install openai>=1.0.0
```

### "Local model not found"
This is normal! The app will automatically use the pre-trained model. No action needed.

### Slow first run
The first run downloads the pre-trained model and builds the FAISS index. Subsequent runs will be faster.

### OpenAI API errors
- Check your `.env` file has the correct API key
- Verify your OpenAI account has credits
- The app will fall back to template mode if OpenAI fails

### Port already in use
```bash
streamlit run app/app.py --server.port 8502
```

## 🔮 Future Enhancements

- [ ] Conversation history and context
- [ ] Export conversation transcripts
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Mobile app version
- [ ] Integration with mental health resources database
- [ ] Mood tracking over time

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **GoEmotions Dataset**: Google Research
- **Pre-trained Model**: [joeddav/distilbert-base-uncased-go-emotions](https://huggingface.co/joeddav/distilbert-base-uncased-go-emotions)
- **Frameworks**: Hugging Face Transformers, Streamlit, FAISS

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Remember**: MindMate is a supportive tool, not a substitute for professional mental health care. Always seek help from qualified professionals when needed. 💜
