#!/usr/bin/env python3
"""
Test script to verify MindMate components are working correctly.
Run this before starting the Streamlit app to ensure everything is set up.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.append(str(root / "src"))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import streamlit
        from dotenv import load_dotenv
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def test_emotion_detection():
    """Test emotion detection pipeline."""
    print("\nTesting emotion detection...")
    try:
        from utils import load_emotion_pipeline, predict_emotions, map_emotion_names
        
        tokenizer, model, id2label, device = load_emotion_pipeline()
        print(f"✓ Emotion model loaded successfully (device: {device})")
        
        test_text = "I'm feeling anxious about my exam tomorrow"
        idx_scores = predict_emotions(test_text, tokenizer, model, device)
        named = map_emotion_names(idx_scores, id2label)
        
        print(f"  Test input: '{test_text}'")
        print(f"  Detected emotions: {', '.join([f'{name} ({score:.2f})' for name, score in named[:3]])}")
        print("✓ Emotion detection working")
        return True
    except Exception as e:
        print(f"✗ Emotion detection failed: {e}")
        return False

def test_rag_retrieval():
    """Test RAG retrieval system."""
    print("\nTesting RAG retrieval...")
    try:
        from rag_module import RagRetriever
        
        retriever = RagRetriever()
        print(f"✓ RAG retriever initialized with {len(retriever.docs)} knowledge base entries")
        
        query = "I'm feeling anxious"
        results = retriever.search(query, k=3)
        
        print(f"  Test query: '{query}'")
        print(f"  Top result: {results[0]['doc']['title']} (score: {results[0]['score']:.3f})")
        print("✓ RAG retrieval working")
        return True
    except Exception as e:
        print(f"✗ RAG retrieval failed: {e}")
        return False

def test_chatbot():
    """Test chatbot integration."""
    print("\nTesting chatbot...")
    try:
        from chatbot import MindMateBot
        
        bot = MindMateBot(use_openai=False)
        print("✓ Chatbot initialized (fallback mode)")
        
        test_message = "I'm feeling stressed about work"
        result = bot.respond(test_message)
        
        print(f"  Test message: '{test_message}'")
        print(f"  Response preview: {result['answer'][:100]}...")
        print(f"  Emotions detected: {len(result['emotions'])}")
        print(f"  Snippets retrieved: {len(result['retrieval'])}")
        print("✓ Chatbot working")
        return True
    except Exception as e:
        print(f"✗ Chatbot failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("MindMate Component Tests")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Emotion Detection", test_emotion_detection()))
        results.append(("RAG Retrieval", test_rag_retrieval()))
        results.append(("Chatbot", test_chatbot()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You can now run: streamlit run app/app.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before running the app.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
