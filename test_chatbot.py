#!/usr/bin/env python3
"""
Quick test of the chatbot functionality without the Streamlit UI.
"""

import sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.append(str(root / "src"))

from chatbot import MindMateBot

def main():
    print("=" * 60)
    print("MindMate Chatbot Test (Fallback Mode)")
    print("=" * 60)
    print()
    
    print("Initializing MindMate bot...")
    bot = MindMateBot(use_openai=False)
    print("✓ Bot initialized successfully!\n")
    
    test_queries = [
        "I'm feeling anxious about my upcoming exam",
        "I can't sleep at night",
        "I'm feeling really down and unmotivated"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}: {query}")
        print("=" * 60)
        
        result = bot.respond(query)
        
        print("\n📝 RESPONSE:")
        print(result["answer"])
        
        print("\n🎭 DETECTED EMOTIONS:")
        emotions_str = ", ".join([f"{name} ({score:.2f})" for name, score in result["emotions"][:3]])
        print(emotions_str if emotions_str else "neutral")
        
        print("\n📚 HELPFUL SNIPPETS (RAG):")
        for hit in result["retrieval"]:
            print(f"  • {hit['doc']['title']} (score: {hit['score']:.3f})")
            print(f"    {hit['doc']['content']}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)
    print("\nYou can now run the full app with: streamlit run app/app.py")

if __name__ == "__main__":
    main()
