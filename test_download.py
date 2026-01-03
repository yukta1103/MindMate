
from transformers import AutoTokenizer, AutoModelForSequenceClassification

models_to_test = [
    "joeddav/distilbert-base-uncased-go-emotions-student",
    "SamLowe/roberta-base-go_emotions"
]

for model_name in models_to_test:
    print(f"Testing {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print(f"SUCCESS: {model_name}")
        break
    except Exception as e:
        print(f"FAILED: {model_name} with error {e}")
