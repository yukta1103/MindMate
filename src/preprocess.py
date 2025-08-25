# src/preprocess.py
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os

def get_tokenizer(model_name="distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_dataset(dataset, tokenizer, max_length=64):
    # First, combine the emotion columns into a multi-hot label
    emotion_cols = [
        'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',
        'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',
        'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',
        'remorse','sadness','surprise','neutral'
    ]
    
    def make_labels(example):
        example["labels"] = [example[col] for col in emotion_cols]
        return example
    
    dataset = dataset.map(make_labels)

    # Tokenize the text
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
    
    tokenized = dataset.map(_tok, batched=True)
    
    # Set format to PyTorch tensors
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized

def collate_fn(features):
    input_ids = torch.stack([f["input_ids"] for f in features])
    attention_mask = torch.stack([f["attention_mask"] for f in features])
    num_labels = 28
    labels = torch.zeros((len(features), num_labels), dtype=torch.float)
    for i, f in enumerate(features):
        for idx in f["labels"]:
            labels[i, idx] = 1.0
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    save_dir = "data/tokenized_goemotions"
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = get_tokenizer()

    # Load full dataset (train only)
    dataset = load_dataset("go_emotions", "raw")["train"]

    # Split train into train (80%), validation (10%), test (10%)
    train_val_test = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_test["train"]
    temp_dataset = train_val_test["test"]
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    splits = {"train": train_dataset, "validation": val_dataset, "test": test_dataset}

    for split_name, ds_split in splits.items():
        print(f"Processing {split_name} split...")
        tokenized = tokenize_dataset(ds_split, tokenizer)
        split_save_path = os.path.join(save_dir, split_name)
        tokenized.save_to_disk(split_save_path)
        print(f"{split_name} split saved to {split_save_path}")

if __name__ == "__main__":
    main()

