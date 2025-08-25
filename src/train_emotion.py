# src/train_emotion.py
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
)
from sklearn.metrics import f1_score
import torch
from preprocess import collate_fn

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Sigmoid for multi-label
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    # labels already multi-hot float32
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}

def prepare_splits(model_name="distilbert-base-uncased"):
    # Load tokenized datasets from disk
    train_ds = load_from_disk("data/tokenized_goemotions/train")
    val_ds   = load_from_disk("data/tokenized_goemotions/validation")
    test_ds  = load_from_disk("data/tokenized_goemotions/test")

    # Load tokenizer (same as used for preprocessing)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return train_ds, val_ds, test_ds, tokenizer

def main():
    model_name = "distilbert-base-uncased"
    num_labels = 28  # raw config: 27 emotions + neutral

    train_ds, val_ds, test_ds, tokenizer = prepare_splits(model_name=model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    training_args = TrainingArguments(
        output_dir="results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="steps",   # now works
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=collate_fn
    )

    trainer.train()
    trainer.save_model("models/emotion_model")
    tokenizer.save_pretrained("models/emotion_model")

if __name__ == "__main__":
    main()
