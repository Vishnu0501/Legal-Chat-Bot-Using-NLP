from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

class LegalDataset(Dataset):
    """
    Dataset for tokenizing and formatting text-label pairs.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze().to(device),
            "attention_mask": encodings["attention_mask"].squeeze().to(device),
            "labels": torch.tensor(label, dtype=torch.long).to(device),
        }

if __name__ == "__main__":
    from src.utils import preprocess_text, extract_text_from_pdf

    # Ensure GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Extract and preprocess text
    pdf_path = "D:/Vishnu files/python_project/LawSmart/Bharatiya Nyaya Sanhita, 2023.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    structured_data = preprocess_text(raw_text)

    # Prepare text and labels
    texts = [item["content"] for item in structured_data]
    labels = list(range(len(structured_data)))

    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    # Use TinyBERT for faster training
    tokenizer = AutoTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    model = AutoModelForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=len(set(labels)))
    model.to(device)  # Move the model to the GPU

    # Prepare datasets
    train_dataset = LegalDataset(train_texts, train_labels, tokenizer)
    val_dataset = LegalDataset(val_texts, val_labels, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="D:/Vishnu files/python_project/LawSmart/models/trained_model",
        num_train_epochs=2,  # Adjust epochs for faster training
        per_device_train_batch_size=32,  # Larger batch size for faster throughput
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="D:/Vishnu files/python_project/LawSmart/logs",
        logging_steps=10,
        save_total_limit=1,
        load_best_model_at_end=True,
        fp16=True,  # Enable mixed precision for faster training on RTX 3050
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    print("Starting training with TinyBERT...")
    trainer.train()

    # Save the model and tokenizer
    print("Training complete. Saving model...")
    model.save_pretrained("D:/Vishnu files/python_project/LawSmart/models/trained_model")
    tokenizer.save_pretrained("D:/Vishnu files/python_project/LawSmart/models/trained_model")
    print("Model and tokenizer saved successfully!")
