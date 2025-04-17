from transformers import Trainer
from src.train import LegalDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from src.utils import preprocess_text, extract_text_from_pdf

    pdf_path = "D:/Vishnu files/python_project/LawSmart/Bharatiya Nyaya Sanhita, 2023.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    structured_data = preprocess_text(raw_text)

    texts = [item["content"] for item in structured_data]
    labels = list(range(len(structured_data)))

    tokenizer = AutoTokenizer.from_pretrained("D:/Vishnu files/python_project/LawSmart/models/trained_model")
    model = AutoModelForSequenceClassification.from_pretrained("D:/Vishnu files/python_project/LawSmart/models/trained_model")

    dataset = LegalDataset(texts, labels, tokenizer)

    trainer = Trainer(
        model=model,
        eval_dataset=dataset,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate()
    print("Evaluation Results:", results)
