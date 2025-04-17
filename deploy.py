from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("D:/Vishnu files/python_project/LawSmart/models/trained_model")
    tokenizer = AutoTokenizer.from_pretrained("D:/Vishnu files/python_project/LawSmart/models/trained_model")
    return model, tokenizer

def predict(question, model, tokenizer, structured_data):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits).item()
    return structured_data[predicted_label]["title"], structured_data[predicted_label]["content"]
