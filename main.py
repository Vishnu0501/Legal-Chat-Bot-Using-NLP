from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.deploy import load_model, predict
from src.utils import extract_text_from_pdf, preprocess_text

app = FastAPI()

model, tokenizer = load_model()

pdf_path = "D:/Vishnu files/python_project/LawSmart/Bharatiya Nyaya Sanhita, 2023.pdf"
raw_text = extract_text_from_pdf(pdf_path)
structured_data = preprocess_text(raw_text)

class LegalQuery(BaseModel):
    question: str

@app.post("/ask")
async def ask_legal_question(query: LegalQuery):
    title, content = predict(query.question, model, tokenizer, structured_data)
    return {"section": title, "content": content[:300]}  # Return the first 300 characters of content

