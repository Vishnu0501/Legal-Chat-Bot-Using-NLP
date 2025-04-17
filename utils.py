import PyPDF2
import re

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

def preprocess_text(text):
    """
    Clean and preprocess raw text into structured sections.
    """
    cleaned_text = re.sub(r"\s+", " ", text)
    sections = re.split(r"(Section \d+|Chapter \d+)", cleaned_text)
    data = []
    for i in range(1, len(sections), 2):
        title = sections[i]
        content = sections[i + 1] if i + 1 < len(sections) else ""
        data.append({"title": title.strip(), "content": content.strip()})
    return data

def save_preprocessed_data(data, output_path):
    """
    Save preprocessed data to a file for inspection or later use.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(f"{item['title']}\n{item['content']}\n\n")
