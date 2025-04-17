from src.utils import extract_text_from_pdf , preprocess_text , save_preprocessed_data
# from src.utils import extract_text_from_pdf, preprocess_text, save_preprocessed_data
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == "__main__":
    pdf_path = "D:/Vishnu files/python_project/LawSmart/Bharatiya Nyaya Sanhita, 2023.pdf"
    raw_text = extract_text_from_pdf(pdf_path)
    structured_data = preprocess_text(raw_text)

    os.makedirs("D:/Vishnu files/python_project/LawSmart/data/", exist_ok=True)
    save_preprocessed_data(structured_data, "D:/Vishnu files/python_project/LawSmart/data/structured_data.txt")

    print("Text extraction and preprocessing complete. Preprocessed data saved to `data/structured_data.txt`.")
