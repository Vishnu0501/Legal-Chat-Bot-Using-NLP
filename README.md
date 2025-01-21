This project builds a legal chatbot using the Bharatiya Nyaya Sanhita, 2023 PDF.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt


2. To run the file :

    do this in the cmd set PYTHONPATH=%PYTHONPATH%;path\to\your\project

3. Commands to be run

    python src/preprocess.py # to convert the pdf into data as txt

    python src/train.py # to train the model

    python src/evaluate.py  # to evaluate the model

    python src/deploy.py  # to deploy the model

    uvicorn main:app --reload to start the chat bot
    