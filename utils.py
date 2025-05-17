from langchain.schema import Document
from pathlib import Path
from PyPDF2 import PdfReader

def load_pdf_texts(folder_path):
    documents = []
    pdf_files = list(Path(folder_path).glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{folder_path}'.")
        return documents
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append(Document(page_content=text, metadata={"source": str(pdf_path), "id": str(pdf_path)}))
        except Exception as e:
            print(f"Error reading PDF '{pdf_path}': {e}")
    return documents

def split_into_chunks(documents, chunk_size=300, overlap=50):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)

import os
from pathlib import Path
import shutil

def save_uploaded_files(uploaded_files, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for file in uploaded_files:
        file_path = os.path.join(save_dir, file.name)
        try:
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        except Exception as e:
            print(f"Error saving file '{file.name}': {e}")

def clear_temp_uploads(folder="temp_uploads"):
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
        except Exception as e:
            print(f"Error clearing temporary uploads folder '{folder}': {e}")