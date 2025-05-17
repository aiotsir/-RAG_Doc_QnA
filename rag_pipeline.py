from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.schema import Document
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  #  Load variables from .env

# Initialize Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_model = None

# Initialize embedding model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading Sentence Transformer model: {e}")
    embedding_model = None
    embedding_fn = None

# Initialize ChromaDB client and collection
try:
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="rag_collection", embedding_function=embedding_fn)
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    client = None
    collection = None

def create_vector_store(chunks):
    if collection and embedding_fn:
        for chunk in chunks:
            try:
                collection.add(
                    documents=[chunk.page_content],
                    metadatas=[chunk.metadata],
                    ids=[chunk.metadata.get("id", str(hash(chunk.page_content)))]
                )
            except Exception as e:
                print(f"Error adding chunk to ChromaDB: {e}")
    else:
        print("ChromaDB collection or embedding function not initialized.")

def retrieve_similar_docs(query, k=3):
    if collection:
        try:
            results = collection.query(query_texts=[query], n_results=k)
            return [r for r in results.get("documents", [[]])[0]]
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return []
    else:
        print("ChromaDB collection not initialized.")
        return []

def generate_answer(query, context):
    if gemini_model:
        full_prompt = f"""You are a helpful assistant. Use the below context to answer the question.

Context:
{context}

Question:
{query}
"""
        try:
            response = gemini_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Error generating answer with Gemini: {e}")
            return "Error generating answer."
    else:
        return "Gemini model not initialized."

if __name__ == "__main__":
    # Example usage (for testing purposes)
    from utils import load_pdf_texts, split_into_chunks

    # Create a dummy documents folder and PDF for testing
    os.makedirs("documents", exist_ok=True)
    with open("documents/dummy.pdf", "w") as f:
        f.write("This is a dummy PDF document for testing.\nIt has multiple lines of text.")

    docs = load_pdf_texts("documents/")
    chunks = split_into_chunks(docs)
    create_vector_store(chunks)

    query = "What is this document about?"
    similar_docs = retrieve_similar_docs(query)
    print("\nSimilar Documents:")
    for doc in similar_docs:
        print(doc)

    if similar_docs:
        context = "\n\n".join(similar_docs)
        answer = generate_answer(query, context)
        print("\nAnswer:")
        print(answer)

    # Clean up dummy file and folder
    os.remove("documents/dummy.pdf")
    os.rmdir("documents")