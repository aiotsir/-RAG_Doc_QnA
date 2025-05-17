import streamlit as st
from rag_pipeline import create_vector_store, retrieve_similar_docs, generate_answer
from utils import load_pdf_texts, split_into_chunks, save_uploaded_files, clear_temp_uploads
import os
import tempfile

st.title("🔍 Gemini-Powered RAG Assistant")

# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        temp_dir = "temp_uploads"
        save_uploaded_files(uploaded_files, temp_dir)
        st.info(f"{len(uploaded_files)} file(s) uploaded to '{temp_dir}'.")

        if st.button("Process Uploaded Documents"):
            docs = load_pdf_texts(temp_dir)
            chunks = split_into_chunks(docs)
            create_vector_store(chunks)
            clear_temp_uploads()
            st.success("Documents processed and embedded!")

# --- Main Area for Question Answering ---
st.header("Ask Questions")
query = st.text_input("Enter your question:")

if query:
    similar_docs = retrieve_similar_docs(query)
    if similar_docs:
        st.subheader("Retrieved Context:")
        for doc in similar_docs:
            st.markdown(f"- {doc}")
        context = "\n\n".join(similar_docs)
        answer = generate_answer(query, context)
        st.subheader("Answer:")
        st.markdown(answer)
    else:
        st.warning("No relevant documents found.")