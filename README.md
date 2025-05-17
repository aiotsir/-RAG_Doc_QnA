# -RAG_Doc_QnA
RAG-based PDF Q&amp;A app using SentenceTransformers, ChromaDB &amp; Gemini, with real-time semantic retrieval and LLM generation.
Ask questions and get answers from your pdf docs.

## Metrics
| Item                   | Value              |
| ---------------------- | ------------------ |
| Chunk size             | 300 characters     |
| Overlap between chunks | 50 characters      |
| Retrieval Top-k        | 3 chunks           |
| LLM used               | Gemini 2.0 Flash   |
| Embedding model        | MiniLM-L6-v2       |
| Max tested docs/pages  | \~100 pages (fast) |

-----------------------------------------------------------------------------------
Setup Instructions
1. Clone the Repo.
2. Create Environment & Install Dependencies.
3. Configure Your Gemini API Key
      -Create a .env file in the root directory with your Gemini key.
       I haven't exposed my dot env file for it has the API key.
4. Run the App - streamlit run app.py

-------------------------------------------------------------------------------------

## Tech Stack
| Component   | Tool / Library                   |
| ----------- | -------------------------------- |
| Embeddings  | SentenceTransformers (MiniLM-L6) |
| Vector DB   | ChromaDB                         |
| LLM         | Google Gemini (gemini-2.0-flash) |
| PDF Reader  | PyPDF2                           |
| Chunking    | LangChain                        |
| UI          | Streamlit                        |
| Environment | Python 3.8+                      |

