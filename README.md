
# Agentic RAG System with Local LLMs

(Document-Grounded Question Answering)

This project implements a **fully local, agentic Retrieval-Augmented Generation (RAG) system** that answers user queries strictly from uploaded documents. It combines semantic search with a local Large Language Model to reduce hallucinations and simulate real-world enterprise AI systems.

The project was designed with **placements and industry relevance** in mind, focusing on system design, performance trade-offs, and modular AI agents rather than just model usage.

---

## Key Features

* Fully local RAG system (no paid APIs)
* Dynamic PDF upload with automatic re-indexing
* Semantic search using FAISS
* Query rewriting agent for better retrieval
* Optional hallucination validation agent
* Conversation memory for follow-up questions
* Cached embeddings for performance
* Modular design with feature toggles
* Web-based chat interface

---

## Tech Stack

* Python
* Flask (backend server)
* PyPDF (PDF text extraction)
* SentenceTransformers (all-MiniLM-L6-v2) for embeddings
* FAISS for vector similarity search
* Mistral-7B (GGUF) via ctransformers for local LLM inference
* HTML + CSS for frontend UI

---

## System Workflow

1. PDFs are uploaded via the web interface or placed in `data/docs`
2. Text is extracted and split into overlapping chunks
3. Each chunk is converted into a vector embedding
4. Embeddings are stored in a FAISS vector index
5. User queries are optionally rewritten for better retrieval
6. Relevant document chunks are retrieved using semantic search
7. The local LLM generates answers strictly from retrieved context
8. Conversation memory enables contextual multi-turn Q&A
9. New document uploads trigger re-indexing without reloading the model

---

## Performance Notes

* The system runs **Mistral-7B on CPU**, which is intentionally slow but realistic for local deployment
* Architecture supports faster models or GPU inference without structural changes
* Embeddings and vector indexes are cached to avoid recomputation

---

## Project Structure

```
RAG/
├── app.py
├── src/
│   └── rag_core.py
├── templates/
│   └── index.html
├── data/
│   └── docs/
├── models/        # model files (not committed)
├── cache/         # vector cache (ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup Instructions

1. Create a virtual environment and install dependencies
2. Download a GGUF version of Mistral-7B and place it in `models/`
3. Run the application:

   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in the browser
5. Upload PDFs and start asking questions

Model files are not included in the repository due to size constraints.

---

## What This Project Demonstrates

* Practical understanding of RAG architecture
* Handling hallucinations in LLM systems
* Trade-offs between latency and accuracy
* Real-world GenAI system design
* Clean, modular, and scalable code structure


