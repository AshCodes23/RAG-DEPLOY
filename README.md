
# Agentic RAG System | Gemini Powered

(Document-Grounded Question Answering)

This project implements a **modern, agentic Retrieval-Augmented Generation (RAG) system** using the **Google Gemini API**. It combines semantic search with advanced LLM reasoning to provide accurate, document-grounded answers while minimizing hallucinations.

---

## Key Features

* **Cloud-Powered Intelligence**: Leveraging Gemini Pro for reasoning and Gemini Embedding for semantic search.
* **Agentic Workflow**: Includes query rewriting and optional answer validation agents.
* **Dynamic PDF Hub**: Upload a PDF via the web UI and the system automatically re-indexes its knowledge.
* **Semantic Vector Store**: Uses FAISS for high-performance similarity search.
* **Conversation Memory**: Remembers context across multiple turns for intuitive follow-up questions.
* **Premium UI**: Dark-themed, glassmorphic web interface with micro-animations.

---

## Tech Stack

* **Core**: Python 3.x
* **AI Models**: Google Gemini (Flash & Embedding)
* **Backend**: Flask
* **Vector DB**: FAISS (Facebook AI Similarity Search)
* **Text Extraction**: PyPDF
* **Frontend**: HTML5, CSS3 (Vanilla), JavaScript

---

## System Architecture

1. **Ingestion**: PDFs in `data/docs` are parsed and split into overlapping chunks.
2. **Indexing**: Chunks are embedded via `models/gemini-embedding-001` and stored in a FAISS index.
3. **Query Agent**: User questions are rewritten by Gemini to improve retrieval accuracy.
4. **Retrieval**: Relevant context is pulled from the FAISS index using vector similarity.
5. **Generation**: Gemini generates a final response restricted strictly to the retrieved context and conversation memory.
6. **Validation**: (Optional) A validation agent checks if the response is fully supported by the source text.

---

## Setup Instructions

1. **Environment Setup**:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **API Configuration**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Run Application**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

---

## What This Project Demonstrates

* **Agentic RAG Design**: Moving beyond simple retrieval to multi-step AI reasoning.
* **API Integration**: Production-grade usage of Google GenAI SDK.
* **Modern Web UX**: A high-impact, responsive interface for AI tools.
* **Vector Similarity Search**: Practical implementation of FAISS for knowledge retrieval.

---

Local Agentic RAG System &copy; 2026
