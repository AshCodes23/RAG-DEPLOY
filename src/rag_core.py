from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
import faiss
import numpy as np
import os
import pickle

# ---------------- FEATURE TOGGLES ----------------
USE_QUERY_REWRITE = True      # Turn OFF if too slow
USE_VALIDATION = False        # Turn ON only if you accept high latency
TOP_K = 2
MAX_TOKENS = 120
# -------------------------------------------------


class RAGSystem:
    def __init__(self):
        print("🔥 Initializing RAG system (models load ONCE)")

        # Embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Local LLM (CPU)
        self.llm = AutoModelForCausalLM.from_pretrained(
            "models",
            model_file="mistral.gguf",
            model_type="mistral"
        )

        self.chunks = []
        self.index = None
        self.conversation_memory = []

        self._load_or_build_index()

    # --------------------------------------------------
    # Load documents and cached vector store
    # --------------------------------------------------
    def _load_or_build_index(self):
        cache_path = "cache/vector_store.pkl"

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.chunks, self.index = pickle.load(f)
            print("✅ Loaded cached vector store")
            return

        print("📄 Building vector store from documents...")
        full_text = ""

        for file in os.listdir("data/docs"):
            if file.endswith(".pdf"):
                reader = PdfReader(os.path.join("data/docs", file))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text

        self.chunks = self._chunk_text(full_text)

        embeddings = self.embed_model.encode(self.chunks)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

        with open(cache_path, "wb") as f:
            pickle.dump((self.chunks, self.index), f)

        print("✅ Vector store built and cached")

    # --------------------------------------------------
    # Rebuild index (called after new PDF upload)
    # --------------------------------------------------
    def rebuild_index(self):
        cache_path = "cache/vector_store.pkl"
        if os.path.exists(cache_path):
            os.remove(cache_path)

        self.chunks = []
        self.index = None
        self.conversation_memory = []

        self._load_or_build_index()

    # --------------------------------------------------
    # Chunking logic
    # --------------------------------------------------
    def _chunk_text(self, text, chunk_size=400, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    # --------------------------------------------------
    # Query Rewriting Agent
    # --------------------------------------------------
    def _rewrite_query(self, user_query):
        prompt = f"""
Rewrite the question into a short, clear search query
focused only on factual document information.

Question:
{user_query}

Rewritten query:
"""
        return self.llm(prompt, max_new_tokens=25).strip()

    # --------------------------------------------------
    # Validation Agent (OPTIONAL, SLOW)
    # --------------------------------------------------
    def _validate_answer(self, context, answer):
        prompt = f"""
Determine whether the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Reply ONLY:
SUPPORTED
or
NOT_SUPPORTED
"""
        verdict = self.llm(prompt, max_new_tokens=20).upper()
        return verdict.strip().startswith("SUPPORTED")

    # --------------------------------------------------
    # MAIN RAG PIPELINE
    # --------------------------------------------------
    def ask(self, user_question):

        # ----- Step 1: Rewrite query (optional) -----
        if USE_QUERY_REWRITE:
            query = self._rewrite_query(user_question)
        else:
            query = user_question

        # ----- Step 2: Embed query -----
        query_embedding = self.embed_model.encode([query])

        # ----- Step 3: Retrieve context -----
        _, indices = self.index.search(query_embedding, TOP_K)
        context = "\n".join([self.chunks[i] for i in indices[0]])

        # ----- Step 4: Conversation memory -----
        memory_context = "\n".join(
            [f"Q: {m['question']}\nA: {m['answer']}"
             for m in self.conversation_memory[-3:]]
        )

        # ----- Step 5: Answer generation -----
        prompt = f"""
You are an assistant that answers ONLY using the provided document context.
If the answer is not present, say "I don't know based on the document".

Previous conversation:
{memory_context}

Document Context:
{context}

Question:
{user_question}

Answer:
"""
        answer = self.llm(prompt, max_new_tokens=MAX_TOKENS)

        # ----- Step 6: Validation (optional) -----
        if USE_VALIDATION:
            is_valid = self._validate_answer(context, answer)
            if not is_valid:
                answer = "I don't know based on the document."

        # ----- Step 7: Store memory -----
        self.conversation_memory.append({
            "question": user_question,
            "answer": answer
        })

        return answer
