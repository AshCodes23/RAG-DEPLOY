from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
import faiss
import numpy as np
import os
import pickle

conversation_memory = []

# ---------- Load document ----------
reader = PdfReader("data/docs/sample_company_policy.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

# ---------- Chunking ----------
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(full_text)

# ---------- Embeddings + Vector Store (CACHED) ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

CACHE_PATH = "cache/vector_store.pkl"

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        cached_chunks, index = pickle.load(f)
        chunks = cached_chunks

else:
    chunk_embeddings = embed_model.encode(chunks)
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings))

    with open(CACHE_PATH, "wb") as f:
        pickle.dump((chunks, index), f)

# ---------- Load local LLM ----------
llm = AutoModelForCausalLM.from_pretrained(
    "models",
    model_file="mistral.gguf",
    model_type="mistral"
)

# ---------- Query Rewriting Agent ----------
def rewrite_query(user_query):
    prompt = f"""
You are a query rewriting agent for incident investigation.
Rewrite the user's question into ONE clear, specific search query
focused on system failure or root cause.

Return ONLY one sentence.

User question:
{user_query}

Rewritten query:
"""
    rewritten = llm(prompt, max_new_tokens=25)
    return rewritten.strip()

# ---------- Validation Agent ----------
def validate_answer(context, answer):
    prompt = f"""
You are an answer validation agent.
Determine whether the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Reply in this format ONLY:
SUPPORTED | CONFIDENCE=HIGH
SUPPORTED | CONFIDENCE=MEDIUM
NOT_SUPPORTED | CONFIDENCE=LOW
"""
    verdict = llm(prompt, max_new_tokens=20).upper()

    if verdict.strip().startswith("SUPPORTED"):
        return "SUPPORTED"
    else:
        return "NOT_SUPPORTED"


# ---------- User Question ----------
user_question = "what happened to the server?"

rewritten_question = rewrite_query(user_question)
print("Rewritten query:", rewritten_question)

# ---------- Retrieval ----------
question_embedding = embed_model.encode([rewritten_question])
_, indices = index.search(question_embedding, 2)
context = "\n".join([chunks[i] for i in indices[0]])

# ---------- Answer Generation ----------
memory_context = "\n".join(
    [f"Q: {m['question']}\nA: {m['answer']}" for m in conversation_memory[-3:]]
)

answer_prompt = f"""
You are an assistant that answers ONLY using the provided context.

Previous conversation:
{memory_context}

Document Context:
{context}

Question:
{user_question}

Answer:
"""
answer = llm(answer_prompt, max_new_tokens=120)
conversation_memory.append({
    "question": user_question,
    "answer": answer
})

# ---------- Validation ----------
verdict = validate_answer(context, answer)

print("\nGenerated Answer:\n", answer)
print("\nValidation Result:", verdict)

if "SUPPORTED" in verdict:
    print("\nFINAL ANSWER:\n", answer)
else:
    print("\nFINAL ANSWER:\nI don't know based on the provided documents.")

