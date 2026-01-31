from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
import faiss
import numpy as np

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

# ---------- Embeddings (GPU enabled if available) ----------
embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_embeddings = embed_model.encode(chunks)

# ---------- Vector index ----------
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

# ---------- Load local LLM ----------
llm = AutoModelForCausalLM.from_pretrained(
    "models",
    model_file="mistral.gguf",
    model_type="mistral"
)

# ---------- Query Rewriting Agent ----------
def rewrite_query(user_query):
    prompt = f"""
You are a query rewriting agent for an incident investigation system.
Rewrite the user's question into ONE clear, specific search query
focused on system failure, outage, incident, or root cause.

Return ONLY one sentence.
Do not be generic.

User question:
{user_query}

Rewritten query:
"""
    rewritten = llm(prompt, max_new_tokens=25)
    return rewritten.strip()



# ---------- User question ----------
user_question = "what happened to the server?"

rewritten_question = rewrite_query(user_question)

print("Original question:", user_question)
print("Rewritten question:", rewritten_question)

# ---------- Retrieval ----------
question_embedding = embed_model.encode([rewritten_question])
_, indices = index.search(question_embedding, 1)

context = "\n".join([chunks[i] for i in indices[0]])

# ---------- Answer generation ----------
final_prompt = f"""
You are an assistant that answers ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{user_question}

Answer:
"""

answer = llm(final_prompt, max_new_tokens=80)

print("\nFINAL ANSWER:\n")
print(answer)
