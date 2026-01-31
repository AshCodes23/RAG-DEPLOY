from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The server crashed due to high memory usage",
    "A cat is sitting on the sofa",
    "System failure caused downtime last month"
]

embeddings = model.encode(sentences)

print(len(embeddings))
print(len(embeddings[0]))
