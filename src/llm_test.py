from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "models",
    model_file="mistral.gguf",
    model_type="mistral"
)

prompt = """You are a helpful assistant.

Question: Why did the server crash?
Answer:"""

output = llm(prompt, max_new_tokens=150)
print(output)
