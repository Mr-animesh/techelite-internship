#!pip install transformers torch

from transformers import pipeline, set_seed

# Load GPT-2 model for text generation
generator = pipeline("text-generation", model="gpt2")
set_seed(42)  # Optional: for consistent output

# Prompt to generate from
prompt = "why artist should fear A.I." #can change this to get different summary

# Generate paragraph
output = generator(prompt, max_length=150, num_return_sequences=1)

# Display result
print(output[0]['generated_text'])