from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from dotenv import load_dotenv
from transformers import pipeline
import torch
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import pymupdf

# document opening
doc = pymupdf.open("Documents/Test.pdf")    # opening a document(pdf)
print(doc)
print(len(doc))

# storing raw text
text_str = ""
for i in range(len(doc)):
    text_str += doc[i].get_text()   # for getting the texts in the string

# chunking the text
chunk_list = []
chunk_size = 500
overlap = 100   # chunking with overlapping
for ind in range(0, len(text_str), chunk_size-overlap):
    chunk_list.append(text_str[ind:ind+chunk_size])
# for i in chunk_list:
#     print(i, "\n")
print(len(chunk_list), "\n")

# embeddings formation
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2")   # pretrained model
chunk_embeddings = model.encode(chunk_list)
print(chunk_embeddings.shape)   # shape of the embedding

# similarity
query = "What is yolo?"
query_embeddings = model.encode(query)
print(query_embeddings.shape)
similarity_scores = util.cos_sim(chunk_embeddings, query_embeddings)
print(similarity_scores)

# retrieval of closest chunk
max_sim_score = similarity_scores[0].item()
max_sim_chunk = 0
for i in range(1, len(similarity_scores)):
    if similarity_scores[i].item() > max_sim_score:
        max_sim_score = similarity_scores[i].item()
        max_sim_chunk = i
print(max_sim_score)
print(max_sim_chunk)
max_chunk = chunk_list[max_sim_chunk]
print(max_chunk)

# API connection

tokenizer = AutoTokenizer.from_pretrained(
    "google/flan-t5-small"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-small"
)
prompt = f"""
Context:
{max_chunk}

Question:
{query}

Answer:
"""
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True
)
outputs = model.generate(
    **inputs,
    max_new_tokens=100
)
answer = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
print(f"Question: {query}")
print(f"Answer: {answer}")
