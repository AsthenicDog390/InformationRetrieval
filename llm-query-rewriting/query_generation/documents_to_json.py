import json
import os
from datasets import load_dataset

# Retrieve Hugging Face authentication token from environment variable
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Load the 'docs' subset of the CODEC dataset
docs = load_dataset("macavaney/codec", "default", token=hf_token)

# Define the output file path
output_file = "llm-query-rewriting/query_generation/json_files/dataset_docs.json"

# # Convert each document to a dictionary with relevant fields
# sample_docs = docs["train"].select(range(10))  # .to_json(output_file, orient="records")

# Parameters
total_docs = len(docs["train"])
sample_size = 10_000

# Uniformly spaced sampling indices
step = total_docs // sample_size
indices = list(range(0, total_docs, step))[:sample_size]

# Select documents at those indices
sample_docs = docs["train"].select(indices)

# Recombine the columns into a list of dictionaries
docs_dict = [
    {
        "doc_id": doc["id"],
        "title": doc["title"],
        "text": doc["contents"],
        "url": doc["url"]
    }
    for doc in sample_docs
]


# Write the list of document dictionaries to a JSON file
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(docs_dict, json_file, indent=4)

if __name__ == "__main__":
    print(f"Documents have been exported to {output_file}.")