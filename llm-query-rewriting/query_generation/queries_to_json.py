# This script is used to print CODEC dataset queries to a json file
import json
import os

from datasets import load_dataset

hf_token = os.getenv("HUGGINGFACE_TOKEN")
queries = load_dataset("irds/codec", "queries", token=hf_token)

for q in queries:
    print(q)

# Convert Database objects to dictionaries
queries_dict = [
    {"query_id": q["query_id"], "query": q["query"], "domain": q["domain"], "guidelines": q["guidelines"]}
    for q in queries
]


with open("llm-query-rewriting/query_generation/json_files/dataset_queries.json", "w", encoding="utf-8") as json_file:
    json.dump(queries_dict, json_file, indent=4)

if __name__ == "__main__":
    print("Queries have been converted to JSON format.")
