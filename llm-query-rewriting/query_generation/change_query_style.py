import json
import os
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field


class Query(BaseModel):
    query_id: str = Field(description="please put an empty string")
    query: str = Field(description="The query you rewritten")
    domain: str = Field(description="please put an empty string")
    guidelines: str = Field(description="please put an empty string")



def chat_ollama(query):
    prompt = f"""
You are an expert query rewriter.

Your task is to take a short, high-level user query and rewrite it to be more detailed and information-rich,
while preserving its original intent and context. The rewritten query should be more suitable for retrieving in-depth documents.

### Query:
------------
{query}
------------
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model="llama3.1",
        format=Query.model_json_schema(),
    )

    gen_query = Query.model_validate_json(response.message.content)
    gen_query.query_id = query["query_id"]
    gen_query.domain = query["domain"]
    gen_query.guidelines = query["guidelines"]

    return gen_query


def read_queries_from_json(fp):
    with open(fp, "r", encoding="utf-8") as file:
        return json.load(file)


def load_existing_queries(fp):
    if not os.path.exists(fp):
        return []
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_queries(fp, queries):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_fp = "llm-query-rewriting/query_generation/json_files/dataset_queries.json"
    output_fp = "llm-query-rewriting/query_generation/json_files/rewritten_queries.json"

    Path(output_fp).parent.mkdir(parents=True, exist_ok=True)

    queries = read_queries_from_json(input_fp)

    try:
        rewritten_queries = []
        for entry in queries:
            rewritten = chat_ollama(entry)
            rewritten_queries.append(rewritten.model_dump())
            save_queries(output_fp, rewritten_queries)
            print(f"Saved: {rewritten.query_id}")
    except KeyboardInterrupt:
        print("Execution interrupted. Progress saved.")