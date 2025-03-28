import json
import os
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field


class PromptGeneratedQuery(BaseModel):
    query_id: str = Field(description="The ID of the original query")
    original_query: str = Field(description="The original query")
    generated_prompt: str = Field(description="The prompt generated to improve the query")
    improved_query: str = Field(description="The improved query")
    domain: str = Field(description="Domain of the query")
    guidelines: str = Field(description="Guidelines for the query")


def generate_optimization_prompt(query):
    prompt = f"""
You are an expert in information retrieval and prompt engineering.

I need you to write a prompt for another LLM that will help it improve the following search query for better retrieval performance.

### Original Query:
------------
{query["query"]}
------------

Create a detailed prompt that instructs the LLM on how to optimize this specific query for information retrieval from a large document corpus.
Your prompt should guide the LLM to focus on:
1. Identifying key concepts and entities in the query
2. Adding relevant context or details that might be missing
3. Removing noise or ambiguity
4. Considering alternative phrasings that better match document language
5. Maintaining the original intent

Return ONLY the prompt text that will be given to the other LLM. Do not include explanations or meta-comments.
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model="llama3.1",
    )

    return response["message"]["content"].strip()


def improve_query_with_prompt(query, generated_prompt):
    prompt = f"""
{generated_prompt}

### Original Query:
------------
{query["query"]}
------------
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model="llama3.1",
    )

    return response["message"]["content"].strip()


def read_queries_from_json(fp):
    with open(fp, "r", encoding="utf-8") as file:
        return json.load(file)


def save_queries(fp, queries):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_fp = "llm-query-rewriting/query_generation/json_files/dataset_queries.json"
    output_fp = "llm-query-rewriting/ai-graphs/json_files/prompt_generated_queries.json"

    Path(output_fp).parent.mkdir(parents=True, exist_ok=True)

    queries = read_queries_from_json(input_fp)

    try:
        improved_queries = []
        for entry in queries:
            # First LLM call: Generate a prompt for improving this specific query
            generated_prompt = generate_optimization_prompt(entry)

            # Second LLM call: Use the generated prompt to improve the query
            improved_query_text = improve_query_with_prompt(entry, generated_prompt)

            improved_entry = PromptGeneratedQuery(
                query_id=entry["query_id"],
                original_query=entry["query"],
                generated_prompt=generated_prompt,
                improved_query=improved_query_text,
                domain=entry["domain"],
                guidelines=entry["guidelines"],
            )

            improved_queries.append(improved_entry.model_dump())
            save_queries(output_fp, improved_queries)
            print(f"Saved improved query: {entry['query_id']}")
    except KeyboardInterrupt:
        print("Execution interrupted. Progress saved.")
