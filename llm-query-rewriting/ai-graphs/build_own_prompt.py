# Auto Prompting
import json
import os
from pathlib import Path

from ollama import chat
from prompts.prompt_v1 import IMPROVE_QUERY_PROMPT, QUERY_OPTIMIZATION_PROMPT
from pydantic import BaseModel, Field


class PromptGeneratedQuery(BaseModel):
    query_id: str = Field(description="The ID of the original query")
    original_query: str = Field(description="The original query")
    generated_prompt: str = Field(description="The prompt generated to improve the query")
    query: str = Field(description="The improved query")
    domain: str = Field(description="Domain of the query")
    guidelines: str = Field(description="Guidelines for the query")


def generate_optimization_prompt(query, model="llama3.2"):
    prompt = QUERY_OPTIMIZATION_PROMPT.format(query=query["query"])

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )

    return response["message"]["content"].strip()


def improve_query_with_prompt(query, generated_prompt, model="llama3.2"):
    prompt = IMPROVE_QUERY_PROMPT.format(generated_prompt=generated_prompt, query=query["query"])

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
    )

    return response["message"]["content"].strip()


def read_queries_from_json(fp):
    with open(fp, "r", encoding="utf-8") as file:
        return json.load(file)


def save_queries(fp, queries):
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    input_fp = "query_generation/json_files/dataset_queries.json"
    queries = read_queries_from_json(input_fp)

    models = ["llama3.3", "gemma3:12b"]
    for model in models:
        output_fp = f"ai-graphs/rewritten_queries/build_own_prompt_v1_{model}.json"
        Path(output_fp).parent.mkdir(parents=True, exist_ok=True)
        try:
            improved_queries = []
            for entry in queries:
                # First LLM call: Generate a prompt for improving this specific query
                generated_prompt = generate_optimization_prompt(entry, model)

                # Second LLM call: Use the generated prompt to improve the query
                improved_query_text = improve_query_with_prompt(entry, generated_prompt, model)

                improved_entry = PromptGeneratedQuery(
                    query_id=entry["query_id"],
                    original_query=entry["query"],
                    generated_prompt=generated_prompt,
                    query=improved_query_text,
                    domain=entry["domain"],
                    guidelines=entry["guidelines"],
                )

                improved_queries.append(improved_entry.model_dump())
                save_queries(output_fp, improved_queries)
                print(f"Saved improved query: {entry['query_id']}")
        except KeyboardInterrupt:
            print("Execution interrupted. Progress saved.")
