# Dynamic Prompting
import json
from enum import Enum
from pathlib import Path

from ollama import chat
from prompts.prompt_v1 import (
    AMBIGUOUS_QUERY_OPTIMIZATION_PROMPT,
    CONCISE_QUERY_OPTIMIZATION_PROMPT,
    DESCRIPTIVE_QUERY_OPTIMIZATION_PROMPT,
    KEYWORD_QUERY_OPTIMIZATION_PROMPT,
    QUERY_STYLE_ANALYSIS_PROMPT,
)
from pydantic import BaseModel, Field


class QueryStyle(str, Enum):
    AMBIGUOUS = "ambiguous"
    CONCISE = "concise"
    DESCRIPTIVE = "descriptive"
    KEYWORD = "keyword_based"


class QueryStyleResponse(BaseModel):
    query_style: QueryStyle = Field(description="The style of the original query")


class OptimizedQuery(BaseModel):
    query_id: str = Field(description="The ID of the original query")
    original_query: str = Field(description="The original query")
    query_style: QueryStyle = Field(description="The style of the original query")
    query: str = Field(description="The optimized query")
    domain: str = Field(description="Domain of the query")
    guidelines: str = Field(description="Guidelines for the query")


def analyze_query_style(query, model="llama3.2"):
    prompt = QUERY_STYLE_ANALYSIS_PROMPT.format(query=query["query"])

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        format=QueryStyleResponse.model_json_schema(),
    )

    style = QueryStyleResponse.model_validate_json(response.message.content)
    return style.query_style

def optimize_query(query, style, model="llama3.2"):
    if style == QueryStyle.AMBIGUOUS:
        prompt = AMBIGUOUS_QUERY_OPTIMIZATION_PROMPT.format(query=query["query"])
    elif style == QueryStyle.CONCISE:
        prompt = CONCISE_QUERY_OPTIMIZATION_PROMPT.format(query=query["query"])
    elif style == QueryStyle.DESCRIPTIVE:
        prompt = DESCRIPTIVE_QUERY_OPTIMIZATION_PROMPT.format(query=query["query"])
    else:  # KEYWORD
        prompt = KEYWORD_QUERY_OPTIMIZATION_PROMPT.format(query=query["query"])

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
        output_fp = f"ai-graphs/rewritten_queries/select_optimized_prompt_v1_{model}.json"
        Path(output_fp).parent.mkdir(parents=True, exist_ok=True)
        try:
            optimized_queries = []
            for entry in queries:
                query_style = analyze_query_style(entry, model)
                query_text = optimize_query(entry, query_style, model)

                optimized_entry = OptimizedQuery(
                    query_id=entry["query_id"],
                    original_query=entry["query"],
                    query_style=query_style,
                    query=query_text,
                    domain=entry["domain"],
                    guidelines=entry["guidelines"],
                )

                optimized_queries.append(optimized_entry.model_dump())
                save_queries(output_fp, optimized_queries)
                print(f"Saved optimized query: {entry['query_id']} (Style: {query_style})")
        except KeyboardInterrupt:
            print("Execution interrupted. Progress saved.")
