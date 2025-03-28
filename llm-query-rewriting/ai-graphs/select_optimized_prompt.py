import json
from enum import Enum
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field


class QueryStyle(str, Enum):
    SHORT = "short_and_not_detailed"
    LONG = "too_long"
    BOOLEAN = "follows_boolean_logic"


class OptimizedQuery(BaseModel):
    query_id: str = Field(description="The ID of the original query")
    original_query: str = Field(description="The original query")
    query_style: QueryStyle = Field(description="The style of the original query")
    optimized_query: str = Field(description="The optimized query")
    domain: str = Field(description="Domain of the query")
    guidelines: str = Field(description="Guidelines for the query")


def analyze_query_style(query):
    prompt = f"""
Analyze the following search query and classify its style as one of the following:
- short_and_not_detailed: Query is too brief and lacks specific details
- too_long: Query is excessively verbose or contains unnecessary information
- follows_boolean_logic: Query uses boolean operators (AND, OR, NOT)

### Query:
------------
{query["query"]}
------------

Return only one of these three values: short_and_not_detailed, too_long, follows_boolean_logic
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model="llama3.1",
    )

    style = response["message"]["content"].strip()

    if style == "short_and_not_detailed":
        return QueryStyle.SHORT
    elif style == "too_long":
        return QueryStyle.LONG
    elif style == "follows_boolean_logic":
        return QueryStyle.BOOLEAN
    else:
        return QueryStyle.SHORT  # Default to short if classification is unclear


def optimize_query(query, style):
    if style == QueryStyle.SHORT:
        prompt = f"""
You are an expert query enhancer for information retrieval.

The following search query is too short and lacks detail. Please expand it to be more specific and comprehensive
while preserving its original intent. Add relevant details, context, and alternative phrasings that would help
retrieve the most relevant documents from a large corpus.

### Original Query:
------------
{query["query"]}
------------

Return only the enhanced query text without any explanations.
"""
    elif style == QueryStyle.LONG:
        prompt = f"""
You are an expert query optimizer for information retrieval.

The following search query is too verbose and contains unnecessary information. Please condense it into a
more focused and effective query that preserves the core information need. Extract the key concepts and
prioritize precision over recall.

### Original Query:
------------
{query["query"]}
------------

Return only the optimized query text without any explanations.
"""
    else:  # BOOLEAN
        prompt = f"""
You are an expert query reformulator for information retrieval.

The following search query uses boolean logic (AND, OR, NOT). Please reformulate it into a natural language query
that preserves the same search intent but is more suitable for modern retrieval systems. Ensure it captures all
the logical relationships expressed in the original query.

### Original Query:
------------
{query["query"]}
------------

Return only the reformulated query text without any explanations.
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
    output_fp = "llm-query-rewriting/ai-graphs/json_files/optimized_queries.json"

    Path(output_fp).parent.mkdir(parents=True, exist_ok=True)

    queries = read_queries_from_json(input_fp)

    try:
        optimized_queries = []
        for entry in queries:
            query_style = analyze_query_style(entry)
            optimized_query_text = optimize_query(entry, query_style)

            optimized_entry = OptimizedQuery(
                query_id=entry["query_id"],
                original_query=entry["query"],
                query_style=query_style,
                optimized_query=optimized_query_text,
                domain=entry["domain"],
                guidelines=entry["guidelines"],
            )

            optimized_queries.append(optimized_entry.model_dump())
            save_queries(output_fp, optimized_queries)
            print(f"Saved optimized query: {entry['query_id']} (Style: {query_style})")
    except KeyboardInterrupt:
        print("Execution interrupted. Progress saved.")
