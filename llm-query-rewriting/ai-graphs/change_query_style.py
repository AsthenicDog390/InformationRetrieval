import json
import os
from pathlib import Path

from ollama import chat
from prompts.prompt_v1 import QUERY_STYLE_PROMPT as PROMPT_V1
from prompts.prompt_v2 import QUERY_STYLE_PROMPT as PROMPT_V2
from prompts.prompt_v3 import QUERY_STYLE_PROMPT as PROMPT_V3
from pydantic import BaseModel, Field


class Query(BaseModel):
    query_id: str = Field(description="please put an empty string")
    query: str = Field(description="The query you rewritten")
    domain: str = Field(description="please put an empty string")
    guidelines: str = Field(description="please put an empty string")


def chat_ollama(query, model="llama3.2", version="v1"):
    if version == "v1":
        prompt = PROMPT_V1.format(query=query["query"])
    elif version == "v2":
        prompt = PROMPT_V2.format(query=query["query"])
    elif version == "v3":
        prompt = PROMPT_V3.format(query=query["query"])

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
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
    input_fp = "query_generation/json_files/dataset_queries.json"
    queries = read_queries_from_json(input_fp)

    models = ["llama3.3", "gemma3:12b"]
    versions = ["v1", "v2", "v3"]
    for version in versions:
        for model in models:
            output_fp = f"ai-graphs/rewritten_queries/query_style_{version}_{model}.json"
            Path(output_fp).parent.mkdir(parents=True, exist_ok=True)
            try:
                rewritten_queries = []
                for entry in queries:
                    rewritten = chat_ollama(entry, model, version)
                    rewritten_queries.append(rewritten.model_dump())
                    save_queries(output_fp, rewritten_queries)
                    print(f"Saved: {rewritten.query_id}")
            except KeyboardInterrupt:
                print("Execution interrupted. Progress saved.")
