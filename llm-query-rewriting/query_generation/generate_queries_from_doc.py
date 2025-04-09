import json
import os
from pathlib import Path

from ollama import chat
from pydantic import BaseModel, Field


class Query(BaseModel):
    query_id: str = Field(description="please put an empty string")
    query: str = Field(description="The query you generated")
    domain: str = Field(description="The domain of the query like politics or fincance etc.")
    guidelines: str = Field(description="please put an empty string")


def chat_ollama(document):
    response = chat(
        messages=[
            {
                "role": "user",
                "content": f"""
                	You are an expert prompt generator for information retrieval tasks in a document database. 
                    You are going to receive a document and you must write a query that could have been written by an user to retrieve a similar document to the one presented. 
                    The query should be a single sentence that is concise and informative. You can use the document text and the document title to generate the query. 
                    Please write the query in natural language and make sure it is relevant to the document content.
                
					### Document:
                    ------------
                    {document}
                    ------------
                
                """,
            }
        ],
        model="llama3.1",
        format=Query.model_json_schema(),
    )

    gen_query = Query.model_validate_json(response.message.content)
    gen_query.query_id = document["doc_id"]
    gen_query.guidelines = document["text"] + "\n" + document["url"]

    return gen_query


def read_docs_from_json(fp):
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
    input_fp = "llm-query-rewriting/query_generation/json_files/dataset_docs.json"
    output_fp = "llm-query-rewriting/query_generation/json_files/generated_queries.json"

    Path(output_fp).parent.mkdir(parents=True, exist_ok=True)

    docs = read_docs_from_json(input_fp)
    existing_queries = load_existing_queries(output_fp)
    processed_ids = {q["query_id"] for q in existing_queries}

    try:

        for doc in docs:
            if doc["doc_id"] in processed_ids:
                continue

            query = chat_ollama(doc)
            existing_queries.append(query.model_dump())
            save_queries(output_fp, existing_queries)
            print(f"Saved: {query.query_id}")
    except KeyboardInterrupt:
        print("Execution interrupted. Progress saved.")
