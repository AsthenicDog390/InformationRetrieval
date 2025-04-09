import os
from datasets import load_dataset

class Query:
    def __init__(self, query_id: str, query: str, domain: str, guidelines: str):
        """
        Represents a query for rewriting.

        Args:
            query_id (str): Unique identifier for the query.
            query (str): The actual search query.
            domain (str): The domain or category of the query.
            guidelines (str): Instructions or constraints for rewriting the query.
        """
        self.query_id = query_id
        self.query = query
        self.domain = domain
        self.guidelines = guidelines

    def __repr__(self):
        return f"Query(query_id={self.query_id}, query={self.query}, domain={self.domain}, guidelines={self.guidelines})"

hf_token = os.getenv("HUGGINGFACE_TOKEN")

def retrieve_queries():
    codec_queries = load_dataset('irds/codec', 'queries', token=hf_token)

    queries = [Query(**record) for record in codec_queries]
    
    return queries

