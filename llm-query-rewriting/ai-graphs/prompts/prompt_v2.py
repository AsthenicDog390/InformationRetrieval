
QUERY_STYLE_PROMPT = """
Rewrite the query to maximize precision in document retrieval. Focus on making it specific, disambiguating vague terms, and including only the most relevant keywords to the user’s likely intent. Do not broaden the topic. Output the refined query only.

### Query:
------------
{query}
------------
"""
