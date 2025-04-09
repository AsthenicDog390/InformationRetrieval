"""
This file contains all the prompts used in the query optimization and generation processes.
"""

# Prompts from build_own_prompt.py

QUERY_OPTIMIZATION_PROMPT = """
You are an expert in information retrieval and prompt engineering.

I need you to write a prompt for another LLM that will help it improve the following search query for better retrieval performance.

### Original Query:
------------
{query}
------------

Create a detailed prompt that instructs the LLM on how to optimize this specific query for information retrieval from a large document corpus.
Your prompt should guide the LLM to focus on:
1. Identifying key concepts and entities in the query
2. Adding relevant context or details that might be missing
3. Removing noise or ambiguity
4. Considering alternative phrasings that better match document language
5. Maintaining the original intent

Please follow these guidelines and restrictions:
1. Do not include the original query in the prompt.  
2. Only provide instructions for the LLM to optimize the query.
3. Do not mention the query or topics from the query in the prompt.
4. You must only give instructions about what needs to be changed in the query to improve retrieval performance. The instructions should be information retrieval methods that are applied to the query such as lemmatization, stemming, etc.
   For example, you can say:
   - Remove noise or ambiguity
   - Apply stemming or lemmatization
   Alwats give these instructions based on the query.
   The retrieval performance is going to be measured using: map score,recip_rank, P_10, recall_100, ndcg_cut_10. 
   You must only give instructions about what needs to be changed in the query to improve retrieval performance.
5. In general, shorter, clear, to the point queries perform better.
Return ONLY the prompt text that will be given to the other LLM.
"""

IMPROVE_QUERY_PROMPT = """
{generated_prompt}

Only return the improved query text without any explanations or additional text.
### Original Query:
------------
{query}
------------
"""

# Prompts from select_optimized_prompt.py

"""
This file contains improved prompts for query optimization based on different query styles.
"""

# Query classification prompt
QUERY_STYLE_ANALYSIS_PROMPT = """
Analyze the following search query and classify its style as one of the following:
- ambiguous: Query is unclear, has multiple possible interpretations, or lacks specific intent
- concise: Query is brief but clear, and could benefit from more context
- descriptive: Query is detailed and provides context, but may be verbose
- keyword_based: Query consists mainly of keywords without natural language structure

### Query:
------------
{query}
------------

Return only one of these four values: 'ambiguous', 'concise', 'descriptive' or 'keyword_based'. You MUST only return the value, nothing else, no explanations, no other text. You MUST not invent any other values.
"""

# Ambiguous query optimization - uses domain and guidelines to clarify intent
AMBIGUOUS_QUERY_OPTIMIZATION_PROMPT = """
You are an expert query reformulator for information retrieval.

The following search query is ambiguous and could have multiple interpretations. Please reformulate it into 
a clearer query that would perform better in an information retrieval system. Use the provided domain and 
guidelines to understand the intended meaning and context.

### Original Query:
------------
{query}
------------

Your task is to create a specific, unambiguous query that will improve retrieval performance.
Focus on:
1. Clarifying the primary intent
2. Adding specific terms from the domain
3. Removing words that could be interpreted in multiple ways
4. Creating a natural language query that would match relevant documents

The reformulated query should be measurably better for metrics like MAP score, reciprocal rank, P@10, recall@100, and NDCG@10.

Return only the reformulated query text without any explanations.
"""

# Concise query optimization - expands with relevant context
CONCISE_QUERY_OPTIMIZATION_PROMPT = """
You are an expert query enhancer for information retrieval.

The following search query is concise but could benefit from additional context and terms to improve retrieval.
Please expand it to be more specific and comprehensive while preserving its original intent. Use the domain
to add relevant terminology that would match documents in this field.

### Original Query:
------------
{query}
------------

Your task is to enhance this query by:
1. Adding specific domain terminology that would appear in relevant documents
2. Including alternative terms or synonyms for key concepts
3. Providing additional context that clarifies the information need
4. Maintaining a natural language structure that modern retrieval systems can leverage

The enhanced query should improve metrics like MAP score, reciprocal rank, P@10, recall@100, and NDCG@10.

Return only the enhanced query text without any explanations.
"""

# Descriptive query optimization - reduces verbosity while maintaining intent
DESCRIPTIVE_QUERY_OPTIMIZATION_PROMPT = """
You are an expert query optimizer for information retrieval.

The following search query is descriptive but potentially too verbose. Please condense it into a
more focused and effective query that preserves the core information need. Extract the key concepts
and create a more efficient query.

### Original Query:
------------
{query}
------------

Your task is to optimize this query by:
1. Identifying and retaining the essential concepts
2. Removing unnecessary qualifiers or redundant terms
3. Structuring the query to emphasize the most important terms
4. Creating a balanced query that prioritizes both precision and recall

The optimized query should improve metrics like MAP score, reciprocal rank, P@10, recall@100, and NDCG@10.

Return only the optimized query text without any explanations.
"""

# Keyword-based query optimization - adds structure and context
KEYWORD_QUERY_OPTIMIZATION_PROMPT = """
You are an expert query reformulator for information retrieval.

The following search query consists primarily of keywords without natural language structure.
Please reformulate it into a more effective natural language query that would better match
relevant documents in modern retrieval systems.

### Original Query:
------------
{query}
------------

Your task is to reformulate this query by:
1. Adding natural language structure to the keywords
2. Including any missing connecting words or context
3. Arranging terms in a logical order that reflects information need
4. Maintaining all important concepts from the original keywords

The reformulated query should improve metrics like MAP score, reciprocal rank, P@10, recall@100, and NDCG@10.

Return only the reformulated query text without any explanations.
"""

# Question-based query optimization - refines the question for better retrieval
QUESTION_QUERY_OPTIMIZATION_PROMPT = """
You are an expert query optimizer for information retrieval.

The following search query is phrased as a question. Please optimize it for better performance
in an information retrieval system while maintaining its question format.

### Original Query:
------------
{query}
------------

Your task is to optimize this question by:
1. Focusing on the key concepts and entities in the question
2. Removing any unnecessary conversational elements
3. Adding specific terminology that would match relevant documents
4. Ensuring the question clearly expresses the information need

The optimized query should improve metrics like MAP score, reciprocal rank, P@10, recall@100, and NDCG@10.

Return only the optimized question without any explanations.
"""

# Prompts from change_query_style.py

QUERY_STYLE_PROMPT = """
You are a search assistant optimizing queries for a high-performance information retrieval system. Given an original user query, rewrite it to improve retrieval effectiveness by:
 - clarifying ambiguous terms.
 - expanding acronyms and abbreviations.
 - adding relevant contextual keywords.
 - preserving the original intent without changing the core topic.

Output the rewritten query only

### Query:
------------
{query}
------------
"""
