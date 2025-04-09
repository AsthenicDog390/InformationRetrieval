# Evaluating Query Rewriting Techniques: Traditional vs. LLM-Based Approaches

This repository contains the code used in our paper: *Evaluating Query Rewriting Techniques: Traditional vs. LLM-Based Approaches*. The focus of this work is on exploring different query rewriting techniques and evaluating their impact on document retrieval performance using the CODEX dataset. 

We implement two different approaches for LLM-based query rewriting:

### 1. **Prompting Techniques for Query Rewriting** 
We explore various prompting strategies for rewriting queries, including:
- Zero-shot
- Few-shot
- Chain of Thought

These approaches are combined with three rewriting strategies:
- Clarity
- Search-engine-friendly
- Conciseness

The implementation for this approach can be found in the script: `llm-query-rewriting/llama_rewriting.py`.

### 2. **Autonomous LLM Query Rewriting**
In this approach, we allow the LLM to design its own suitable prompt. We provide this prompt to the LLM to rewrite the query in a way that it deems appropriate.

### Requirements:
- HuggingFace Access Token
- Access to the `LLama-3.2-3B-Instruct` model
- Install the dependencies by running: 
```bash
pip install -r llm-query-rewriting/requirements.txt
```

### Evaluation

THe evaluation is done in the `evaluation/evaluation_pipeline.ipynb`. It uses the BM25 retriever for document retriever and outputs: the following metrics `"map", "ndcg_cut_10", "P_10", "recall_100", "recip_rank"`