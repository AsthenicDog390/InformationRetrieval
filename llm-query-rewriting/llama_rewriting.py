import os
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import query_loader
import torch
import time

torch.mps.empty_cache()  # Clears unused MPS memory
torch.mps.synchronize()  # Ensures all operations are completed

hf_token = os.getenv("HUGGINGFACE_TOKEN") 


# ----------------------------
# Prompting Strategies
# ----------------------------
class PromptingStrategies:
    """
    Contains the rewriting strategies.
    """
    def __init__(self):
        self.strategies = {
            "expansion": "using an expansion approach to add synonyms and related keywords",
            "search_friendly": "rewriting it to be search engine friendly, focusing on concise keywords",
            "clarity": "rewriting it for clarity, removing ambiguities and improving understanding"
        }

    def get_instruction(self, strategy: str) -> str:
        return self.strategies.get(strategy, "")

# ----------------------------
# Prompting Techniques
# ----------------------------
class PromptingTechniques:
    """
    Contains methods to build a complete prompt based on the technique (zero_shot, cot, few_shot)
    and an optional rewriting strategy.
    """
    def __init__(self, strategies: PromptingStrategies):
        self.strategies = strategies

    def get_prompt(self, query: str, technique: str, strategy: str = None) -> str:
        # Baseline: no additional strategy
        if technique == "zero_shot" and strategy is None:
            return f"Rewrite the following query:\nQuery: \"{query}\"\nRewritten Query:"
        
        if technique == "cot" and strategy is None:
            # Chain-of-Thought (CoT) prompt: encourage a step-by-step analysis before rewriting.
           return (
                f"You are an expert in search query reformulation. Your task is to refine the given query by first answering it and then using that answer to improve clarity, completeness, and relevance.\n\n"

                f"Follow these steps:\n"
                f"1. **Answer the Query:** Based on your knowledge, generate a well-structured and informative response.\n"
                f"2. **Analyze the Response:** Identify missing details or ambiguities in the original query.\n"
                f"3. **Rewrite the Query:** Using the answer, generate a more precise and complete version of the original query.\n\n"

                f"Original Query: \"{query}\"\n\n"
                
                f"Rewritten Query:\n"
            )
        
        # Get the strategy instruction text
        strat_inst = self.strategies.get_instruction(strategy)

        if technique == "zero_shot":
            return (f"Rewrite the following query {strat_inst}.\n"
                    f"Query: \"{query}\"\nRewritten Query:")

        elif technique == "few_shot":
            # Few-shot examples differ per strategy.
            if strategy == "expansion":
                examples = (
                    "Example 1:\n"
                    "Query: 'cheap smartphones'\n"
                    "Rewritten: 'affordable smartphones with high performance and extended battery life'\n\n"
                    "Example 2:\n"
                    "Query: 'budget laptops'\n"
                    "Rewritten: 'inexpensive laptops ideal for students and professionals'\n"
                )
            elif strategy == "search_friendly":
                examples = (
                    "Example 1:\n"
                    "Query: 'weather forecast'\n"
                    "Rewritten: 'current weather forecast and updates'\n\n"
                    "Example 2:\n"
                    "Query: 'news today'\n"
                    "Rewritten: 'latest news headlines and breaking news'\n"
                )
            elif strategy == "clarity":
                examples = (
                    "Example 1:\n"
                    "Query: 'AI books'\n"
                    "Rewritten: 'What are the best books for learning artificial intelligence in 2024?'\n\n"
                    "Example 2:\n"
                    "Query: 'healthy food'\n"
                    "Rewritten: 'What are some healthy food options for a balanced diet?'\n"
                )
            else:
                examples = ""
            return (f"{examples}\nNow, rewrite the following query {strat_inst}.\n"
                    f"Query: \"{query}\"\nRewritten Query:")
        else:
            # Default fallback
            return f"Rewrite the following query:\nQuery: \"{query}\"\nRewritten Query:"

# ----------------------------
# Model Generator Setup
# ----------------------------
def get_llm_generator(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Loads the tokenizer and model, and returns a text generation pipeline.
    Ensure you have access (and pass a Hugging Face token if required).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normalized Float 4 (better accuracy)
        bnb_4bit_compute_dtype="float16"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, token=hf_token, device_map="sequential")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)
    return generator

# ----------------------------
# Rewriting Pipeline
# ----------------------------
def rewriting_pipeline(queries):
    """
    For each query in the list, generate rewrites using various prompting combinations.
    Total possibilities: 1 baseline + (3 strategies * 3 techniques) = 10 possibilities.
    Saves results in separate JSON files under the "results" folder.
    """
    # Create results folder if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    generator = get_llm_generator()
    ps = PromptingStrategies()
    pt = PromptingTechniques(ps)

    # Define the combinations:
    # Baseline: zero_shot without strategy (represented as (technique, strategy) = ("zero_shot", None))
    # Then for each strategy in ["expansion", "search_friendly", "clarity"], apply:
    # - zero_shot, cot, few_shot
    possibilities = []
    possibilities.append(("zero_shot", None))  # baseline
    possibilities.append(("cot", None))  # cot only
    for strat in ["expansion", "search_friendly", "clarity"]:
        possibilities.append(("zero_shot", strat))
    # for strat in ["expansion", "search_friendly", "clarity"]:
    #     possibilities.append(("cot", strat))
    for strat in ["expansion", "search_friendly", "clarity"]:
        possibilities.append(("few_shot", strat))

    # Iterate over each possibility and each query, then save the results
    for technique, strategy in possibilities:
        # Create a key for the current combination
        key = f"{technique}" if strategy is None else f"{technique}_{strategy}"
        print(f"Processing combination: {key}")
        results = []
        for query_obj in queries:
            # Build the prompt for this query based on the technique and strategy.
            prompt_text = pt.get_prompt(query_obj.query, technique, strategy)
            print(f"Prompt: {prompt_text}")
            # Generate the rewritten query.
            response = generator(prompt_text)
            generated_text = response[0]['generated_text']
            print(f"----Result: {generated_text}")
            # Post-process: extract the part after the marker "Rewritten Query:" or "Rewritten:"
            if "Rewritten Query:" in generated_text:
                rewrite = generated_text.split("Rewritten Query:")[-1].strip()
            elif "Rewritten:" in generated_text:
                rewrite = generated_text.split("Rewritten:")[-1].strip()
            else:
                rewrite = generated_text.strip()
            

            match = re.search(r'([^"\n]*\?)', rewrite)
            only_query = rewrite
            if match:
                only_query = match.group(1).strip(' "\n')
        
            result_item = {
                "query_id": query_obj.query_id,
                "query": query_obj.query,
                "query_rewrite": only_query,
                "technique": technique,
                "strategy": strategy if strategy is not None else "baseline"
            }
            results.append(result_item)
            time.sleep(30)
        # Save the results for this combination in a JSON file
        file_name = os.path.join("results", f"{key}.json")
        with open(file_name, "a") as f:
            json.dump(results, f, indent=2)
        print(f"Results for {key} saved to {file_name}")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    queries = query_loader.retrieve_queries()
    print(f"Rewriting {len(queries)} queries...")
    rewriting_pipeline(queries[:5])
