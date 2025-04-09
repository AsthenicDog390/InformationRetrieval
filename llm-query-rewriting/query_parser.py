import json
import re

def clean_to_single_string(text):
    # Replace escaped quotes and backslashes
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace('"\"', '"').replace("\"", "")
    # Replace escaped newlines and real newlines with space
    text = text.replace('\\n', ' ').replace('\n', ' ')

    # Collapse any other weird spacing
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    return text.strip()

def clean_query_rewrites(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in data:
        rewrite_text = item.get("query_rewrite", "")
        item["query_rewrite"] = clean_to_single_string(rewrite_text)
        

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)

# Example usage
if __name__ == "__main__":
    clean_query_rewrites("./results/01-llm-rewriting/few_shot_clarity.json", "./results/01-llm-rewriting/cleaned/few_shot_clarity_cleaned.json")
    clean_query_rewrites("./results/01-llm-rewriting/few_shot_expansion.json", "./results/01-llm-rewriting/cleaned/few_shot_expansion_cleaned.json")
    clean_query_rewrites("./results/01-llm-rewriting/few_shot_search_friendly.json", "./results/01-llm-rewriting/cleaned/few_shot_search_friendly_cleaned.json")
    clean_query_rewrites("./results/01-llm-rewriting/zero_shot_clarity.json", "./results/01-llm-rewriting/cleaned/zero_shot_clarity_cleaned.json")
    clean_query_rewrites("./results/01-llm-rewriting/zero_shot_expansion.json", "./results/01-llm-rewriting/cleaned/zero_shot_expansion_cleaned.json")
    clean_query_rewrites("./results/01-llm-rewriting/zero_shot_search_friendly.json", "./results/01-llm-rewriting/cleaned/zero_shot_search_friendly_cleaned.json")
    clean_query_rewrites("./results/01-llm-rewriting/zero_shot.json", "./results/01-llm-rewriting/cleaned/zero_shot_cleaned.json")