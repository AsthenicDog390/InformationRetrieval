#!/usr/bin/env python3
"""
Evaluation Pipeline for Rewritten Queries
This script evaluates rewritten queries from a JSON file using PyTerrier.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pyterrier as pt
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rewritten queries using PyTerrier")
    parser.add_argument(
        "--rewritten-queries",
        type=str,
        default="ai-graphs/rewritten_queries/query_style_v3_gemma3:12b.json",
        help="Path to the JSON file containing rewritten queries",
    )
    parser.add_argument(
        "--skip-original", action="store_true", help="Skip comparison with original queries"
    )
    parser.add_argument(
        "--rebuild-index", action="store_true", help="Force rebuild of the index even if it exists"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ai-graphs/evaluation_results",
        help="Directory to save evaluation results",
    )
    return parser.parse_args()


class DatasetComponents:
    """Container for dataset components that must be provided"""

    def __init__(self, corpus_iter, queries_df, qrels_df):
        self.corpus_iter = corpus_iter  # Iterator yielding {'docno': str, 'text': str}
        self.queries_df = queries_df  # DataFrame with columns ['qid', 'query']
        self.qrels_df = qrels_df  # DataFrame with columns ['qid', 'docno', 'label']


def load_pt_dataset():
    """Load codec dataset"""
    docs = load_dataset("macavaney/codec")["train"]
    qrels = load_dataset("irds/codec", "qrels", trust_remote_code=True)
    queries = load_dataset("irds/codec", "queries", trust_remote_code=True)

    # Convert dataset to correct format
    corpus_iter = ({"docno": str(doc["id"]), "text": doc["contents"]} for doc in docs)

    queries_df = pd.DataFrame(queries)[["query_id", "query"]]
    queries_df.columns = ["qid", "query"]

    qrels_df = pd.DataFrame(qrels)[["query_id", "doc_id", "relevance"]]
    qrels_df.columns = ["qid", "docno", "label"]

    return DatasetComponents(corpus_iter, queries_df, qrels_df)


def preprocess_corpus(corpus_iter):
    """Generator that applies preprocessing to each document"""
    for doc in corpus_iter:
        yield {"docno": doc["docno"], "text": doc["text"]}


def preprocess_queries(queries_df):
    """Apply preprocessing to queries dataframe"""
    queries_df = queries_df.copy()
    tokeniser = pt.java.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
    queries_df["query"] = queries_df["query"].apply(
        lambda text: " ".join(tokeniser.getTokens(text))
    )
    return queries_df


def load_rewritten_queries(json_file_path):
    """Load rewritten queries from JSON file"""
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to DataFrame
    queries_df = pd.DataFrame(data)

    # Ensure we have the right column names
    if "query_id" in queries_df.columns and "query" in queries_df.columns:
        queries_df = queries_df[["query_id", "query"]]
        queries_df.columns = ["qid", "query"]
    else:
        raise ValueError(f"JSON file {json_file_path} must have 'query_id' and 'query' fields")

    return queries_df


def build_or_load_index(corpus_iter, index_path, rebuild=False):
    """Build or load index"""
    index_ref = None

    # Check if we need to rebuild the index
    if rebuild and index_path.exists():
        shutil.rmtree(index_path)
        print(f"Removed existing index at {index_path}")

    # Build index if it doesn't exist or needs rebuilding
    if not index_path.exists() or rebuild:
        print("Building new index...")
        index_ref = pt.index.IterDictIndexer(
            str(index_path),
            meta={"docno": 32, "text": 131072},
            type=pt.index.IndexingType.CLASSIC,
            properties={"index.meta.data-source": "fileinmem"},
        ).index(corpus_iter)
        print(f"Index built successfully at {index_path}")
    else:
        print(f"Loading existing index from {index_path}")
        index_ref = pt.IndexFactory.of(str(index_path))

    # Verify index contains documents
    index = pt.IndexFactory.of(str(index_path))
    index_stats = index.getCollectionStatistics()
    print("\nIndex statistics:")
    print(f"Number of documents: {index_stats.getNumberOfDocuments()}")
    print(f"Number of terms: {index_stats.getNumberOfUniqueTerms()}")
    print(f"Number of postings: {index_stats.getNumberOfPostings()}")
    print(f"Number of tokens: {index_stats.getNumberOfTokens()}")

    return index_ref


def evaluate_queries(bm25, queries_df, qrels_df, name, eval_metrics):
    """Evaluate queries and return results"""
    results = pt.Experiment([bm25], queries_df, qrels_df, eval_metrics, names=[name], baseline=0)
    return results


def main():
    args = parse_args()

    # Initialize PyTerrier
    if not pt.started():
        pt.init()

    # Set paths
    index_path = Path.cwd() / "index"
    rewritten_queries_path = Path.cwd() / args.rewritten_queries
    results_dir = Path.cwd() / args.results_dir
    results_dir.mkdir(exist_ok=True)

    # Extract filename after the last slash and remove extension
    input_file_name = args.rewritten_queries.split("/")[-1]
    if input_file_name.endswith(".json"):
        input_file_name = input_file_name[:-5]  # Remove .json extension

    # Load the dataset
    print("Loading dataset...")
    data = load_pt_dataset()
    print("Dataset loaded successfully.")

    # Apply preprocessing to corpus
    print("Preprocessing corpus...")
    preprocessed_corpus = preprocess_corpus(data.corpus_iter)

    # Load or build the index
    index_ref = build_or_load_index(preprocessed_corpus, index_path, args.rebuild_index)

    # Create BM25 retrieval pipeline
    bm25 = pt.BatchRetrieve(
        index_ref,
        wmodel="BM25",
        metadata=["docno", "text"],
        properties={"termpipelines": ""},
        controls={"qe": "off"},
    )
    print("Retrieval system set up with BM25.")

    # Load rewritten queries
    print(f"Loading rewritten queries from {rewritten_queries_path}...")
    rewritten_queries = load_rewritten_queries(rewritten_queries_path)
    rewritten_queries = preprocess_queries(rewritten_queries)
    print(f"Loaded {len(rewritten_queries)} rewritten queries.")

    # Define evaluation metrics
    eval_metrics = ["map", "ndcg_cut_10", "P_10", "recall_100", "recip_rank"]

    # Run evaluation on rewritten queries
    print("Evaluating rewritten queries...")
    rewritten_results = evaluate_queries(
        bm25, rewritten_queries, data.qrels_df, f"BM25 with {input_file_name}", eval_metrics
    )

    print("\nResults for rewritten queries:")
    print(rewritten_results.to_string(index=False))

    # Save rewritten results
    rewritten_results.to_csv(results_dir / f"{input_file_name}.csv", index=False)
    print(f"Rewritten query results saved to {results_dir} / {input_file_name}.csv")

    # Compare with original queries if not skipped
    if not args.skip_original:
        print("\nProcessing original queries for comparison...")
        preprocessed_queries = preprocess_queries(data.queries_df)

        # Run baseline evaluation
        print("Evaluating original queries...")
        baseline_results = evaluate_queries(
            bm25, preprocessed_queries, data.qrels_df, "BM25 Baseline", eval_metrics
        )

        print("\nResults for original queries:")
        print(baseline_results.to_string(index=False))

        # Compare the baseline and rewritten query results
        print("\nDirect comparison of original vs. rewritten queries:")
        # Instead of using a list of dataframes, we'll manually create a comparison dataframe
        original_df = pd.DataFrame(
            {
                "name": ["Original Queries"],
                "map": [baseline_results["map"].values[0]],
                "recip_rank": [baseline_results["recip_rank"].values[0]],
                "P_10": [baseline_results["P_10"].values[0]],
                "recall_100": [baseline_results["recall_100"].values[0]],
                "ndcg_cut_10": [baseline_results["ndcg_cut_10"].values[0]],
            }
        )

        rewritten_df = pd.DataFrame(
            {
                "name": ["Rewritten Queries"],
                "map": [rewritten_results["map"].values[0]],
                "recip_rank": [rewritten_results["recip_rank"].values[0]],
                "P_10": [rewritten_results["P_10"].values[0]],
                "recall_100": [rewritten_results["recall_100"].values[0]],
                "ndcg_cut_10": [rewritten_results["ndcg_cut_10"].values[0]],
            }
        )

        comparison_results = pd.concat([original_df, rewritten_df])

        # Calculate differences and percentages
        print("\nDifference (Rewritten - Original):")
        diff_df = pd.DataFrame(
            {
                "map": [rewritten_results["map"].values[0] - baseline_results["map"].values[0]],
                "recip_rank": [
                    rewritten_results["recip_rank"].values[0]
                    - baseline_results["recip_rank"].values[0]
                ],
                "P_10": [rewritten_results["P_10"].values[0] - baseline_results["P_10"].values[0]],
                "recall_100": [
                    rewritten_results["recall_100"].values[0]
                    - baseline_results["recall_100"].values[0]
                ],
                "ndcg_cut_10": [
                    rewritten_results["ndcg_cut_10"].values[0]
                    - baseline_results["ndcg_cut_10"].values[0]
                ],
            }
        )

        print(diff_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\nPercentage change ((Rewritten - Original) / Original) * 100:")
        pct_df = pd.DataFrame(
            {
                "map": [
                    (rewritten_results["map"].values[0] - baseline_results["map"].values[0])
                    / baseline_results["map"].values[0]
                    * 100
                ],
                "recip_rank": [
                    (
                        rewritten_results["recip_rank"].values[0]
                        - baseline_results["recip_rank"].values[0]
                    )
                    / baseline_results["recip_rank"].values[0]
                    * 100
                ],
                "P_10": [
                    (rewritten_results["P_10"].values[0] - baseline_results["P_10"].values[0])
                    / baseline_results["P_10"].values[0]
                    * 100
                ],
                "recall_100": [
                    (
                        rewritten_results["recall_100"].values[0]
                        - baseline_results["recall_100"].values[0]
                    )
                    / baseline_results["recall_100"].values[0]
                    * 100
                ],
                "ndcg_cut_10": [
                    (
                        rewritten_results["ndcg_cut_10"].values[0]
                        - baseline_results["ndcg_cut_10"].values[0]
                    )
                    / baseline_results["ndcg_cut_10"].values[0]
                    * 100
                ],
            }
        )

        print(pct_df.to_string(index=False, float_format=lambda x: f"{x:.2f}%"))

        print("\nComparison results:")
        print(comparison_results.to_string(index=False))

        # Save baseline and comparison results
        baseline_results.to_csv(
            results_dir / f"{input_file_name}_baseline_results.csv", index=False
        )
        comparison_results.to_csv(
            results_dir / f"{input_file_name}_comparison_results.csv", index=False
        )
        diff_df.to_csv(results_dir / f"{input_file_name}_difference_results.csv", index=False)
        pct_df.to_csv(results_dir / f"{input_file_name}_percentage_change_results.csv", index=False)
        print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
