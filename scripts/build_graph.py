"""Build the knowledge graph from the contract corpus.

Usage:
    python scripts/build_graph.py [--enrich]

--enrich runs the LLM text enrichment (needs an API key); without it the
graph is built from the structured fields only. The result (graph +
communities + summaries) is saved to data/graph.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.extract import enrich_text, extract_corpus, load_contracts
from src.graph import build_graph, detect_communities, save_graph
from src.llm import LLMClient
from src.summarize import summarize_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrich", action="store_true",
                        help="LLM text enrichment (needs API key)")
    args = parser.parse_args()

    contracts = load_contracts(config.DATA_PATH)
    print(f"loaded {len(contracts)} contracts")
    extraction = extract_corpus(contracts)

    llm = LLMClient()
    if args.enrich:
        if not llm.enabled:
            print("warning: --enrich needs an API key, skipping")
        else:
            for c in contracts:
                ex = enrich_text(c, llm)
                for name, ntype in ex.nodes.items():
                    extraction.add_node(name, ntype)
                extraction.triples.extend(ex.triples)
            n_text = sum(1 for t in extraction.triples
                         if t.provenance == "text")
            print(f"enriched with {n_text} text triples")

    g = build_graph(extraction)
    communities = detect_communities(g)
    print(f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
          f"{len(communities)} communities "
          f"(sizes: {sorted(len(c) for c in communities)})")

    summaries, cost = summarize_all(g, communities, llm)
    print(f"summaries written (LLM cost: ${cost:.4f})")
    save_graph(g, config.GRAPH_PATH, communities, summaries)
    print(f"saved to {config.GRAPH_PATH}")


if __name__ == "__main__":
    main()
