"""Graph construction, entity resolution, communities, persistence.

The graph is a NetworkX MultiDiGraph serialized to a plain JSON dict so
the demo and evals can load it without rebuilding. Nodes carry a `type`,
edges a `relation`, a `weight` and a `provenance`.
"""
import json
import unicodedata

import networkx as nx

import config
from src.extract import Extraction


def normalize(name: str) -> str:
    """Lowercase, strip accents — the key used for entity resolution."""
    name = name.strip().lower()
    name = unicodedata.normalize("NFKD", name)
    return "".join(c for c in name if not unicodedata.combining(c))


def build_graph(extraction: Extraction) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    # entity resolution: merge nodes whose normalized form collides
    canonical: dict[str, str] = {}
    for name, ntype in extraction.nodes.items():
        key = normalize(name)
        if key not in canonical:
            canonical[key] = name
            g.add_node(name, type=ntype)
        # else: alias of an existing node — merged silently
    for t in extraction.triples:
        src = canonical[normalize(t.source)]
        tgt = canonical[normalize(t.target)]
        g.add_edge(src, tgt, relation=t.relation,
                   weight=t.weight, provenance=t.provenance)
    return g


def detect_communities(g: nx.MultiDiGraph, seed: int = config.LOUVAIN_SEED
                       ) -> list[set]:
    """Louvain communities on the undirected projection (deterministic).

    Edge weights are ignored here on purpose: the A_ATTRIBUÉ edges carry
    euro amounts (millions) which would drown the graph structure.
    Communities should reflect who works with whom, not how much.
    """
    undirected = nx.Graph()
    for u, v in g.edges():
        if undirected.has_edge(u, v):
            undirected[u][v]["weight"] += 1.0
        else:
            undirected.add_edge(u, v, weight=1.0)
    # keep isolated nodes as singleton communities
    for n in g.nodes:
        if n not in undirected:
            undirected.add_node(n)
    communities = nx.community.louvain_communities(
        undirected, weight="weight", seed=seed)
    return [set(c) for c in communities]


def community_stats(g: nx.MultiDiGraph, community: set) -> dict:
    """Aggregate facts about a community (used by summaries and answers)."""
    members = list(community)
    types: dict[str, int] = {}
    total_amount = 0.0
    contracts = []
    for m in members:
        t = g.nodes[m].get("type", "?")
        types[t] = types.get(t, 0) + 1
        if t == "contrat":
            contracts.append(m)
    for u, v, d in g.edges(data=True):
        if (d.get("relation") == "ATTRIBUÉ_À"
                and u in community and v in community):
            total_amount += d.get("weight", 0.0)
    return {"size": len(members), "types": types,
            "contracts": sorted(contracts),
            "total_amount": round(total_amount, 2)}


def to_dict(g: nx.MultiDiGraph, communities: list[set] | None = None,
            summaries: dict[int, str] | None = None) -> dict:
    return {
        "nodes": [{"name": n, "type": d.get("type", "?")}
                  for n, d in g.nodes(data=True)],
        "edges": [{"source": u, "relation": d.get("relation", "?"),
                   "target": v, "weight": d.get("weight", 1.0),
                   "provenance": d.get("provenance", "?")}
                  for u, v, d in g.edges(data=True)],
        "communities": [sorted(c) for c in (communities or [])],
        "summaries": {str(k): v for k, v in (summaries or {}).items()},
    }


def from_dict(data: dict) -> tuple[nx.MultiDiGraph, list[set], dict[int, str]]:
    g = nx.MultiDiGraph()
    for n in data["nodes"]:
        g.add_node(n["name"], type=n["type"])
    for e in data["edges"]:
        g.add_edge(e["source"], e["target"],
                   relation=e["relation"], weight=e["weight"],
                   provenance=e["provenance"])
    communities = [set(c) for c in data.get("communities", [])]
    summaries = {int(k): v for k, v in data.get("summaries", {}).items()}
    return g, communities, summaries


def save_graph(g: nx.MultiDiGraph, path: str = config.GRAPH_PATH,
               communities: list[set] | None = None,
               summaries: dict[int, str] | None = None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(g, communities, summaries), f,
                  ensure_ascii=False, indent=1)


def load_graph(path: str = config.GRAPH_PATH
               ) -> tuple[nx.MultiDiGraph, list[set], dict[int, str]]:
    with open(path, encoding="utf-8") as f:
        return from_dict(json.load(f))
