"""Local / global / hybrid search over the contract knowledge graph.

- local:  link question entities to graph nodes, return their neighbourhood
  (1-2 hops) as a verbalized fact sheet — ideal for specific questions.
- global: match the question against community summaries, return the most
  relevant ones — ideal for holistic questions ("quels sont les grands
  domaines…").
- hybrid: local fact sheet + TF-IDF contract texts.

Without an API key the fact sheet itself is the answer (deterministic
French rendering). With a key, the LLM turns it into prose with citations.
"""
import time
from dataclasses import dataclass, field

import networkx as nx

import config
from src.graph import community_stats, normalize
from src.llm import LLMClient
from src.retrieve import TfidfRetriever

ANSWER_SYSTEM = """Tu es un analyste des marchés publics français. On te donne une
question et une fiche de faits extraite d'un graphe de connaissances
(entités, relations, montants). Réponds en français en 2-4 phrases en
citant les entités et chiffres de la fiche. Ne rajoute rien qui n'y figure
pas. Termine par la liste des sources : les identifiants de marchés cités."""


@dataclass
class Evidence:
    mode: str
    linked: list = field(default_factory=list)      # entity names
    facts: list = field(default_factory=list)       # verbalized triples
    contracts: list = field(default_factory=list)   # contract ids
    summaries: list = field(default_factory=list)   # community summaries
    chunks: list = field(default_factory=list)      # tfidf texts (hybrid)
    amounts: dict = field(default_factory=dict)     # entity -> total € won/let


@dataclass
class QueryResult:
    question: str
    mode: str
    answer: str
    evidence: Evidence
    citations: list = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    latency_s: float = 0.0
    cost_usd: float = 0.0


WRITE_VERBS = {"supprime", "supprimer", "efface", "effacer", "modifie",
               "modifier", "ajoute", "ajouter", "crée", "créer", "retire",
               "retirer"}


TYPE_WORDS = {
    "marché": "contrat", "marches": "contrat", "marche": "contrat",
    "acheteur": "acheteur", "acheteurs": "acheteur",
    "titulaire": "titulaire", "titulaires": "titulaire",
    "ville": "ville", "villes": "ville",
    "domaine": "domaine", "domaines": "domaine",
}


def link_entities(question: str, g: nx.MultiDiGraph,
                  include_types: bool = True) -> list[str]:
    """Match known node names inside the question (normalized, longest first).

    Also links every node whose *type* is named in the question
    ("quels titulaires…" → all titulaire nodes), unless include_types
    is False.
    """
    q = normalize(question)
    candidates = sorted(g.nodes, key=len, reverse=True)
    linked, covered = [], set()
    for name in candidates:
        key = normalize(name)
        if len(key) < 3 or key in covered:
            continue
        if key in q:
            linked.append(name)
            covered.add(key)
    if include_types:
        words = set(q.replace("’", " ").split())
        for word, ntype in TYPE_WORDS.items():
            if word in words:
                for n, d in g.nodes(data=True):
                    if d.get("type") == ntype and n not in linked:
                        linked.append(n)
    return linked


def _contracts_of(g: nx.MultiDiGraph, entity: str) -> list[str]:
    out = set()
    for u, v, d in g.edges(data=True):
        rel = d.get("relation")
        if rel == "ATTRIBUÉ_À" and v == entity:
            out.add(u)
        elif rel == "PASSÉ_PAR" and v == entity:
            out.add(u)
    return sorted(out)


def _amount_of(g: nx.MultiDiGraph, contract: str) -> float:
    for u, v, d in g.edges(data=True):
        if u == contract and d.get("relation") == "ATTRIBUÉ_À":
            return float(d.get("weight", 0.0))
    return 0.0


def _eur(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") + " €"


def _verbalize(g: nx.MultiDiGraph, u: str, v: str, d: dict) -> str:
    rel = d.get("relation", "?").replace("_", " ").lower()
    extra = ""
    if d.get("relation") == "ATTRIBUÉ_À" and d.get("weight"):
        extra = f" ({_eur(float(d['weight']) )})"
    return f"{u} — {rel} → {v}{extra}"


def _entity_total(g: nx.MultiDiGraph, entity: str) -> float:
    """Total € won (titulaire) or let (acheteur) by an entity."""
    ntype = g.nodes[entity].get("type")
    total = 0.0
    if ntype == "titulaire":
        for u, v, d in g.edges(data=True):
            if v == entity and d.get("relation") == "ATTRIBUÉ_À":
                total += float(d.get("weight", 0.0))
    elif ntype == "acheteur":
        for u, v, d in g.edges(data=True):
            if v == entity and d.get("relation") == "PASSÉ_PAR":
                total += _amount_of(g, u)
    return total


def local_evidence(question: str, g: nx.MultiDiGraph,
                   depth: int = config.LOCAL_DEPTH) -> Evidence:
    ev = Evidence(mode="local")
    named = link_entities(question, g, include_types=False)
    ev.linked = link_entities(question, g, include_types=True)
    seen_edges, seen_contracts = set(), set()
    explored = set(ev.linked)
    frontier = list(ev.linked)
    for _ in range(depth):
        nxt = []
        for node in frontier:
            for u, v, d in g.edges(data=True):
                if u != node and v != node:
                    continue
                key = (u, d.get("relation"), v)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                ev.facts.append(_verbalize(g, u, v, d))
                other = v if u == node else u
                explored.add(other)
                if other not in ev.linked:
                    nxt.append(other)
                for c in _contracts_of(g, other):
                    seen_contracts.add(c)
                if g.nodes[other].get("type") == "contrat":
                    seen_contracts.add(other)
        frontier = nxt
    for ent in ev.linked:
        for c in _contracts_of(g, ent):
            seen_contracts.add(c)
    # focused citations: the nearest ring of contracts around the *named*
    # entities (distance 1 if any, else 2, …); fall back to the whole
    # neighbourhood when nothing was named
    if named:
        ug = g.to_undirected()
        dist: dict[str, int] = {}
        for ent in named:
            for node, d in nx.single_source_shortest_path_length(
                    ug, ent, cutoff=3).items():
                if g.nodes[node].get("type") == "contrat":
                    dist[node] = min(dist.get(node, 99), d)
        if dist:
            nearest = min(dist.values())
            ev.contracts = sorted(c for c in dist if dist[c] == nearest)
        else:
            ev.contracts = sorted(seen_contracts)
    else:
        ev.contracts = sorted(seen_contracts)
    for ent in explored:
        if g.nodes[ent].get("type") in ("acheteur", "titulaire"):
            total = _entity_total(g, ent)
            if total:
                ev.amounts[ent] = total
    return ev


def global_evidence(question: str, g: nx.MultiDiGraph,
                    communities: list[set], summaries: dict[int, str],
                    top_k: int = 2) -> Evidence:
    ev = Evidence(mode="global")
    q = set(normalize(question).split())
    scored = []
    for i, summary in summaries.items():
        s = set(normalize(summary).split())
        overlap = len(q & s)
        # bonus for community size relevance is implicit in the summary
        scored.append((overlap, i))
    scored.sort(reverse=True)
    for overlap, i in scored[:top_k]:
        if overlap > 0 or top_k == len(summaries):
            ev.summaries.append(f"[Communauté {i}] {summaries[i]}")
            ev.contracts.extend(community_stats(g, communities[i])["contracts"])
    ev.contracts = sorted(set(ev.contracts))
    return ev


def hybrid_evidence(question: str, g: nx.MultiDiGraph,
                    retriever: TfidfRetriever,
                    communities: list[set], summaries: dict[int, str]
                    ) -> Evidence:
    ev = local_evidence(question, g)
    ev.mode = "hybrid"
    # TF-IDF chunks enrich the evidence text; citations stay the focused
    # graph-derived set so the answer doesn't list the whole corpus.
    for c, score in retriever.search(question):
        ev.chunks.append(f"[{c['id']}] {c['objet']} — {c['titulaire']} "
                         f"({_eur(c['montant_eur'])})")
    return ev


def render_factsheet(ev: Evidence) -> str:
    lines = []
    if ev.linked:
        lines.append("Entités reconnues : " + ", ".join(ev.linked) + ".")
    for f in ev.facts[:25]:
        lines.append("- " + f)
    if len(ev.facts) > 25:
        lines.append(f"… ({len(ev.facts) - 25} autres relations)")
    if ev.amounts:
        lines.append("Montants cumulés : " + "; ".join(
            f"{ent} : {_eur(t)}"
            for ent, t in sorted(ev.amounts.items(),
                                 key=lambda kv: kv[1], reverse=True)))
    for s in ev.summaries:
        lines.append("- " + s)
    for c in ev.chunks:
        lines.append("- " + c)
    if ev.contracts:
        lines.append("Marchés concernés : " + ", ".join(ev.contracts) + ".")
    return "\n".join(lines) if lines else "Aucun élément pertinent trouvé."


def answer(question: str, g: nx.MultiDiGraph, mode: str = "hybrid",
           communities: list[set] | None = None,
           summaries: dict[int, str] | None = None,
           retriever: TfidfRetriever | None = None,
           llm: LLMClient | None = None) -> QueryResult:
    t0 = time.time()
    # the query layer is read-only: write intents are refused, not executed
    if set(normalize(question).split()) & WRITE_VERBS:
        return QueryResult(question=question, mode=mode, answer="",
                           evidence=Evidence(mode=mode), refused=True,
                           refusal_reason="read-only graph: write refused",
                           latency_s=time.time() - t0)
    llm = llm or LLMClient()
    communities = communities or []
    summaries = summaries or {}

    if mode == "local":
        ev = local_evidence(question, g)
    elif mode == "global":
        ev = global_evidence(question, g, communities, summaries)
    elif mode == "hybrid":
        ev = hybrid_evidence(question, g, retriever, communities, summaries)
    else:
        raise ValueError(f"unknown mode: {mode}")

    if not ev.linked and not ev.facts and not ev.summaries and not ev.chunks:
        return QueryResult(question=question, mode=mode, answer="",
                           evidence=ev, refused=True,
                           refusal_reason="out of scope for this graph",
                           latency_s=time.time() - t0)

    factsheet = render_factsheet(ev)
    cost = 0.0
    if llm.enabled and (ev.facts or ev.summaries or ev.chunks):
        text, pt, ct = llm.complete(
            ANSWER_SYSTEM,
            f"Question : {question}\nFiche de faits :\n{factsheet}")
        cost = llm.estimate_cost(pt, ct)
        final = text.strip()
    else:
        final = factsheet  # offline: the fact sheet is the answer

    return QueryResult(question=question, mode=mode, answer=final,
                       evidence=ev, citations=list(ev.contracts),
                       latency_s=time.time() - t0, cost_usd=cost)
