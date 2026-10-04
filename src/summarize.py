"""Community summaries for global search.

Two modes: a deterministic template summary (always available, used by
offline evals and the no-key demo) and an LLM summary (richer prose when
a key is configured). Both are stored on the graph so the demo loads
them without recomputing.
"""
import networkx as nx

from src.graph import community_stats, normalize
from src.llm import LLMClient


def _eur(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ") + " €"


def template_summary(g: nx.MultiDiGraph, community: set, idx: int) -> str:
    stats = community_stats(g, community)
    titulaires = sorted(m for m in community
                        if g.nodes[m].get("type") == "titulaire")
    acheteurs = sorted(m for m in community
                       if g.nodes[m].get("type") == "acheteur")
    domaines = sorted(m for m in community
                      if g.nodes[m].get("type") == "domaine")
    parts = [f"Communauté {idx} : {stats['size']} entités"]
    if acheteurs:
        parts.append("acheteurs : " + ", ".join(acheteurs))
    if titulaires:
        parts.append("titulaires : " + ", ".join(titulaires))
    if domaines:
        parts.append("domaines : " + ", ".join(domaines))
    if stats["contracts"]:
        parts.append(f"{len(stats['contracts'])} marchés "
                     f"({', '.join(stats['contracts'])})")
    if stats["total_amount"]:
        parts.append(f"montant cumulé {_eur(stats['total_amount'])}")
    return ". ".join(parts) + "."


SUMMARY_SYSTEM = """Tu résumes une communauté d'un graphe de marchés publics français
en 3-4 phrases : qui achète, qui fournit, dans quels domaines, quels montants.
Reste factuel, ne rajoute rien."""


def llm_summary(g: nx.MultiDiGraph, community: set, idx: int,
                llm: LLMClient) -> tuple[str, float]:
    stats = community_stats(g, community)
    members = "\n".join(
        f"- {m} ({g.nodes[m].get('type')})" for m in sorted(community))
    user = (f"Membres de la communauté {idx} :\n{members}\n"
            f"Contrats : {', '.join(stats['contracts']) or 'aucun'}\n"
            f"Montant cumulé : {_eur(stats['total_amount'])}")
    text, pt, ct = llm.complete(SUMMARY_SYSTEM, user)
    return text.strip(), llm.estimate_cost(pt, ct)


def summarize_all(g: nx.MultiDiGraph, communities: list[set],
                  llm: LLMClient | None = None) -> tuple[dict[int, str], float]:
    """Returns {community_idx: summary} and the total LLM cost (0 offline)."""
    llm = llm or LLMClient()
    summaries: dict[int, str] = {}
    cost = 0.0
    for i, c in enumerate(communities):
        if llm.enabled:
            try:
                s, ccost = llm_summary(g, c, i, llm)
                cost += ccost
            except Exception:
                s = template_summary(g, c, i)
        else:
            s = template_summary(g, c, i)
        summaries[i] = s
    return summaries, cost
