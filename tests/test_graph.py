"""Graph unit tests. Run with: pytest tests/ -q (no API key needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.extract import extract_corpus, load_contracts
from src.graph import build_graph, detect_communities
from src.query import answer, link_entities
from src.retrieve import TfidfRetriever
from src.summarize import summarize_all


def _graph():
    contracts = load_contracts(config.DATA_PATH)
    return build_graph(extract_corpus(contracts)), contracts


def test_graph_has_expected_nodes():
    g, _ = _graph()
    assert g.number_of_nodes() >= 30  # 14 contracts + buyers + winners + misc
    assert g.has_node("BatiRhône SAS")
    assert g.nodes["BatiRhône SAS"]["type"] == "titulaire"


def test_batirhone_has_three_contracts():
    g, _ = _graph()
    contracts = [u for u, v, d in g.edges(data=True)
                 if v == "BatiRhône SAS"
                 and d.get("relation") == "ATTRIBUÉ_À"]
    assert sorted(contracts) == ["2024-001", "2024-006", "2024-009"]


def test_total_amount_batirhone():
    g, _ = _graph()
    total = sum(float(d["weight"]) for u, v, d in g.edges(data=True)
                if v == "BatiRhône SAS"
                and d.get("relation") == "ATTRIBUÉ_À")
    assert total == 1240000 + 3850000 + 2100000


def test_communities_cover_all_nodes():
    g, _ = _graph()
    communities = detect_communities(g)
    covered = set().union(*communities)
    assert covered == set(g.nodes)
    assert len(communities) >= 2  # buyers/winners cluster apart


def test_communities_deterministic():
    g, _ = _graph()
    a = sorted(sorted(c) for c in detect_communities(g))
    b = sorted(sorted(c) for c in detect_communities(g))
    assert a == b


def test_entity_linking():
    g, _ = _graph()
    linked = link_entities("Quels marchés ont été attribués à BatiRhône SAS ?", g)
    assert "BatiRhône SAS" in linked


def test_local_evidence_finds_contracts():
    g, _ = _graph()
    res = answer("Quels marchés ont été attribués à BatiRhône SAS ?",
                 g, mode="local")
    assert res.citations == ["2024-001", "2024-006", "2024-009"]
    assert "7 190 000 €" in res.answer


def test_global_evidence_returns_summaries():
    g, _ = _graph()
    communities = detect_communities(g)
    summaries, _ = summarize_all(g, communities)
    res = answer("Quels sont les grands domaines d'achat ?", g,
                 mode="global", communities=communities, summaries=summaries)
    assert res.evidence.summaries, "global search should return summaries"


def test_hybrid_combines_sources():
    g, contracts = _graph()
    communities = detect_communities(g)
    summaries, _ = summarize_all(g, communities)
    retr = TfidfRetriever(contracts)
    res = answer("marchés de restauration scolaire", g, mode="hybrid",
                 communities=communities, summaries=summaries,
                 retriever=retr)
    assert res.evidence.chunks, "hybrid should include TF-IDF chunks"
    assert "2024-002" in res.citations or "2024-010" in res.citations


def test_tfidf_baseline_finds_repas():
    _, contracts = _graph()
    retr = TfidfRetriever(contracts)
    hits = retr.search("fourniture de repas pour les collèges")
    assert hits and hits[0][0]["id"] in ("2024-002", "2024-010")


def test_summaries_offline_are_free_and_deterministic():
    g, _ = _graph()
    communities = detect_communities(g)
    s1, c1 = summarize_all(g, communities)
    s2, c2 = summarize_all(g, communities)
    assert c1 == c2 == 0.0
    assert s1 == s2
    assert all("Communauté" in s for s in s1.values())
