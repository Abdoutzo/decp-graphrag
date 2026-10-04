"""Knowledge extraction from contracts.

Two layers, deliberately separated:

1. Structured extraction (deterministic, always runs): the corpus is
   structured JSON, so contracts, buyers, winners, CPV codes and cities
   become typed nodes and edges with zero hallucination risk.
2. Text enrichment (LLM, optional): extra entities mentioned in the free
   description (places, equipment, quantities). Skipped without an API
   key — the graph stays fully functional on the structured layer.
"""
import json
import re
from dataclasses import dataclass, field

from src.llm import LLMClient

# CPV division labels (first 2 digits) for the divisions in this corpus.
CPV_LABELS = {
    "30": "Matériel informatique",
    "35": "Équipement de sécurité",
    "45": "Travaux de construction",
    "50": "Services de maintenance",
    "55": "Services de restauration",
    "60": "Services de transport",
    "71": "Services d'ingénierie",
    "77": "Services paysagers",
    "90": "Services de nettoyage",
}


@dataclass
class Triple:
    source: str
    source_type: str
    relation: str
    target: str
    target_type: str
    weight: float = 1.0
    provenance: str = "structured"  # structured | text


@dataclass
class Extraction:
    nodes: dict = field(default_factory=dict)  # name -> type
    triples: list = field(default_factory=list)  # list[Triple]

    def add_node(self, name: str, ntype: str):
        name = name.strip()
        if name and name not in self.nodes:
            self.nodes[name] = ntype

    def add_triple(self, source: str, stype: str, relation: str,
                   target: str, ttype: str, weight: float = 1.0,
                   provenance: str = "structured"):
        self.add_node(source, stype)
        self.add_node(target, ttype)
        self.triples.append(Triple(source.strip(), stype, relation,
                                   target.strip(), ttype, weight, provenance))


def extract_structured(contract: dict) -> Extraction:
    """Deterministic extraction from the structured contract fields."""
    ex = Extraction()
    cid = contract["id"]
    ex.add_node(cid, "contrat")

    acheteur = contract["acheteur"]
    titulaire = contract["titulaire"]
    montant = float(contract["montant_eur"])
    ex.add_triple(cid, "contrat", "PASSÉ_PAR", acheteur, "acheteur")
    ex.add_triple(cid, "contrat", "ATTRIBUÉ_À", titulaire, "titulaire",
                  weight=montant)
    # aggregated buyer -> winner edge (weighted by amount)
    ex.add_triple(acheteur, "acheteur", "A_ATTRIBUÉ", titulaire, "titulaire",
                  weight=montant)

    cpv = contract["cpv"]
    division = cpv[:2]
    label = CPV_LABELS.get(division, f"CPV {division}")
    ex.add_triple(cid, "contrat", "CATÉGORIE", label, "domaine")

    ville = contract.get("ville_titulaire", "").strip()
    if ville:
        ex.add_triple(titulaire, "titulaire", "BASÉ_À", ville, "ville")
    return ex


def extract_corpus(contracts: list[dict]) -> Extraction:
    merged = Extraction()
    for c in contracts:
        ex = extract_structured(c)
        for name, ntype in ex.nodes.items():
            merged.add_node(name, ntype)
        merged.triples.extend(ex.triples)
    return merged


ENRICH_SYSTEM = """Tu extrais des entités d'une description de marché public français.
Réponds UNIQUEMENT avec des lignes au format :
entité | type | relation | entité_liée
où type ∈ {lieu, équipement, quantité, organisme} et relation décrit le lien
avec le marché (ex: CONCERNE, SITUÉ_À, QUANTITÉ_DE).
Maximum 6 lignes, pas de doublons, rien d'autre."""


def enrich_text(contract: dict, llm: LLMClient | None = None) -> Extraction:
    """LLM enrichment from the free-text description. Empty without a key."""
    llm = llm or LLMClient()
    ex = Extraction()
    if not llm.enabled:
        return ex
    user = (f"Marché {contract['id']} — {contract['objet']} :\n"
            f"{contract['description']}")
    text, _, _ = llm.complete(ENRICH_SYSTEM, user)
    cid = contract["id"]
    for line in text.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        ent, etype, rel, _linked = parts
        if etype not in {"lieu", "équipement", "quantité", "organisme"}:
            continue
        ex.add_triple(cid, "contrat", rel.upper().replace(" ", "_"),
                      ent, etype, provenance="text")
    return ex


def load_contracts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
