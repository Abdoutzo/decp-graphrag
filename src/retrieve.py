"""TF-IDF baseline retriever.

The honest baseline for the eval: classic sparse retrieval over the
contract texts. The sibling project already proved dense embeddings;
here the question is whether the graph adds anything over a strong
cheap baseline — especially on multi-hop questions where chunk
retrieval structurally struggles.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


def contract_text(c: dict) -> str:
    return " ".join([
        c["id"], c["objet"], c["description"], c["acheteur"],
        c["titulaire"], c.get("ville_titulaire", ""), c["cpv"],
        str(c["montant_eur"]),
    ])


class TfidfRetriever:
    def __init__(self, contracts: list[dict]):
        self.contracts = contracts
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [contract_text(c) for c in contracts])

    def search(self, query: str, k: int = config.TFIDF_TOP_K
               ) -> list[tuple[dict, float]]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix)[0]
        ranked = sorted(range(len(self.contracts)),
                        key=lambda i: scores[i], reverse=True)[:k]
        return [(self.contracts[i], float(scores[i])) for i in ranked
                if scores[i] > 0]
