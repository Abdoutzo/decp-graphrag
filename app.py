"""Streamlit demo: GraphRAG over French public procurement contracts.

Run locally:  streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import config
from src import viz
from src.extract import extract_corpus, load_contracts
from src.graph import (build_graph, detect_communities, load_graph,
                       save_graph)
from src.llm import LLMClient
from src.query import answer
from src.retrieve import TfidfRetriever
from src.summarize import summarize_all


@st.cache_resource
def load_all():
    if not os.path.exists(config.GRAPH_PATH):
        contracts = load_contracts(config.DATA_PATH)
        g = build_graph(extract_corpus(contracts))
        communities = detect_communities(g)
        summaries, _ = summarize_all(g, communities, LLMClient())
        save_graph(g, config.GRAPH_PATH, communities, summaries)
    else:
        g, communities, summaries = load_graph(config.GRAPH_PATH)
        contracts = load_contracts(config.DATA_PATH)
    retriever = TfidfRetriever(contracts)
    return g, communities, summaries, retriever, contracts


st.set_page_config(page_title="GraphRAG — marchés publics", layout="wide")
st.title("GraphRAG : interroger un graphe de marchés publics")

g, communities, summaries, retriever, contracts = load_all()
llm = LLMClient()

with st.sidebar:
    st.header("Réglages")
    mode_labels = {"hybrid": "Hybride (graphe + texte)",
                   "local": "Local (voisinage d'entités)",
                   "global": "Global (résumés de communautés)"}
    mode = st.radio("Mode de recherche",
                    ["hybrid", "local", "global"],
                    format_func=mode_labels.__getitem__,
                    help="Local : répond depuis le voisinage des entités citées. "
                         "Global : répond depuis les résumés de communautés. "
                         "Hybride : combine les deux avec une recherche TF-IDF.")
    st.caption(f"Graphe : {g.number_of_nodes()} nœuds, "
               f"{g.number_of_edges()} relations, "
               f"{len(communities)} communautés.")
    if not llm.enabled:
        st.info("Pas de clé API : les réponses sont des fiches de faits "
                "déterministes. Ajoutez MISTRAL_API_KEY ou OPENAI_API_KEY "
                "dans .env pour la synthèse en langage naturel.")

tab_qa, tab_graph = st.tabs(["Poser une question", "Explorer le graphe"])

with tab_qa:
    examples = [
        "Quels marchés ont été attribués à BatiRhône SAS ?",
        "Quels acheteurs ont fait appel à un titulaire basé à Villeurbanne ?",
        "Quel titulaire a remporté le plus gros montant cumulé ?",
        "Quels sont les grands domaines d'achat de ce corpus ?",
    ]
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Votre question", examples[1])
    with col2:
        st.write("")
        st.write("")
        ask_clicked = st.button("Analyser", type="primary")
    st.caption("Exemples : " + " · ".join(f"« {e} »" for e in examples))

    if ask_clicked and question:
        with st.spinner("Interrogation du graphe…"):
            res = answer(question, g, mode=mode, communities=communities,
                         summaries=summaries, retriever=retriever, llm=llm)
        if res.refused:
            st.warning(f"Je ne peux pas répondre : {res.refusal_reason}")
        else:
            st.subheader("Réponse")
            st.write(res.answer)
            if res.citations:
                st.caption("Sources : " + ", ".join(res.citations))
            with st.expander("Voir les faits utilisés"):
                for f in res.evidence.facts[:20]:
                    st.markdown(f"- {f}")
                for s in res.evidence.summaries:
                    st.markdown(f"- {s}")
            st.caption(f"Mode : {res.mode} · latence : {res.latency_s:.2f}s"
                       + (f" · coût estimé : ${res.cost_usd:.5f}"
                          if res.cost_usd else ""))

with tab_graph:
    st.subheader("Le graphe de connaissances")
    st.caption("Contrats, acheteurs, titulaires, domaines et villes — "
               "construit par extraction déterministe, communautés "
               "détectées par Louvain. Survolez un nœud pour ses relations.")
    st.plotly_chart(viz.figure(g), use_container_width=True,
                    config={"displayModeBar": False})
    with st.expander("Résumés des communautés (recherche globale)"):
        for i in sorted(summaries):
            st.markdown(f"**Communauté {i}** — {summaries[i]}")
