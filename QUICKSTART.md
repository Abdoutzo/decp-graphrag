# QUICKSTART

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Build the knowledge graph

```bash
python scripts/build_graph.py
```

Deterministic (Louvain seed 42): ~40 nodes, ~100 relations, a handful of
communities. Saved to `data/graph.json` (gitignored).

With an API key you can also enrich the graph from the free-text
descriptions:

```bash
python scripts/build_graph.py --enrich   # needs MISTRAL_API_KEY or OPENAI_API_KEY
```

## 3. Run the offline evals (no API key needed)

```bash
pytest tests/ -q
python evals/run_eval.py
```

Compares TF-IDF baseline vs local / global / hybrid graph search on
12 questions (10 retrieval + 2 adversarial) and writes `evals/report.md`.

## 4. Try the demo

```bash
streamlit run app.py
```

Without an API key, answers are deterministic fact sheets; with a key
(`.env`), the LLM turns them into prose with citations. Good questions
to try:

- Quels marchés ont été attribués à BatiRhône SAS ?
- Quels acheteurs ont fait appel à un titulaire basé à Villeurbanne ?
- Quels sont les grands domaines d'achat de ce corpus ?
