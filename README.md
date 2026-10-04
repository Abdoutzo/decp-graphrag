# decp-graphrag

GraphRAG over French public procurement contracts: instead of retrieving
text chunks, this project builds a **knowledge graph** from the contracts
(buyers, winners, amounts, CPV domains, cities), detects communities with
Louvain, and answers questions by traversing the graph.

![demo walkthrough](assets/demo.svg)

## Why this exists

Classic RAG answers "find me the paragraph about X" well, but collapses on
questions like *"which buyers hired a contractor based in Villeurbanne?"* —
that requires hopping ville → titulaire → acheteur, and no single chunk
contains the answer. A knowledge graph turns those multi-hop questions
into traversals, and community summaries answer holistic questions
(*"what are the big procurement themes?"*) that chunk retrieval can't
even attempt.

The part I'm proudest of: the whole thing runs **deterministically offline**.
Extraction from the structured fields, Louvain communities (seed 42),
template summaries and fact-sheet answers need no API key — the LLM only
adds prose on top. Every number in the eval report is reproducible.

## What it does

- **Extraction** (`src/extract.py`): deterministic triples from the
  structured contract fields (contrat —ATTRIBUÉ_À→ titulaire,
  acheteur —A_ATTRIBUÉ→ titulaire, …) + optional LLM enrichment from the
  free-text descriptions
- **Graph** (`src/graph.py`): NetworkX MultiDiGraph, accent-insensitive
  entity resolution, Louvain communities (47 nodes, 70 edges, 6 communities)
- **Summaries** (`src/summarize.py`): per-community summaries — LLM prose
  with a key, deterministic templates offline
- **Search modes** (`src/query.py`):
  - *local* — link question entities, return the 2-hop neighbourhood as a
    verbalized fact sheet with per-entity totals
  - *global* — match the question against community summaries
  - *hybrid* — local evidence + TF-IDF contract texts
- **Refusals**: the query layer is read-only (write intents rejected) and
  out-of-scope questions get no answer instead of a hallucinated one

## Results

Offline eval: 12 questions (10 retrieval + 2 adversarial), TF-IDF baseline
vs graph modes, contracts recall per question.

| id | type | tfidf | local | hybrid |
|---|---|---|---|---|
| g06 | comparison (biggest cumulative winner) | 0.00 | 1.00 | 1.00 |
| g07 | multi-hop (most diverse buyers) | 0.33 | 1.00 | 1.00 |
| g05 | multi-hop (buyers via contractor city) | 1.00 | 1.00 | 1.00 |
| g10 | one-hop (how many construction contracts) | 0.80 | 1.00 | 1.00 |

The pattern is the point: the baseline is competitive on simple lookups,
but aggregation and multi-hop questions structurally need the graph.
Full table in [evals/report.md](evals/report.md) — run
`python evals/run_eval.py` to reproduce.

## Architecture

```text
contracts.json
   │  src/extract.py — deterministic triples (+ LLM enrichment --enrich)
   ▼
src/graph.py — MultiDiGraph, entity resolution, Louvain communities
   │  src/summarize.py — community summaries
   ▼
question ──► src/query.py — entity linking
   ├── local:  neighbourhood → fact sheet (+ totals)
   ├── global: community summaries
   └── hybrid: local + TF-IDF texts
   ▼
answer: fact sheet (offline) or LLM prose with citations (with API key)
```

## Quickstart

See [QUICKSTART.md](QUICKSTART.md). The short version:

```bash
pip install -r requirements.txt
python scripts/build_graph.py   # deterministic, seed 42
pytest tests/ -q                # 11 tests
python evals/run_eval.py        # offline evals, no API key needed
streamlit run app.py            # demo: Q&A + interactive graph explorer
```

The demo above was recorded with `scripts/record_demo.py`
(Playwright → animated SVG), running with no API key.

## Repo map

```text
├── app.py                  # Streamlit demo (Q&A + graph explorer)
├── config.py               # env-overridable settings
├── data/
│   ├── contracts.json      # 14 synthetic French procurement contracts
│   └── graph.json          # built graph (generated, gitignored)
├── src/
│   ├── extract.py          # structured + LLM text extraction
│   ├── graph.py            # build, resolve, communities, persistence
│   ├── summarize.py        # community summaries (LLM or template)
│   ├── retrieve.py         # TF-IDF baseline
│   ├── query.py            # local / global / hybrid engines
│   ├── viz.py              # Plotly network visualization
│   └── llm.py              # minimal Mistral/OpenAI client
├── evals/                  # questions.jsonl, run_eval.py, report.md
├── scripts/
│   ├── build_graph.py
│   └── record_demo.py
└── tests/                  # 11 unit tests
```
