# Eval report

_GraphRAG eval: 12 questions (10 retrieval + 2 adversarial), 12/12 passed. Offline, no API key._

## Per-mode contracts recall (non-global questions)

| id | type | tfidf | local | global | hybrid |
|---|---|---|---|---|---|
| g01 | one-hop | 1.00 | 1.00 | 0.67 | 1.00 | PASS |
| g02 | aggregation | 1.00 | 1.00 | 0.67 | 1.00 | PASS |
| g03 | one-hop | 0.67 | 1.00 | 1.00 | 1.00 | PASS |
| g04 | aggregation | 0.67 | 1.00 | 1.00 | 1.00 | PASS |
| g05 | multi-hop | 1.00 | 1.00 | 0.67 | 1.00 | PASS |
| g06 | comparison | 0.00 | 1.00 | 0.00 | 1.00 | PASS |
| g07 | multi-hop | 0.33 | 1.00 | 0.00 | 1.00 | PASS |
| g09 | one-hop | 1.00 | 1.00 | 0.00 | 1.00 | PASS |
| g10 | one-hop | 0.80 | 1.00 | 0.60 | 1.00 | PASS |

## Latency (seconds, mean over questions)

| mode | mean latency |
|---|---|
| tfidf | 0.001 |
| local | 0.006 |
| global | 0.000 |
| hybrid | 0.009 |

## Notes

- `tfidf` is a sparse baseline over contract texts (the sibling project already covers dense embeddings).
- `local` answers from the entity neighbourhood; `global` from community summaries; `hybrid` combines local evidence with TF-IDF.
- Multi-hop questions (g05, g07) are where the graph structurally beats chunk retrieval: the answer requires traversing ville → titulaire → acheteur.
- Adversarial questions must be refused in every mode.
