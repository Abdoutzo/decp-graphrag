"""Eval harness: TF-IDF baseline vs local / global / hybrid graph search.

Fully offline (no LLM key needed). For each question and each mode it
measures:
- contracts recall: fraction of expected contract ids in the citations
- numbers: expected amounts present in the answer text
- entities: expected entity names present in the answer text
- keywords (global questions): expected keywords present
- latency

Adversarial questions must be refused. Writes evals/report.md.

Usage:
    python evals/run_eval.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.extract import extract_corpus, load_contracts
from src.graph import build_graph, detect_communities, load_graph, save_graph
from src.llm import LLMClient
from src.query import answer, normalize
from src.retrieve import TfidfRetriever, contract_text
from src.summarize import summarize_all

MODES = ["tfidf", "local", "global", "hybrid"]


def load_questions(path: str = "evals/questions.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def score_answer(q: dict, ans_text: str, citations: list) -> dict:
    """Returns per-aspect 0/1 (or recall) scores for the expectations."""
    out = {}
    if "expected_contracts" in q:
        exp = set(q["expected_contracts"])
        got = set(citations) & exp
        out["contracts_recall"] = len(got) / len(exp) if exp else 1.0
    if "expected_numbers" in q:
        flat = digits(ans_text)
        out["numbers"] = float(all(str(n) in flat
                                   for n in q["expected_numbers"]))
    if "expected_entities" in q:
        norm = normalize(ans_text)
        out["entities"] = float(all(normalize(e) in norm
                                    for e in q["expected_entities"]))
    if "expected_keywords" in q:
        norm = normalize(ans_text)
        hits = sum(1 for k in q["expected_keywords"]
                   if normalize(k) in norm)
        out["keywords_recall"] = hits / len(q["expected_keywords"])
    return out


def ensure_graph():
    if os.path.exists(config.GRAPH_PATH):
        return load_graph(config.GRAPH_PATH)
    contracts = load_contracts(config.DATA_PATH)
    g = build_graph(extract_corpus(contracts))
    communities = detect_communities(g)
    summaries, _ = summarize_all(g, communities, LLMClient())
    save_graph(g, config.GRAPH_PATH, communities, summaries)
    return g, communities, summaries


def run():
    contracts = load_contracts(config.DATA_PATH)
    g, communities, summaries = ensure_graph()
    retriever = TfidfRetriever(contracts)
    llm = LLMClient()
    questions = load_questions()

    results = []
    for q in questions:
        row = {"id": q["id"], "type": q["type"], "question": q["question"]}
        if q.get("expected_refusal"):
            refused = all(
                answer(q["question"], g,
                       mode=m if m != "tfidf" else "local",
                       communities=communities, summaries=summaries,
                       retriever=retriever, llm=llm).refused
                for m in MODES)
            row.update(status="PASS" if refused else "FAIL",
                       detail="refused in all modes" if refused
                       else "not refused!")
            results.append(row)
            continue
        for mode in MODES:
            t0 = time.time()
            if mode == "tfidf":
                hits = retriever.search(q["question"])
                citations = [c["id"] for c, _ in hits]
                text = " ".join(contract_text(c) for c, _ in hits)
            else:
                res = answer(q["question"], g, mode=mode,
                             communities=communities, summaries=summaries,
                             retriever=retriever, llm=llm)
                citations, text = res.citations, res.answer
            latency = time.time() - t0
            scores = score_answer(q, text, citations)
            row[mode] = {"scores": scores,
                         "latency": round(latency, 3),
                         "citations": citations}
        # question-level verdict: hybrid must reach full contracts recall
        # on non-global questions
        if q["type"] != "global":
            rec = row["hybrid"]["scores"].get("contracts_recall", 0)
            row.update(status="PASS" if rec == 1.0 else "FAIL",
                       detail=f"hybrid contracts recall={rec:.2f}")
        else:
            rec = row["hybrid"]["scores"].get("keywords_recall", 0)
            row.update(status="PASS" if rec >= 0.6 else "FAIL",
                       detail=f"hybrid keywords recall={rec:.2f}")
        results.append(row)
    return results


def write_report(results, path: str = "evals/report.md"):
    valid = [r for r in results if r["type"] != "adversarial"]
    adv = [r for r in results if r["type"] == "adversarial"]
    passed = sum(1 for r in results if r["status"] == "PASS")
    lines = [
        "# Eval report",
        "",
        f"_GraphRAG eval: {len(results)} questions "
        f"({len(valid)} retrieval + {len(adv)} adversarial), "
        f"{passed}/{len(results)} passed. Offline, no API key._",
        "",
        "## Per-mode contracts recall (non-global questions)",
        "",
        "| id | type | tfidf | local | global | hybrid |",
        "|---|---|---|---|---|---|",
    ]
    for r in valid:
        if r["type"] == "global":
            continue
        cells = []
        for m in MODES:
            v = r[m]["scores"].get("contracts_recall", float("nan"))
            cells.append(f"{v:.2f}")
        lines.append(f"| {r['id']} | {r['type']} | " + " | ".join(cells)
                     + f" | {r['status']} |")
    lines += [
        "",
        "## Latency (seconds, mean over questions)",
        "",
        "| mode | mean latency |",
        "|---|---|",
    ]
    for m in MODES:
        lat = [r[m]["latency"] for r in valid]
        lines.append(f"| {m} | {sum(lat) / len(lat):.3f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- `tfidf` is a sparse baseline over contract texts (the sibling "
        "project already covers dense embeddings).",
        "- `local` answers from the entity neighbourhood; `global` from "
        "community summaries; `hybrid` combines local evidence with TF-IDF.",
        "- Multi-hop questions (g05, g07) are where the graph structurally "
        "beats chunk retrieval: the answer requires traversing "
        "ville → titulaire → acheteur.",
        "- Adversarial questions must be refused in every mode.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{passed}/{len(results)} passed — report written to {path}")


def main():
    results = run()
    for r in results:
        print(f"{r['id']:4} {r['status']:4} {r['detail']}")
    write_report(results)


if __name__ == "__main__":
    main()
