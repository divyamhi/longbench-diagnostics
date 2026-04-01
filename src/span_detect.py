from __future__ import annotations

from typing import Any

import nltk
from nltk.tokenize import sent_tokenize
from rank_bm25 import BM25Okapi


def _ensure_punkt() -> None:
    # NLTK 3.8+ may require both punkt and punkt_tab
    for resource in ("punkt", "punkt_tab"):
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download(resource, quiet=True)


def segment_sentences(text: str) -> list[str]:
    _ensure_punkt()
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def find_relevant_span(instance: dict[str, Any], threshold: float = 0.5) -> dict[str, Any]:
    if instance.get("category") != "single-document QA":
        raise ValueError("Only valid for single-document QA")

    sentences = segment_sentences(instance["input"])
    if not sentences:
        return {
            "span_text": "",
            "sentence_index": -1,
            "score": 0.0,
            "flagged": True,
        }

    tokenized_corpus = [sentence.lower().split() for sentence in sentences]
    bm25 = BM25Okapi(tokenized_corpus)

    query = instance["question"].lower().split()
    scores = bm25.get_scores(query)

    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    max_score = float(scores.max()) if len(scores) else 0.0
    normalized_score = best_score / max_score if max_score > 0 else 0.0

    return {
        "span_text": sentences[best_idx],
        "sentence_index": best_idx,
        "score": float(normalized_score),
        "flagged": normalized_score < threshold,
    }