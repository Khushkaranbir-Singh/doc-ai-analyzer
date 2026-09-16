"""Summaries and bullet points.

Local path: a TextRank-style ranking over the document's own sentences.
AI path: the same document sent to Claude for an abstractive summary.
"""

from __future__ import annotations

from typing import Dict, List

from .analyze import content_words, split_sentences, tokenize
from . import llm

LENGTH_PRESETS = {"Short": 4, "Medium": 7, "Detailed": 12}


def _rank_sentences(sentences: List[str]) -> List[float]:
    """PageRank over a sentence-similarity graph, with a frequency fallback."""
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(sentences)
        similarity = (matrix @ matrix.T).toarray()
        np.fill_diagonal(similarity, 0.0)
        row_sums = similarity.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        transition = similarity / row_sums

        n = len(sentences)
        scores = np.full(n, 1.0 / n)
        for _ in range(60):
            updated = 0.15 / n + 0.85 * transition.T.dot(scores)
            if float(np.abs(updated - scores).sum()) < 1e-7:
                scores = updated
                break
            scores = updated
        return scores.tolist()
    except Exception:
        from collections import Counter

        frequencies = Counter(content_words(tokenize(" ".join(sentences))))
        top = dict(frequencies.most_common(300))
        return [
            sum(top.get(w, 0) for w in content_words(tokenize(s))) / max(len(s.split()), 1)
            for s in sentences
        ]


def extractive_summary(text: str, max_sentences: int = 7) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= max_sentences:
        return sentences

    working = sentences[:1200]
    scores = _rank_sentences(working)
    # Sentences that open a document usually carry its thesis.
    for i in range(min(3, len(scores))):
        scores[i] *= 1.15

    chosen = sorted(range(len(working)), key=lambda i: scores[i], reverse=True)[:max_sentences]
    return [working[i] for i in sorted(chosen)]


def key_points(text: str, count: int = 8) -> List[str]:
    """Short, scannable bullets built from the highest-signal sentences."""
    points = []
    for sentence in extractive_summary(text, max_sentences=count * 2):
        clean = " ".join(sentence.split())
        if len(clean) > 240:
            clean = clean[:237].rsplit(" ", 1)[0] + "…"
        if len(clean.split()) >= 5 and clean not in points:
            points.append(clean)
        if len(points) == count:
            break
    return points


def ai_summary(text: str, length: str = "Medium") -> Dict[str, object]:
    """Ask Claude for an abstractive summary. Caller handles exceptions."""
    budget = {"Short": "about 90 words", "Medium": "about 180 words", "Detailed": "about 350 words"}
    excerpt = text[:120_000]
    prompt = (
        "Summarise the document below.\n\n"
        f"1. An overview paragraph of {budget.get(length, 'about 180 words')}.\n"
        "2. A line starting with 'KEY POINTS:' followed by one bullet per line, "
        "each beginning with '- '. Six to nine bullets.\n"
        "Use only what the document says. Plain sentences, no preamble.\n\n"
        f"---\n{excerpt}\n---"
    )
    raw = llm.complete(prompt, system="You summarise documents accurately and plainly.")

    overview, bullets = raw, []
    if "KEY POINTS:" in raw.upper():
        marker = raw.upper().index("KEY POINTS:")
        overview = raw[:marker].strip()
        bullets = [
            line.lstrip("-•* ").strip()
            for line in raw[marker:].splitlines()[1:]
            if line.strip().startswith(("-", "•", "*"))
        ]
    return {"overview": overview.strip(), "bullets": bullets, "engine": "Claude"}


def build_summary(text: str, length: str = "Medium", use_ai: bool = True) -> Dict[str, object]:
    """Preferred entry point: AI when available, extractive otherwise."""
    sentence_budget = LENGTH_PRESETS.get(length, 7)

    if use_ai and llm.ai_available():
        try:
            result = ai_summary(text, length)
            if result["overview"]:
                if not result["bullets"]:
                    result["bullets"] = key_points(text, sentence_budget)
                return result
        except Exception as exc:
            fallback = _extractive_result(text, sentence_budget)
            fallback["engine"] = f"Local ranking (Claude call failed: {exc})"
            return fallback

    return _extractive_result(text, sentence_budget)


def _extractive_result(text: str, sentence_budget: int) -> Dict[str, object]:
    sentences = extractive_summary(text, sentence_budget)
    return {
        "overview": " ".join(sentences),
        "bullets": key_points(text, max(6, sentence_budget)),
        "engine": "Local ranking",
    }
