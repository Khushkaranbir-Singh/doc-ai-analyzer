"""Counting, keywords and readability — the numbers behind every chart."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

STOPWORDS = set(
    """a about above after again against all am an and any are aren't as at be because been
before being below between both but by can cannot could couldn't did didn't do does doesn't
doing don't down during each few for from further had hadn't has hasn't have haven't having he
her here hers herself him himself his how i if in into is isn't it its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own
same shan't she should shouldn't so some such than that the their theirs them themselves then
there these they this those through to too under until up very was wasn't we were weren't what
when where which while who whom why with won't would wouldn't you your yours yourself yourselves
also may might shall upon within without per via etc et al fig figure table section chapter
one two three four five six seven eight nine ten first second third new use used using make made
many much well even still just like""".split()
)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])[\s\n]+(?=[A-Z0-9\"'(])")


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts: List[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        parts.extend(s.strip() for s in SENTENCE_RE.split(block) if s.strip())
    return [p for p in parts if len(p.split()) >= 3]


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def content_words(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def count_syllables(word: str) -> int:
    word = word.lower().strip("'-")
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def flesch_reading_ease(words: List[str], sentence_count: int) -> float:
    if not words or sentence_count == 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    return round(
        206.835 - 1.015 * (len(words) / sentence_count) - 84.6 * (syllables / len(words)), 1
    )


def reading_level(score: float) -> str:
    bands = [
        (90, "Very easy — around 5th grade"),
        (80, "Easy — around 6th grade"),
        (70, "Fairly easy — around 7th grade"),
        (60, "Plain English — 8th to 9th grade"),
        (50, "Fairly hard — high school"),
        (30, "Hard — university level"),
        (0, "Very hard — professional or academic"),
    ]
    for threshold, label in bands:
        if score >= threshold:
            return label
    return "Very hard — professional or academic"


def keyword_scores(tokens: List[str], top_n: int = 25) -> List[Tuple[str, int]]:
    return Counter(content_words(tokens)).most_common(top_n)


def phrase_scores(tokens: List[str], size: int = 2, top_n: int = 15) -> List[Tuple[str, int]]:
    useful = [t if t not in STOPWORDS and len(t) > 2 else None for t in tokens]
    grams: Counter = Counter()
    for i in range(len(useful) - size + 1):
        window = useful[i : i + size]
        if all(window):
            grams[" ".join(window)] += 1
    return [(g, c) for g, c in grams.most_common(top_n) if c > 1]


def analyze_text(text: str, pages: List[str] | None = None) -> Dict:
    pages = pages or []
    tokens = tokenize(text)
    sentences = split_sentences(text)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    unique = set(tokens)
    sentence_lengths = [len(s.split()) for s in sentences]
    word_lengths = [len(w) for w in tokens]
    score = flesch_reading_ease(tokens, max(len(sentences), 1))

    longest_idx = (
        max(range(len(sentence_lengths)), key=lambda i: sentence_lengths[i])
        if sentence_lengths
        else None
    )

    return {
        "word_count": len(tokens),
        "character_count": len(text),
        "character_count_no_spaces": len(re.sub(r"\s", "", text)),
        "unique_words": len(unique),
        "lexical_diversity": round(len(unique) / len(tokens), 3) if tokens else 0.0,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "page_count": len(pages),
        "avg_sentence_length": round(sum(sentence_lengths) / len(sentence_lengths), 1)
        if sentence_lengths
        else 0.0,
        "avg_word_length": round(sum(word_lengths) / len(word_lengths), 1) if word_lengths else 0.0,
        "longest_sentence": sentences[longest_idx] if longest_idx is not None else "",
        "reading_minutes": round(len(tokens) / 225, 1),
        "speaking_minutes": round(len(tokens) / 130, 1),
        "readability_score": score,
        "reading_level": reading_level(score),
        "keywords": keyword_scores(tokens),
        "bigrams": phrase_scores(tokens, 2),
        "trigrams": phrase_scores(tokens, 3, top_n=10),
        "sentence_lengths": sentence_lengths,
        "words_per_page": [len(tokenize(p)) for p in pages],
        "sentences": sentences,
    }
