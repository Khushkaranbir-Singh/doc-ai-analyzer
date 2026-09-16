"""Ask questions about a document.

Retrieval is local and always runs: the document is chunked, scored against the
question with TF-IDF, and the best passages are returned as evidence. If a
Claude key is configured, those passages are turned into a written answer.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .analyze import STOPWORDS, split_sentences, tokenize
from . import llm

SUFFIXES = ("ations", "ation", "ingly", "ments", "ment", "ness", "ing", "ies", "ied",
            "ers", "er", "ed", "es", "ly", "s")


def stem(word: str) -> str:
    """Strip common endings so 'accepted' matches 'accepts' and 'files' matches 'file'."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            root = word[: -len(suffix)]
            if root.endswith("i"):
                root = root[:-1] + "y"
            return root
    return word


def analyzer(text: str) -> List[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]


class DocumentQA:
    def __init__(self, text: str, chunk_words: int = 180, overlap: int = 40):
        self.text = text
        self.chunks = self._chunk(text, chunk_words, overlap)
        self._vectorizer = None
        self._matrix = None
        self._build_index()

    @staticmethod
    def _chunk(text: str, size: int, overlap: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        step = max(size - overlap, 40)
        return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]

    def _build_index(self) -> None:
        if not self.chunks:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(analyzer=analyzer, sublinear_tf=True)
            self._matrix = self._vectorizer.fit_transform(self.chunks)
        except Exception:
            self._vectorizer = None

    def retrieve(self, question: str, k: int = 4) -> List[Tuple[str, float]]:
        if not self.chunks:
            return []
        if self._vectorizer is not None:
            try:
                vector = self._vectorizer.transform([question])
                scores = (self._matrix @ vector.T).toarray().ravel()
                order = scores.argsort()[::-1][:k]
                hits = [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]
                if hits:
                    return hits
            except Exception:
                pass

        keyword_hits = self._keyword_retrieve(question, k)
        if keyword_hits:
            return keyword_hits
        # Broad questions ("what is this about?") share no rare terms with the
        # text, so fall back to the opening passages rather than refusing.
        return [(chunk, 0.0) for chunk in self.chunks[:k]]

    def _keyword_retrieve(self, question: str, k: int) -> List[Tuple[str, float]]:
        terms = set(analyzer(question))
        scored = []
        for chunk in self.chunks:
            words = set(analyzer(chunk))
            hits = len(terms & words)
            if hits:
                scored.append((chunk, hits / max(len(terms), 1)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def answer(self, question: str, use_ai: bool = True) -> Dict[str, object]:
        passages = self.retrieve(question)
        if not passages:
            return {
                "answer": "Nothing in this document matches that question. Try different wording, "
                          "or ask about a term that appears in the text.",
                "sources": [],
                "engine": "Local retrieval",
            }

        context = "\n\n---\n\n".join(p for p, _ in passages)

        if use_ai and llm.ai_available():
            try:
                prompt = (
                    "Answer the question using only the document extracts below. "
                    "If the extracts do not contain the answer, say so plainly.\n\n"
                    f"QUESTION: {question}\n\nEXTRACTS:\n{context}"
                )
                text = llm.complete(
                    prompt,
                    system="You answer questions strictly from the supplied document extracts.",
                    max_tokens=900,
                )
                return {"answer": text, "sources": [p for p, _ in passages], "engine": "AI Engine"}
            except Exception as exc:
                result = self._local_answer(question, passages)
                result["engine"] = f"Local retrieval (AI call failed: {exc})"
                return result

        return self._local_answer(question, passages)

    def _local_answer(self, question: str, passages: List[Tuple[str, float]]) -> Dict[str, object]:
        """Pick the sentences inside the best passages that overlap the question."""
        terms = set(analyzer(question))
        ranked = []
        for passage, score in passages:
            for sentence in split_sentences(passage) or [passage]:
                words = set(analyzer(sentence))
                overlap = len(terms & words)
                if overlap:
                    ranked.append((overlap + score, sentence.strip()))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        picked, seen = [], set()
        for _, sentence in ranked:
            key = re.sub(r"\W+", "", sentence.lower())[:80]
            if key not in seen:
                seen.add(key)
                picked.append(sentence)
            if len(picked) == 3:
                break

        body = " ".join(picked) if picked else passages[0][0][:600]
        return {
            "answer": body,
            "sources": [p for p, _ in passages],
            "engine": "Local retrieval",
        }