"""End-to-end check of the analysis engine. Run: python tests_smoke.py"""

import sys

from core.analyze import analyze_text
from core.extract import extract_document
from core.qa import DocumentQA
from core.report import build_summary_pdf
from core.summarize import build_summary

SAMPLE = """
Doc AI Analyzer reads documents and explains what is inside them. The application
accepts PDF, Word, plain text, image and executable files up to one hundred megabytes.

Text is extracted first. A PDF is read page by page so that page level statistics stay
accurate. Word documents are read paragraph by paragraph, including the contents of tables.
Images are passed through optical character recognition. Executables are inspected rather
than run, and only their readable strings are reported.

The summary engine ranks every sentence against every other sentence and keeps the ones
that sit at the centre of the document. When an Anthropic API key is configured, Claude
writes the summary instead, and the ranking engine becomes the fallback path.

Questions are answered from the document itself. The text is split into overlapping
passages, each passage is scored against the question, and the strongest passages are
returned alongside the answer so that any claim can be checked against the source.

The report builder produces a PDF that carries the overview, the key points, the document
statistics and a chart of the most frequent terms.
""".strip()


def main() -> int:
    failures = []

    doc = extract_document(SAMPLE.encode("utf-8"), "sample.txt")
    if not doc.has_text:
        failures.append("extraction produced no text")

    stats = analyze_text(doc.text, doc.pages)
    print(f"words={stats['word_count']} sentences={stats['sentence_count']} "
          f"readability={stats['readability_score']} ({stats['reading_level']})")
    if stats["word_count"] < 100:
        failures.append("word count is implausibly low")
    if not stats["keywords"]:
        failures.append("no keywords found")

    summary = build_summary(doc.text, length="Short", use_ai=False)
    print(f"summary engine={summary['engine']} bullets={len(summary['bullets'])}")
    if len(summary["overview"].split()) < 15:
        failures.append("summary is too short")

    qa = DocumentQA(doc.text)
    result = qa.answer("Which file types are accepted?", use_ai=False)
    print(f"answer -> {result['answer'][:110]}…")
    if not result["answer"]:
        failures.append("question answering returned nothing")

    pdf = build_summary_pdf("sample.txt", summary, stats, {"File name": "sample.txt"},
                            [{"question": "Which file types are accepted?",
                              "answer": result["answer"]}])
    print(f"pdf bytes={len(pdf)}")
    if not pdf.startswith(b"%PDF") or len(pdf) < 2000:
        failures.append("PDF report was not generated correctly")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
