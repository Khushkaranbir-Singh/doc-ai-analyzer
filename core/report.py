"""Build the downloadable summary PDF."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#16222E")
ACCENT = colors.HexColor("#2F4BFF")
MUTED = colors.HexColor("#5C6B7A")
RULE = colors.HexColor("#D8DEE6")


def _styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="Times-Bold", fontSize=26,
            leading=30, textColor=INK, alignment=0, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=10, leading=14,
            textColor=MUTED, spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
            leading=17, textColor=INK, spaceBefore=16, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=10.5, leading=16,
            textColor=INK, alignment=TA_JUSTIFY, spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontSize=10.5, leading=15, textColor=INK,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=8.5, leading=12, textColor=MUTED,
        ),
    }


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _keyword_chart(keywords: List) -> io.BytesIO | None:
    if not keywords:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = list(reversed(keywords[:10]))
        labels = [k for k, _ in top]
        values = [v for _, v in top]

        fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=200)
        ax.barh(labels, values, color="#2F4BFF", height=0.62)
        ax.set_xlabel("Occurrences", fontsize=8, color="#5C6B7A")
        ax.tick_params(labelsize=8, colors="#16222E", length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#D8DEE6")
        ax.grid(axis="x", color="#EEF1F5", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", transparent=False)
        plt.close(fig)
        buffer.seek(0)
        return buffer
    except Exception:
        return None


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.setStrokeColor(RULE)
    canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
    canvas.drawString(20 * mm, 10 * mm, "Doc AI Analyzer")
    canvas.drawRightString(A4[0] - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_summary_pdf(
    doc_name: str,
    summary: Dict[str, object],
    stats: Dict[str, object],
    file_meta: Dict[str, object] | None = None,
    qa_history: List[Dict[str, str]] | None = None,
) -> bytes:
    """Return the report as PDF bytes, ready for a download button."""
    style = _styles()
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
        title=f"Summary of {doc_name}",
        author="Doc AI Analyzer",
    )

    story: List = [
        Paragraph("Document summary", style["title"]),
        Paragraph(
            f"{_escape(doc_name)} &nbsp;·&nbsp; generated "
            f"{datetime.now().strftime('%d %B %Y at %H:%M')} &nbsp;·&nbsp; "
            f"summary engine: {_escape(summary.get('engine', 'Local ranking'))}",
            style["subtitle"],
        ),
    ]

    rows = [
        ["Words", f"{stats.get('word_count', 0):,}", "Sentences", f"{stats.get('sentence_count', 0):,}"],
        ["Unique words", f"{stats.get('unique_words', 0):,}", "Paragraphs", f"{stats.get('paragraph_count', 0):,}"],
        ["Characters", f"{stats.get('character_count', 0):,}", "Pages", f"{stats.get('page_count', 0):,}"],
        ["Reading time", f"{stats.get('reading_minutes', 0)} min", "Readability", str(stats.get("reading_level", "—"))],
    ]
    table = Table(rows, colWidths=[32 * mm, 38 * mm, 32 * mm, 68 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
                ("LINEBEFORE", (0, 0), (0, -1), 2, ACCENT),
                ("LEFTPADDING", (0, 0), (0, -1), 8),
            ]
        )
    )
    story += [table, Paragraph("Overview", style["h2"])]

    overview = str(summary.get("overview") or "No summary could be produced for this file.")
    for block in overview.split("\n\n"):
        if block.strip():
            story.append(Paragraph(_escape(block.strip()), style["body"]))

    bullets = summary.get("bullets") or []
    if bullets:
        story.append(Paragraph("Key points", style["h2"]))
        story.append(
            ListFlowable(
                [ListItem(Paragraph(_escape(b), style["bullet"]), leftIndent=12) for b in bullets],
                bulletType="bullet",
                bulletColor=ACCENT,
                start="•",
                leftIndent=14,
            )
        )

    keywords = stats.get("keywords") or []
    chart = _keyword_chart(keywords)
    if chart is not None:
        story.append(Paragraph("Most frequent terms", style["h2"]))
        story.append(Image(chart, width=165 * mm, height=82 * mm))
    elif keywords:
        story.append(Paragraph("Most frequent terms", style["h2"]))
        story.append(
            Paragraph(
                _escape(", ".join(f"{word} ({count})" for word, count in keywords[:15])),
                style["body"],
            )
        )

    if file_meta:
        story.append(Paragraph("File details", style["h2"]))
        meta_rows = [[_escape(k), _escape(v)] for k, v in list(file_meta.items())[:12]]
        meta_table = Table(meta_rows, colWidths=[45 * mm, 125 * mm])
        meta_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
                    ("TEXTCOLOR", (1, 0), (1, -1), INK),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, RULE),
                ]
            )
        )
        story.append(meta_table)

    if qa_history:
        story += [PageBreak(), Paragraph("Questions and answers", style["h2"])]
        for item in qa_history:
            story.append(Paragraph(f"<b>{_escape(item['question'])}</b>", style["body"]))
            story.append(Paragraph(_escape(item["answer"]), style["body"]))
            story.append(Spacer(1, 4))

    pdf.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
