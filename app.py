"""Doc AI Analyzer — upload a file, understand it, take the summary with you.

Run locally:   streamlit run app.py
Live deploy:   see README.md (Streamlit Community Cloud, Render, or Docker)
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import llm
from core.analyze import analyze_text
from core.extract import MAX_FILE_MB, SUPPORTED_EXTENSIONS, extract_document
from core.qa import DocumentQA
from core.report import build_summary_pdf
from core.summarize import build_summary

PALETTE = {
    "paper": "#EEF2F6",
    "surface": "#FFFFFF",
    "ink": "#16222E",
    "muted": "#040404",
    "rule": "#D8DEE6",
    "accent": "#2F4BFF",
    "teal": "#0F8B8D",
    "amber": "#E08D1B",
}
CHART_SEQUENCE = ["#2F4BFF", "#0F8B8D", "#E08D1B", "#6C4BFF", "#1B9AAA", "#C2410C", "#3F6C9E"]

st.set_page_config(
    page_title="Doc AI Analyzer",
    page_icon="◨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');

        .stApp {{ background: {PALETTE['paper']}; }}
        html, body, [class*="css"], .stMarkdown, .stTextInput input, .stSelectbox div {{
            font-family: 'Inter', system-ui, sans-serif;
            color: {PALETTE['ink']};
        }}

        .masthead {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['rule']};
            border-left: 5px solid {PALETTE['accent']};
            border-radius: 4px;
            padding: 26px 30px 22px 30px;
            margin-bottom: 18px;
        }}
        .masthead h1 {{
            font-family: 'Instrument Serif', Georgia, serif;
            font-size: 44px; line-height: 1.05; margin: 0 0 6px 0;
            letter-spacing: -0.3px; font-weight: 400;
        }}
        .masthead p {{ margin: 0; color: {PALETTE['muted']}; font-size: 15px; max-width: 64ch; }}

        .panel {{
            background: {PALETTE['surface']};
            border: 1px solid {PALETTE['rule']};
            border-radius: 4px;
            padding: 20px 24px;
        }}
        .panel h3 {{
            font-family: 'Instrument Serif', Georgia, serif;
            font-weight: 400; font-size: 24px; margin: 0 0 10px 0;
        }}
        .panel p {{ font-size: 15px; line-height: 1.65; margin: 0 0 10px 0; }}

        .stat-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 4px 0 18px 0; }}
        .stat {{
            flex: 1 1 150px; background: {PALETTE['surface']};
            border: 1px solid {PALETTE['rule']}; border-top: 3px solid {PALETTE['accent']};
            border-radius: 3px; padding: 14px 16px;
        }}
        .stat .value {{ font-size: 27px; font-weight: 600; letter-spacing: -0.5px; }}
        .stat .label {{ font-size: 12.5px; color: {PALETTE['muted']}; margin-top: 2px; }}
        .stat.teal {{ border-top-color: {PALETTE['teal']}; }}
        .stat.amber {{ border-top-color: {PALETTE['amber']}; }}

        .keyline {{
            display: inline-block; border: 1px solid {PALETTE['rule']};
            border-radius: 999px; padding: 4px 12px; margin: 0 6px 8px 0;
            font-size: 13px; background: {PALETTE['surface']};
        }}
        .answer {{
            background: {PALETTE['surface']}; border: 1px solid {PALETTE['rule']};
            border-left: 4px solid {PALETTE['teal']}; border-radius: 3px;
            padding: 16px 20px; margin-bottom: 12px; font-size: 15px; line-height: 1.65;
        }}
        .answer .q {{ font-weight: 600; margin-bottom: 8px; }}
        .meta {{ font-size: 12.5px; color: {PALETTE['muted']}; }}

        .stButton > button {{
            background: {PALETTE['accent']}; color: #fff; border: 0;
            border-radius: 3px; padding: 10px 20px; font-weight: 500;
        }}
        .stButton > button:hover {{ background: #1F36D8; color: #fff; }}
        .stDownloadButton > button {{
            background: {PALETTE['ink']}; color: #fff; border: 0;
            border-radius: 3px; padding: 10px 20px; font-weight: 500;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 26px; border-bottom: 1px solid {PALETTE['rule']}; }}
        .stTabs [data-baseweb="tab"] {{ padding: 8px 0; font-size: 15px; }}
        [data-testid="stExpander"] {{
            background: {PALETTE['surface']}; border: 1px solid {PALETTE['rule']};
            border-radius: 4px; margin-bottom: 16px;
        }}
        [data-testid="stExpander"] summary {{
            font-family: 'Instrument Serif', Georgia, serif; font-size: 20px;
            padding: 14px 20px;
        }}
        [data-testid="stSidebar"] {{ background: {PALETTE['surface']}; border-right: 1px solid {PALETTE['rule']}; }}
        footer, #MainMenu {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_cards(items) -> None:
    cards = "".join(
        f"<div class='stat {tone}'><div class='value'>{value}</div>"
        f"<div class='label'>{label}</div></div>"
        for label, value, tone in items
    )
    st.markdown(f"<div class='stat-grid'>{cards}</div>", unsafe_allow_html=True)


def styled_figure(fig, height: int = 380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=PALETTE["ink"], size=13),
        title_font=dict(size=15),
        colorway=CHART_SEQUENCE,
        legend=dict(orientation="h", y=-0.18),
    )
    fig.update_xaxes(gridcolor="#050505", zeroline=False)
    fig.update_yaxes(gridcolor="#050505", zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

def init_state() -> None:
    defaults = {
        "doc": None,
        "stats": None,
        "summary": None,
        "qa_engine": None,
        "qa_history": [],
        "show_visuals": False,
        "file_key": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def load_file(upload) -> None:
    data = upload.getvalue()
    key = f"{upload.name}:{len(data)}"
    if st.session_state.file_key == key:
        return

    if len(data) > MAX_FILE_MB * 1024 * 1024:
        st.error(f"That file is larger than the {MAX_FILE_MB} MB limit. Upload a smaller file.")
        return

    with st.spinner("Reading the file…"):
        doc = extract_document(data, upload.name)
        stats = analyze_text(doc.text, doc.pages)
        qa = DocumentQA(doc.text) if doc.has_text else None

    st.session_state.update(
        doc=doc,
        stats=stats,
        qa_engine=qa,
        summary=None,
        qa_history=[],
        show_visuals=False,
        file_key=key,
    )


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def tab_overview(doc, stats) -> None:
    stat_cards(
        [
            ("Words", f"{stats['word_count']:,}", ""),
            ("Unique words", f"{stats['unique_words']:,}", ""),
            ("Sentences", f"{stats['sentence_count']:,}", "teal"),
            ("Paragraphs", f"{stats['paragraph_count']:,}", "teal"),
            ("Pages", f"{doc.page_count or stats['page_count']:,}", "amber"),
            ("Reading time", f"{stats['reading_minutes']} min", "amber"),
        ]
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("<div class='panel'><h3>What is in this file</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p><b>{doc.name}</b> is a {doc.extension.upper()} file of "
            f"{doc.size_mb:.2f} MB. It contains {stats['word_count']:,} words across "
            f"{stats['sentence_count']:,} sentences, at {stats['avg_sentence_length']} words "
            f"per sentence on average. Reading it aloud takes about "
            f"{stats['speaking_minutes']} minutes.</p>"
            f"<p class='meta'>Readability {stats['readability_score']} — "
            f"{stats['reading_level']}. Lexical diversity "
            f"{stats['lexical_diversity']} (unique words ÷ total words).</p></div>",
            unsafe_allow_html=True,
        )
        if stats["keywords"]:
            st.markdown("##### Terms that carry this document")
            chips = "".join(
                f"<span class='keyline'>{word} <b>{count}</b></span>"
                for word, count in stats["keywords"][:18]
            )
            st.markdown(chips, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='panel'><h3>File details</h3></div>", unsafe_allow_html=True)
        details = {
            "Type": doc.extension.upper(),
            "Size": f"{doc.size_mb:.2f} MB",
            "Pages / blocks": doc.page_count,
            "Characters": f"{stats['character_count']:,}",
            "SHA-256": doc.sha256[:24] + "…",
        }
        details.update({k: v for k, v in list(doc.meta.items())[:8]})
        st.dataframe(
            pd.DataFrame({"Field": list(details.keys()), "Value": [str(v) for v in details.values()]}),
            hide_index=True,
            use_container_width=True,
        )

    for note in doc.notes:
        st.info(note)

    with st.expander("Read the extracted text"):
        st.text_area("Extracted text", doc.text[:60_000], height=320, label_visibility="collapsed")


def tab_summary(doc, stats, length, use_ai) -> None:
    st.markdown("### Summary")
    st.caption(
        "The AI engine writes the summary when an API key is configured. Without one, the app ranks "
        "the document's own sentences and returns the most central ones."
    )

    if st.button("Summarise this document", type="primary"):
        with st.spinner("Working through the document…"):
            st.session_state.summary = build_summary(doc.text, length=length, use_ai=use_ai)

    summary = st.session_state.summary
    if not summary:
        st.markdown(
            "<div class='panel'><p class='meta'>No summary yet. Press the button above and "
            "the overview plus key points will appear here.</p></div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"<div class='panel'><h3>Overview</h3><p>{summary['overview']}</p>"
        f"<p class='meta'>Engine: {summary['engine']} · length: {length}</p></div>",
        unsafe_allow_html=True,
    )

    if summary["bullets"]:
        st.markdown("#### Key points")
        for bullet in summary["bullets"]:
            st.markdown(f"- {bullet}")

    st.download_button(
        "Download summary as text",
        data=(summary["overview"] + "\n\n" + "\n".join(f"- {b}" for b in summary["bullets"])),
        file_name=f"summary-{doc.name.rsplit('.', 1)[0]}.txt",
        mime="text/plain",
    )


def tab_qa(doc, use_ai) -> None:
    st.markdown("### Ask this document a question")
    engine = st.session_state.qa_engine
    if engine is None:
        st.warning("No readable text was found in this file, so questions cannot be answered.")
        return

    st.caption(
        "Questions are answered from the file's own passages. Every answer shows the extracts "
        "it came from, so you can check it."
    )

    suggestions = [
        "What is this document about?",
        "What are the main conclusions?",
        "What dates or numbers appear?",
        "Who is responsible for what?",
    ]
    columns = st.columns(len(suggestions))
    preset = None
    for column, suggestion in zip(columns, suggestions):
        if column.button(suggestion, key=f"suggest-{suggestion}"):
            preset = suggestion

    question = st.text_input(
        "Your question",
        value=preset or "",
        placeholder="e.g. What does section 3 require?",
    )
    ask = st.button("Ask", type="primary")

    if (ask or preset) and question.strip():
        with st.spinner("Searching the document…"):
            result = engine.answer(question.strip(), use_ai=use_ai)
        st.session_state.qa_history.insert(
            0,
            {
                "question": question.strip(),
                "answer": result["answer"],
                "sources": result["sources"],
                "engine": result["engine"],
                "time": datetime.now().strftime("%H:%M"),
            },
        )

    for item in st.session_state.qa_history:
        st.markdown(
            f"<div class='answer'><div class='q'>{item['question']}</div>{item['answer']}"
            f"<div class='meta' style='margin-top:10px'>{item['engine']} · {item['time']}</div></div>",
            unsafe_allow_html=True,
        )
        if item["sources"]:
            with st.expander("Show the passages this came from"):
                for index, source in enumerate(item["sources"], start=1):
                    st.markdown(f"**Extract {index}**")
                    st.write(source)

    if st.session_state.qa_history and st.button("Clear questions"):
        st.session_state.qa_history = []
        st.rerun()


def tab_visuals(doc, stats) -> None:
    st.markdown("### Visual breakdown")
    st.caption("Charts are built on request so large files stay fast.")

    if st.button("Show the charts", type="primary"):
        st.session_state.show_visuals = True

    if not st.session_state.show_visuals:
        st.markdown(
            "<div class='panel'><p class='meta'>Press the button to chart word frequency, "
            "phrase patterns, sentence rhythm, page density and readability.</p></div>",
            unsafe_allow_html=True,
        )
        return

    keywords = stats["keywords"]
    if not keywords:
        st.warning("There is not enough text in this file to chart.")
        return

    left, right = st.columns(2, gap="large")

    with left:
        frame = pd.DataFrame(keywords[:12], columns=["Term", "Count"]).sort_values("Count")
        fig = px.bar(frame, x="Count", y="Term", orientation="h", title="Most frequent terms")
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    with right:
        share = pd.DataFrame(keywords[:8], columns=["Term", "Count"])
        fig = px.pie(share, names="Term", values="Count", hole=0.55,
                     title="Share of the top eight terms")
        fig.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(styled_figure(fig), use_container_width=True)

    if stats["bigrams"]:
        frame = pd.DataFrame(stats["bigrams"][:12], columns=["Phrase", "Count"]).sort_values("Count")
        fig = px.bar(frame, x="Count", y="Phrase", orientation="h",
                     title="Repeated two-word phrases")
        st.plotly_chart(styled_figure(fig, height=420), use_container_width=True)

    left, right = st.columns(2, gap="large")

    with left:
        lengths = stats["sentence_lengths"]
        if lengths:
            fig = px.histogram(pd.DataFrame({"Words per sentence": lengths}),
                               x="Words per sentence", nbins=24,
                               title="Sentence length distribution")
            st.plotly_chart(styled_figure(fig), use_container_width=True)

    with right:
        per_page = stats["words_per_page"]
        if per_page:
            frame = pd.DataFrame(
                {"Page": list(range(1, len(per_page) + 1)), "Words": per_page}
            )
            fig = px.area(frame, x="Page", y="Words", title="Words per page")
            st.plotly_chart(styled_figure(fig), use_container_width=True)

    left, right = st.columns(2, gap="large")

    with left:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=max(min(stats["readability_score"], 100), 0),
                title={"text": f"Readability — {stats['reading_level']}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": PALETTE["accent"]},
                    "steps": [
                        {"range": [0, 30], "color": "#E3E8EF"},
                        {"range": [30, 60], "color": "#EDF1F6"},
                        {"range": [60, 100], "color": "#F5F8FB"},
                    ],
                },
            )
        )
        st.plotly_chart(styled_figure(gauge, height=330), use_container_width=True)

    with right:
        rhythm = stats["sentence_lengths"][:120]
        if rhythm:
            frame = pd.DataFrame(
                {"Sentence": list(range(1, len(rhythm) + 1)), "Words": rhythm}
            )
            fig = px.line(frame, x="Sentence", y="Words",
                          title="Sentence rhythm through the document")
            st.plotly_chart(styled_figure(fig, height=330), use_container_width=True)

    st.markdown("#### Word cloud")
    try:
        from wordcloud import WordCloud

        cloud = WordCloud(
            width=1400,
            height=520,
            background_color="white",
            colormap="cividis",
            collocations=False,
        ).generate_from_frequencies(dict(stats["keywords"][:120]))
        st.image(cloud.to_array(), use_container_width=True)
    except Exception:
        st.caption("The word cloud library is not installed, so the charts above cover the same ground.")

    frame = pd.DataFrame(stats["keywords"], columns=["Term", "Occurrences"])
    st.markdown("#### Full term table")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button(
        "Download term counts as CSV",
        data=frame.to_csv(index=False).encode(),
        file_name=f"terms-{doc.name.rsplit('.', 1)[0]}.csv",
        mime="text/csv",
    )


def tab_export(doc, stats, length, use_ai) -> None:
    st.markdown("### Export a summary PDF")
    st.caption("The report carries the overview, key points, document statistics and a term chart.")

    include_qa = st.checkbox(
        "Include my questions and answers", value=bool(st.session_state.qa_history)
    )

    if st.button("Build the PDF", type="primary"):
        summary = st.session_state.summary
        with st.spinner("Composing the report…"):
            if not summary:
                summary = build_summary(doc.text, length=length, use_ai=use_ai)
                st.session_state.summary = summary
            meta = {
                "File name": doc.name,
                "File type": doc.extension.upper(),
                "File size": f"{doc.size_mb:.2f} MB",
                "SHA-256": doc.sha256,
            }
            meta.update({k: str(v) for k, v in list(doc.meta.items())[:6]})
            st.session_state.pdf_bytes = build_summary_pdf(
                doc_name=doc.name,
                summary=summary,
                stats=stats,
                file_meta=meta,
                qa_history=st.session_state.qa_history if include_qa else None,
            )

    if st.session_state.get("pdf_bytes"):
        st.success("The report is ready.")
        st.download_button(
            "Download summary PDF",
            data=st.session_state.pdf_bytes,
            file_name=f"doc-ai-summary-{doc.name.rsplit('.', 1)[0]}.pdf",
            mime="application/pdf",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    inject_css()
    init_state()

    st.markdown(
        "<div class='masthead'><h1>Doc AI Analyzer</h1>"
        "<p>Upload a PDF, Word file, image or executable. Get a summary you can question, "
        "charts of what is inside, and a report you can take away.</p>"
        "<p class='meta' style='margin-top:10px'>Developed by Khushkaranbir Singh</p></div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Settings")
        length = st.select_slider("Summary length", options=["Short", "Medium", "Detailed"],
                                  value="Medium")
        ai_ready = llm.ai_available()
        use_ai = st.toggle("Use AI-enhanced engine", value=ai_ready, disabled=not ai_ready)
        if ai_ready:
            st.success("AI engine connected")
        else:
            st.info(
                "Running on the built-in engine. Add an API key in secrets for "
                "AI-enhanced summaries and answers."
            )

        st.markdown("---")
        st.markdown("### Accepted files")
        st.markdown(
            "PDF · DOC · DOCX · TXT · MD · JPG · JPEG · PNG · EXE  \n"
            f"Up to **{MAX_FILE_MB} MB** per file."
        )
        st.caption("Files are processed in memory for this session only. Executables are never run.")
        st.markdown("---")
        st.caption("Developed by Khushkaranbir Singh")

        if st.session_state.doc and st.button("Clear this file"):
            for key in ("doc", "stats", "summary", "qa_engine", "file_key", "pdf_bytes"):
                st.session_state[key] = None
            st.session_state.qa_history = []
            st.session_state.show_visuals = False
            st.rerun()

    upload = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_EXTENSIONS,
        help=f"Maximum {MAX_FILE_MB} MB.",
    )
    if upload is not None:
        load_file(upload)

    doc = st.session_state.doc
    if doc is None:
        st.markdown(
            "<div class='panel'><h3>Start with a file</h3>"
            "<p>Everything runs from one upload: word counts and readability, a summary, "
            "a question box answered from the text itself, charts of the language inside, "
            "and a PDF report.</p>"
            "<p class='meta'>No account, no storage — the file lives in this session and "
            "disappears when you close the tab.</p></div>",
            unsafe_allow_html=True,
        )
        return

    stats = st.session_state.stats
    if doc.kind == "image" and not doc.has_text:
        st.warning("This image has no readable text, so the text tools will be mostly empty.")
    if doc.kind == "binary":
        st.warning("This is an executable. You are seeing its identity and readable strings only.")

    section_titles = {
        "overview": "① Overview",
        "summary": "② Summary",
        "questions": "③ Ask questions",
        "visuals": "④ Visuals",
        "export": "⑤ Export PDF",
    }

    with st.expander(section_titles["overview"], expanded=True):
        tab_overview(doc, stats)

    with st.expander(section_titles["summary"], expanded=True):
        tab_summary(doc, stats, length, use_ai)

    with st.expander(section_titles["questions"], expanded=True):
        tab_qa(doc, use_ai)

    with st.expander(section_titles["visuals"], expanded=True):
        tab_visuals(doc, stats)

    with st.expander(section_titles["export"], expanded=True):
        tab_export(doc, stats, length, use_ai)


if __name__ == "__main__":
    main()