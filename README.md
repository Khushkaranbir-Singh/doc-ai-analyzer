# Doc AI Analyzer

Upload a document and understand it in one place: counts and readability, a summary you can
question, charts of the language inside, and a PDF report you can take away.

Built with Python and Streamlit. It runs with no API key at all — summaries and answers come
from a local ranking and retrieval engine. Add an Anthropic key and Claude writes them instead.

---

## Features

| Feature | What it does |
|---|---|
| Uploads | PDF, DOC, DOCX, TXT, MD, JPG, JPEG, PNG, EXE — up to **100 MB** per file |
| Word count | Words, unique words, characters, sentences, paragraphs, pages, reading and speaking time |
| Summary | Overview plus key-point bullets, at three lengths (short, medium, detailed) |
| Q&A | Ask the file anything; every answer shows the passages it was drawn from |
| Visuals | One button renders frequency bars, a share donut, repeated phrases, sentence-length histogram, words-per-page area chart, a readability gauge, sentence rhythm, a word cloud and a downloadable term table |
| PDF export | A formatted report: summary, key points, statistics table, term chart, file details, and optionally your Q&A history |
| OCR | Text is read out of images with Tesseract |
| Executables | Inspected, never run — file identity, entropy and readable strings only |

---

## Run it locally

```bash
git clone https://github.com/<your-username>/doc-ai-analyzer.git
cd doc-ai-analyzer

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

For OCR on images, install Tesseract on your machine:

- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: [UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki)

Everything else works without it.

### Optional: connect Claude

Create `.streamlit/secrets.toml` (already git-ignored):

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_MODEL = "claude-sonnet-5"
```

The sidebar shows whether Claude is connected. Without a key the app stays fully usable.

---

## Put it on GitHub

```bash
cd doc-ai-analyzer
git init
git add .
git commit -m "Doc AI Analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/doc-ai-analyzer.git
git push -u origin main
```

Create the empty repository on github.com first, without a README, so the push is clean.
`.gitignore` already keeps `secrets.toml` and `.env` out of the repository — never commit a key.

---

## Deploy it live

### Streamlit Community Cloud — free, five minutes

1. Push the repository to GitHub as above.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app → Deploy a public app from GitHub**.
4. Repository `<your-username>/doc-ai-analyzer`, branch `main`, main file `app.py`.
5. Open **Advanced settings → Secrets** and paste your key if you have one:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ANTHROPIC_MODEL = "claude-sonnet-5"
   ```
6. Deploy. Your live URL is `https://<app-name>.streamlit.app`.

`packages.txt` installs Tesseract on the server, and `.streamlit/config.toml` sets the
100 MB upload limit. Both are read automatically.

### Render — free tier, Docker

1. Push to GitHub.
2. On [render.com](https://render.com): **New → Web Service → Build and deploy from a Git repository**.
3. Pick the repository. Render reads `render.yaml`, so runtime and health check are already set.
4. Add `ANTHROPIC_API_KEY` under **Environment** if you want Claude.
5. Deploy. Your live URL is `https://doc-ai-analyzer.onrender.com`.

### Hugging Face Spaces

Create a Space, choose the **Streamlit** SDK, push this repository into it, and add the key
under **Settings → Variables and secrets**.

### Docker, anywhere

```bash
docker build -t doc-ai-analyzer .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... doc-ai-analyzer
```

---

## Project layout

```
doc-ai-analyzer/
├── app.py                  Streamlit interface: upload, tabs, charts, downloads
├── core/
│   ├── extract.py          Readers for PDF, Word, text, images (OCR), executables
│   ├── analyze.py          Counts, keywords, phrases, Flesch readability
│   ├── summarize.py        TextRank ranking + Claude abstractive summary
│   ├── qa.py               Chunking, TF-IDF retrieval with stemming, answers
│   ├── report.py           ReportLab PDF report
│   └── llm.py              Optional Anthropic client
├── tests_smoke.py          End-to-end check of the engine
├── requirements.txt        Python dependencies
├── packages.txt            System packages for Streamlit Cloud (Tesseract)
├── Dockerfile / render.yaml / Procfile
└── .streamlit/config.toml  100 MB upload limit and theme
```

## How it works

Text extraction runs per format — page by page for PDFs, paragraphs and tables for Word,
OCR for images, printable strings for executables.

Summarising builds a similarity graph over the document's sentences and runs PageRank across
it, so the sentences that sit at the centre of the document rise to the top. With a Claude key
the same text goes to the API for an abstractive summary, and the local engine becomes the
fallback if the call fails.

Questions are answered by splitting the text into overlapping passages, scoring them against
the question with stemmed TF-IDF, and returning both the answer and its source passages.

## Verify the engine

```bash
python tests_smoke.py
```

It extracts, analyses, summarises, answers a question and builds a PDF, then reports pass or fail.
The same check runs on every push through GitHub Actions.

## Notes

- Files are held in memory for the session only. Nothing is written to disk or stored.
- Legacy `.doc` extraction is approximate; save as `.docx` or PDF for a clean read.
- Scanned PDFs have no selectable text — export them with OCR, or upload the pages as images.

MIT licensed.
