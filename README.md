# RunGopher Conversation Evaluator

Automated evaluation pipeline for RunGopher voice agent conversations. Upload a CSV export, and the app strips PII, summarizes every call, clusters them into patterns, and generates a branded HTML report — all from a browser.

---

## How it works

| Phase | What it does | Where it runs |
|-------|-------------|---------------|
| 1. PII Stripping | Removes names, phones, addresses from transcripts | Local (no API) |
| 2. Summarization | One-sentence summary + outcome per conversation | OpenAI GPT-4o |
| 3. Clustering | Groups conversations into patterns and trends | OpenAI GPT-4o |
| 4. Report | Generates a branded, downloadable HTML report | OpenAI GPT-4o |

After the pipeline completes, a **chat interface** lets you ask questions about the results directly.

---

## Setup

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/alipinch93/rungopher-conversation-evaluator.git
cd rungopher-conversation-evaluator

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Configure environment
cp .env.example .env
# Edit .env and fill in your values (see below)

# 5. Run
python app.py
```

Then open **http://localhost:8000** in your browser.

---

## Configuration

Copy `.env.example` to `.env` and set the following:

```env
OPENAI_API_KEY=sk-...          # Your OpenAI API key
APP_PASSWORD=yourpassword      # Password to access the app
OPENAI_MODEL=gpt-4o            # Model (gpt-4o or gpt-4o-mini)
BATCH_SIZE=300                 # Conversations per clustering batch
PII_THRESHOLD=0.4              # Presidio confidence threshold (0–1)
```

> **Never commit `.env`** — it's in `.gitignore` by default.

---

## CSV format

The app expects a RunGopher CSV export with at least a `Transcript` column. Optional columns used if present: `Call ID`, `Direction`, `User Intent`, `Duration (Seconds)`, `Assistant Phone`, `Customer Phone`.

---

## Deployment

The app runs on any host that supports Python. Recommended options:

- **[Render](https://render.com)** — connect your GitHub repo, set env vars in the dashboard, deploy
- **[Railway](https://railway.app)** — same workflow, free tier available

Set your `OPENAI_API_KEY`, `APP_PASSWORD`, and `OPENAI_MODEL` as environment variables on the host — no `.env` file needed in production.
