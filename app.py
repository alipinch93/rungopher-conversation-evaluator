#!/usr/bin/env python3
"""
RunGopher Conversation Evaluator — Web Server
FastAPI backend that serves the UI and runs the pipeline via API endpoints.

Usage:
    python app.py
    # Then open http://localhost:8000 in your browser

Team members just need the URL — no CLI or Python knowledge required.
"""

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ─── Auth ──────────────────────────────────────────────────────────────
_APP_PASSWORD = os.getenv("APP_PASSWORD", "")
# Deterministic token derived from password — survives server restarts
_VALID_TOKEN = hashlib.sha256(f"rungopher:{_APP_PASSWORD}".encode()).hexdigest() if _APP_PASSWORD else None

_bearer = HTTPBearer(auto_error=False)

def require_auth(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not _VALID_TOKEN:
        return  # No password set — open access (local dev)
    if not credentials or credentials.credentials != _VALID_TOKEN:
        raise HTTPException(401, "Unauthorized")

app = FastAPI(title="RunGopher Conversation Evaluator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Directories ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
STRIPPED_DIR = OUTPUT_DIR / "stripped"
SUMMARIES_DIR = OUTPUT_DIR / "summaries"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [DATA_DIR, STRIPPED_DIR, SUMMARIES_DIR, CLUSTERS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── In-memory job tracking ───────────────────────────────────────────
jobs = {}  # job_id -> { status, phase, progress, logs, results, ... }


def create_job(filename: str, row_count: int) -> str:
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "id": job_id,
        "filename": filename,
        "row_count": row_count,
        "status": "uploaded",  # uploaded, running, completed, error
        "phase": 0,
        "phase_name": "Ready",
        "progress": 0,
        "logs": [],
        "results": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
    }
    return job_id


def log_to_job(job_id: str, message: str, level: str = "info"):
    if job_id in jobs:
        ts = datetime.now().strftime("%H:%M:%S")
        jobs[job_id]["logs"].append({
            "timestamp": ts,
            "message": message,
            "level": level,
        })


# ─── Routes ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main dashboard UI."""
    html_path = BASE_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.post("/api/auth")
async def auth(body: dict):
    """Validate password and return session token."""
    if not _VALID_TOKEN:
        return {"token": "open"}  # No password configured
    if body.get("password") == _APP_PASSWORD:
        return {"token": _VALID_TOKEN}
    raise HTTPException(401, "Incorrect password")


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...), _=Depends(require_auth)):
    """Upload a RunGopher conversation CSV export."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    # Save uploaded file
    save_path = DATA_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    # Parse to validate and count rows
    try:
        df = pd.read_csv(save_path, dtype={"Participant ID": str}, keep_default_na=False)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")

    # Validate required columns
    if "Transcript" not in df.columns:
        save_path.unlink(missing_ok=True)
        raise HTTPException(400, f"CSV missing 'Transcript' column. Found: {list(df.columns)}")

    # Filter empty transcripts
    valid = df[df["Transcript"].str.strip().astype(bool)]
    row_count = len(valid)

    # Create job
    job_id = create_job(file.filename, row_count)
    log_to_job(job_id, f"Uploaded {file.filename} — {row_count} conversations with transcripts")

    # Detect columns
    columns = list(df.columns)
    log_to_job(job_id, f"Columns detected: {', '.join(columns)}")

    return {
        "job_id": job_id,
        "filename": file.filename,
        "total_rows": len(df),
        "valid_conversations": row_count,
        "columns": columns,
    }


@app.post("/api/run/{job_id}")
async def run_pipeline(job_id: str, _=Depends(require_auth)):
    """Run the full 4-phase pipeline for a given job."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    batch_size = int(os.getenv("BATCH_SIZE", "300"))
    pii_threshold = float(os.getenv("PII_THRESHOLD", "0.4"))
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] == "running":
        raise HTTPException(409, "Pipeline already running for this job")

    job["status"] = "running"

    # Run in background
    asyncio.create_task(
        _run_pipeline_async(job_id, model, batch_size, pii_threshold)
    )

    return {"job_id": job_id, "status": "running"}


async def _run_pipeline_async(
    job_id: str, model: str, batch_size: int, pii_threshold: float
):
    """Execute the full pipeline asynchronously."""
    job = jobs[job_id]
    filename = job["filename"]
    input_path = DATA_DIR / filename

    try:
        # ─── Phase 1: PII Stripping ────────────────
        job["phase"] = 1
        job["phase_name"] = "PII Stripping"
        log_to_job(job_id, "Phase 1: PII Stripping (100% local processing)", "info")

        stripped_df, pii_stats = await asyncio.to_thread(
            _phase1_strip_pii, str(input_path), job_id, pii_threshold
        )

        stripped_path = STRIPPED_DIR / f"{Path(filename).stem}_stripped.csv"
        stripped_df.to_csv(stripped_path, index=False)
        log_to_job(job_id, f"✓ PII stripped: {sum(pii_stats.values()):,} entities replaced", "success")
        job["progress"] = 25

        # ─── Phase 2: Summarization ────────────────
        job["phase"] = 2
        job["phase_name"] = "Summarization"
        concurrency = int(os.getenv("CONCURRENCY", "10"))
        log_to_job(job_id, f"Phase 2: Summarization ({model}, {concurrency} concurrent)", "info")

        summary_df = await _phase2_summarize_async(stripped_df, job_id, model, concurrency)

        summary_path = SUMMARIES_DIR / f"{Path(filename).stem}_summaries.csv"
        summary_df.to_csv(summary_path, index=False)

        summary_only_path = SUMMARIES_DIR / f"{Path(filename).stem}_summaries_only.csv"
        cols = ["Participant ID", "Outcome", "Summary"]
        for c in ["Duration (Seconds)", "User Intent", "Direction"]:
            if c in summary_df.columns:
                cols.append(c)
        summary_df[cols].to_csv(summary_only_path, index=False)

        # Store per-conversation data (summary + truncated transcript) for chat queries
        job["summaries"] = [
            {
                "id": str(row.get("Participant ID", f"row_{i}")),
                "outcome": str(row.get("Outcome", "")),
                "summary": str(row.get("Summary", "")).split("|", 1)[-1].strip()
                    if "|" in str(row.get("Summary", "")) else str(row.get("Summary", "")),
                "transcript": str(row.get("Transcript", ""))[:1200],
            }
            for i, (_, row) in enumerate(summary_df.iterrows())
            if str(row.get("Outcome", "")) not in ("ERROR", "NO_TRANSCRIPT")
        ]
        job["summaries_path"] = str(summary_path)

        # Build 3-example samples per outcome for the report
        outcome_samples: dict = {}
        for s in job["summaries"]:
            outcome = s["outcome"]
            if outcome not in outcome_samples:
                outcome_samples[outcome] = []
            if len(outcome_samples[outcome]) < 3:
                outcome_samples[outcome].append({"id": s["id"], "summary": s["summary"]})

        log_to_job(job_id, f"✓ All conversations summarized", "success")
        job["progress"] = 50

        # ─── Phase 3: Clustering ───────────────────
        job["phase"] = 3
        job["phase_name"] = "Clustering"
        log_to_job(job_id, f"Phase 3: Clustering (batch size: {batch_size})", "info")

        cluster_data = await asyncio.to_thread(
            _phase3_cluster, summary_df, job_id, model, batch_size
        )

        cluster_path = CLUSTERS_DIR / "cluster_results.json"
        with open(cluster_path, "w") as f:
            json.dump(cluster_data, f, indent=2)

        log_to_job(job_id, f"✓ {len(cluster_data.get('clusters', []))} clusters identified", "success")
        job["progress"] = 75

        # ─── Phase 4: Report Generation ────────────
        job["phase"] = 4
        job["phase_name"] = "Report Generation"
        log_to_job(job_id, "Phase 4: Generating branded HTML report", "info")

        report_html = await asyncio.to_thread(
            _phase4_report, cluster_data, outcome_samples, job_id, model
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        report_path = REPORTS_DIR / f"evaluation_report_{timestamp}.html"
        report_path.write_text(report_html)

        log_to_job(job_id, f"✓ Report saved: {report_path.name}", "success")
        job["progress"] = 100

        # ─── Done ──────────────────────────────────
        job["status"] = "completed"
        job["phase_name"] = "Complete"
        job["completed_at"] = datetime.now().isoformat()
        job["results"] = {
            "total_conversations": len(stripped_df),
            "pii_stats": pii_stats,
            "pii_total": sum(pii_stats.values()),
            "cluster_data": cluster_data,
            "report_file": report_path.name,
            "outcome_distribution": summary_df["Outcome"].value_counts().to_dict() if "Outcome" in summary_df.columns else {},
        }
        log_to_job(job_id, "✓ Pipeline complete — all 4 phases finished", "success")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        log_to_job(job_id, f"✗ Pipeline failed: {str(e)}", "error")


# ─── Phase Implementations ─────────────────────────────────────────────

def _phase1_strip_pii(csv_path: str, job_id: str, threshold: float) -> tuple:
    """Phase 1: Strip PII using Presidio."""
    from scripts.utils.pii_patterns import get_all_custom_recognizers, ENTITY_PLACEHOLDERS
    from scripts.utils.brand import PII_METADATA_COLUMNS

    # Lazy import — only load heavy deps when needed
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    df = pd.read_csv(csv_path, dtype={"Participant ID": str}, keep_default_na=False)
    df = df[df["Transcript"].str.strip().astype(bool)].copy()
    df.reset_index(drop=True, inplace=True)

    # Build engines
    log_to_job(job_id, "Loading NLP model and custom recognizers...")
    analyzer = AnalyzerEngine()
    for rec in get_all_custom_recognizers():
        analyzer.registry.add_recognizer(rec)
    anonymizer = AnonymizerEngine()

    # Strip metadata phones
    for col in PII_METADATA_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "[PHONE]" if x and str(x).strip() else "")

    # Strip transcript PII
    entities_to_detect = list(ENTITY_PLACEHOLDERS.keys())
    operators = {
        et: OperatorConfig("replace", {"new_value": ph})
        for et, ph in ENTITY_PLACEHOLDERS.items()
    }

    total_stats = {}
    stripped = []

    for idx, row in df.iterrows():
        transcript = row["Transcript"]
        if not transcript or not isinstance(transcript, str) or not transcript.strip():
            stripped.append("")
            continue

        text = transcript.strip().replace("\\n", "\n")

        try:
            results = analyzer.analyze(
                text=text, language="en",
                entities=entities_to_detect, score_threshold=threshold,
            )
            if results:
                anon = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators)
                stripped.append(anon.text)
                for r in results:
                    total_stats[r.entity_type] = total_stats.get(r.entity_type, 0) + 1
            else:
                stripped.append(text)
        except Exception:
            stripped.append(text)

        if (idx + 1) % 100 == 0:
            log_to_job(job_id, f"Processed {idx + 1}/{len(df)} conversations...")

    df["Transcript"] = stripped
    return df, total_stats


async def _phase2_summarize_async(
    df: pd.DataFrame, job_id: str, model: str, concurrency: int
) -> pd.DataFrame:
    """Phase 2: Concurrent summarization with exponential backoff on rate limits."""
    from openai import AsyncOpenAI
    from scripts.utils.brand import OUTCOME_CATEGORIES

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(concurrency)
    total = len(df)
    completed = 0

    prompt_template = """You are analyzing a debt collection voice agent conversation.
PII has been redacted. Speakers: "AI" (agent) and "User" (debtor).

Generate exactly ONE sentence:
- OUTCOME: one of [{outcomes}]
- DISPOSITION (cooperative, hostile, confused, emotional, evasive, willing, unresponsive)
- NOTABLE event if any

Format: OUTCOME_CATEGORY | One sentence description

Transcript:
{transcript}"""

    async def summarize_one(transcript: str) -> tuple:
        nonlocal completed
        if not transcript.strip():
            completed += 1
            return "NO_TRANSCRIPT", "NO_TRANSCRIPT | Empty transcript"

        async with semaphore:
            for attempt in range(6):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt_template.format(
                            outcomes=", ".join(OUTCOME_CATEGORIES),
                            transcript=transcript[:8000],
                        )}],
                        temperature=0.3,
                        max_tokens=200,
                    )
                    result = resp.choices[0].message.content.strip()
                    if "|" in result:
                        parts = result.split("|", 1)
                        outcome = parts[0].strip().upper().replace(" ", "_")
                        summary = parts[1].strip()
                    else:
                        outcome = "OTHER"
                        summary = result
                    completed += 1
                    if completed % 50 == 0:
                        log_to_job(job_id, f"Summarized {completed}/{total}...")
                    return outcome, f"{outcome} | {summary}"

                except Exception as e:
                    err = str(e)
                    if "rate_limit" in err.lower() or "429" in err or "rate limit" in err.lower():
                        wait = min(2 ** attempt + 1, 60)
                        log_to_job(job_id, f"Rate limit hit — retrying in {wait}s (attempt {attempt + 1}/6)...", "warn")
                        await asyncio.sleep(wait)
                    else:
                        completed += 1
                        return "ERROR", f"ERROR | {err}"

            completed += 1
            return "ERROR", "ERROR | Max retries exceeded"

    transcripts = [str(row.get("Transcript", "")) for _, row in df.iterrows()]
    results = await asyncio.gather(*[summarize_one(t) for t in transcripts])

    df["Outcome"] = [r[0] for r in results]
    df["Summary"] = [r[1] for r in results]
    return df


def _phase3_cluster(
    df: pd.DataFrame, job_id: str, model: str, batch_size: int
) -> dict:
    """Phase 3: Cluster summaries in batches via OpenAI."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    valid_summaries = [
        s for s in df["Summary"].tolist()
        if s and "ERROR" not in s and "NO_TRANSCRIPT" not in s
    ]

    batches = [
        valid_summaries[i:i + batch_size]
        for i in range(0, len(valid_summaries), batch_size)
    ]

    log_to_job(job_id, f"Processing {len(valid_summaries)} summaries in {len(batches)} batches...")

    batch_results = []

    cluster_prompt = """Analyze {count} debt collection conversation summaries.
Each summary: OUTCOME_CATEGORY | description

1. CLUSTER into categories with counts + percentages
2. Per cluster: common patterns, outliers
3. TRENDS: dispositions, effective tactics, compliance concerns

Return valid JSON:
{{
  "total_conversations": {count},
  "clusters": [{{"name":"","count":0,"percentage":0.0,"common_patterns":[],"outliers":[]}}],
  "disposition_breakdown": [{{"disposition":"","count":0,"percentage":0.0}}],
  "top_tactics": [],
  "compliance_flags": [],
  "recommendations": []
}}

Summaries:
{summaries}"""

    for i, batch in enumerate(batches):
        log_to_job(job_id, f"Clustering batch {i + 1}/{len(batches)} ({len(batch)} summaries)...")
        summaries_text = "\n".join(f"{j+1}. {s}" for j, s in enumerate(batch))
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": cluster_prompt.format(
                        count=len(batch), summaries=summaries_text
                    )}],
                    temperature=0.2,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )
                result = json.loads(resp.choices[0].message.content.strip())
                batch_results.append(result)
                break
            except Exception as e:
                err = str(e)
                if ("rate_limit" in err.lower() or "429" in err or "rate limit" in err.lower()) and attempt < 4:
                    wait = min(2 ** attempt + 1, 60)
                    log_to_job(job_id, f"Rate limit — retrying batch {i+1} in {wait}s...", "warn")
                    time.sleep(wait)
                else:
                    log_to_job(job_id, f"⚠ Batch {i + 1} error: {err}", "warn")
                    break

    # Merge
    if not batch_results:
        return {"total_conversations": 0, "clusters": [], "top_tactics": [], "compliance_flags": [], "recommendations": []}

    if len(batch_results) == 1:
        return batch_results[0]

    merged = {
        "total_conversations": sum(b.get("total_conversations", 0) for b in batch_results),
        "clusters": [],
        "top_tactics": [],
        "compliance_flags": [],
        "recommendations": [],
    }

    cluster_map = {}
    for batch in batch_results:
        for c in batch.get("clusters", []):
            name = c["name"]
            if name in cluster_map:
                cluster_map[name]["count"] += c.get("count", 0)
                cluster_map[name]["common_patterns"].extend(c.get("common_patterns", []))
            else:
                cluster_map[name] = {**c, "common_patterns": list(c.get("common_patterns", []))}

    total = merged["total_conversations"] or 1
    for c in cluster_map.values():
        c["percentage"] = round(c["count"] / total * 100, 1)
        c["common_patterns"] = list(set(c["common_patterns"]))[:5]
    merged["clusters"] = sorted(cluster_map.values(), key=lambda x: -x["count"])

    for batch in batch_results:
        merged["top_tactics"].extend(batch.get("top_tactics", []))
        merged["compliance_flags"].extend(batch.get("compliance_flags", []))
        merged["recommendations"].extend(batch.get("recommendations", []))
    merged["top_tactics"] = list(set(merged["top_tactics"]))[:10]
    merged["compliance_flags"] = list(set(merged["compliance_flags"]))[:10]
    merged["recommendations"] = list(set(merged["recommendations"]))[:10]

    return merged


def _phase4_report(cluster_data: dict, outcome_samples: dict, job_id: str, model: str) -> str:
    """Phase 4: Build HTML report programmatically; use LLM only for executive summary."""
    from openai import OpenAI
    from scripts.utils.brand import BRAND

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    colors = BRAND["colors"]
    typo = BRAND["typography"]
    report_date = datetime.now().strftime("%B %d, %Y")

    clusters = cluster_data.get("clusters", [])
    total = cluster_data.get("total_conversations", 0)

    # ── Resolution Rate ───────────────────────────────────────────────────
    cluster_by_name = {c["name"]: c["count"] for c in clusters}
    voicemail_count = cluster_by_name.get("VOICEMAIL", 0)
    no_answer_count = cluster_by_name.get("NO_ANSWER", 0)
    connected_calls = total - voicemail_count - no_answer_count
    resolved_calls = cluster_by_name.get("PAYMENT_ARRANGED", 0) + cluster_by_name.get("TRANSFERRED", 0)
    resolution_rate = round(resolved_calls / connected_calls * 100, 1) if connected_calls > 0 else 0.0

    # ── LLM: executive summary only ──────────────────────────────────────
    log_to_job(job_id, "Generating executive summary...")
    breakdown = [{"name": c["name"], "count": c["count"], "pct": c["percentage"]} for c in clusters]
    summary_prompt = f"""Write a 3-sentence executive summary for a debt collection voice agent evaluation.

Data:
- Total conversations: {total:,}
- Connected calls (user picked up, excluding voicemail/no-answer): {connected_calls:,}
- Resolution rate: {resolution_rate}% ({resolved_calls} resolved out of {connected_calls} connected calls)
- Outcome breakdown: {json.dumps(breakdown)}
- Compliance flags identified: {len(cluster_data.get('compliance_flags', []))}

Be specific — use real numbers. Always mention the resolution rate. No heading, just 3 sentences."""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    exec_summary = resp.choices[0].message.content.strip()

    log_to_job(job_id, "Building report HTML...")

    # ── Helpers ───────────────────────────────────────────────────────────
    POSITIVE = {"PAYMENT_ARRANGED", "CALLBACK_SCHEDULED", "TRANSFERRED"}
    NEGATIVE = {"REFUSED", "HUNG_UP", "COMPLIANCE_ISSUE", "DISPUTE"}

    DISPLAY_NAMES = {
        "PAYMENT_ARRANGED": "Payment Arranged",
        "CALLBACK_SCHEDULED": "Callback Scheduled",
        "TRANSFERRED": "Transferred",
        "VOICEMAIL": "Voicemail",
        "NO_ANSWER": "No Answer",
        "HUNG_UP": "Hung Up",
        "WRONG_NUMBER": "Wrong Number",
        "REFUSED": "Refused",
        "DISPUTE": "Dispute",
        "HARDSHIP_CLAIM": "Hardship Claim",
        "COMPLIANCE_ISSUE": "Compliance Issue",
        "CONFUSED": "Confused",
        "OTHER": "Other",
    }

    def display_name(name):
        return DISPLAY_NAMES.get(name, name.replace("_", " ").title())

    def outcome_color(name):
        if name in POSITIVE:
            return colors["cobalt"]
        if name in NEGATIVE:
            return colors["coral"]
        return colors["muted"]

    # ── Outcome bar chart ─────────────────────────────────────────────────
    bar_html = ""
    for c in clusters:
        color = outcome_color(c["name"])
        pct = c["percentage"]
        bar_html += f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-family:{typo['heading']};font-weight:600;font-size:14px;">{display_name(c['name'])}</span>
            <span style="color:{colors['muted']};font-size:13px;">{c['count']:,} &nbsp;({pct}%)</span>
          </div>
          <div style="background:#e5e7eb;border-radius:8px;height:22px;overflow:hidden;">
            <div style="background:{color};width:{max(pct, 0.5)}%;height:100%;border-radius:8px;"></div>
          </div>
        </div>"""

    # ── Cluster detail cards ──────────────────────────────────────────────
    cluster_cards = ""
    for c in clusters:
        color = outcome_color(c["name"])
        patterns = "".join(f"<li style='margin-bottom:4px;'>{p}</li>" for p in c.get("common_patterns", []))
        outliers = c.get("outliers", [])
        outlier_html = (
            f"<p style='margin:12px 0 0;color:{colors['muted']};font-size:13px;'>"
            f"<strong>Outliers:</strong> {'; '.join(outliers)}</p>"
            if outliers else ""
        )
        cluster_cards += f"""
        <div style="background:{colors['sand']};border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.07);
                    padding:24px;margin-bottom:16px;border-left:5px solid {color};">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <h3 style="margin:0;font-family:{typo['heading']};font-weight:600;">{display_name(c['name'])}</h3>
            <span style="background:{color};color:white;padding:3px 12px;border-radius:20px;font-size:13px;white-space:nowrap;">
              {c['count']:,} &nbsp;({c['percentage']}%)
            </span>
          </div>
          <ul style="margin:0;padding-left:20px;">{patterns}</ul>
          {outlier_html}
        </div>"""

    # ── Sample conversations ───────────────────────────────────────────────
    sample_html = ""
    for outcome, samples in outcome_samples.items():
        if not samples:
            continue
        color = outcome_color(outcome)
        cards = ""
        for s in samples:
            cards += f"""
            <div style="background:white;border-radius:8px;padding:14px;margin-bottom:8px;border-left:3px solid {color};">
              <div style="font-size:11px;color:{colors['muted']};margin-bottom:5px;">ID: {s['id']}</div>
              <p style="margin:0;font-size:14px;">{s['summary']}</p>
            </div>"""
        sample_html += f"""
        <div style="margin-bottom:28px;">
          <h3 style="font-family:{typo['heading']};font-weight:600;color:{color};margin-bottom:10px;">{display_name(outcome)}</h3>
          {cards}
        </div>"""

    # ── Tactics, flags, recommendations ──────────────────────────────────
    tactics_html = "".join(
        f"<li style='margin-bottom:8px;'>{t}</li>"
        for t in cluster_data.get("top_tactics", [])
    )
    flags_html = "".join(
        f"<div style='border:2px solid {colors['coral']};border-radius:12px;padding:16px;"
        f"margin-bottom:10px;background:rgba(255,28,77,0.04);'>{f}</div>"
        for f in cluster_data.get("compliance_flags", [])
    )
    recs_html = "".join(
        f"<div style='border:2px solid {colors['cobalt']};border-radius:12px;padding:16px;"
        f"margin-bottom:10px;background:rgba(50,89,254,0.04);'>{r}</div>"
        for r in cluster_data.get("recommendations", [])
    )

    # ── Assemble HTML ─────────────────────────────────────────────────────
    section = lambda title, body: (
        f"<div style='margin-bottom:52px;'>"
        f"<h2 style='font-family:{typo['heading']};font-weight:600;margin-bottom:20px;"
        f"padding-bottom:10px;border-bottom:2px solid {colors['sand']};'>{title}</h2>"
        f"{body}</div>"
    )

    safe_date = report_date.replace(" ", "_")

    resolution_card = f"""
    <div style="background:{colors['navy']};color:white;padding:32px 24px;">
      <div style="max-width:920px;margin:0 auto;display:flex;gap:24px;flex-wrap:wrap;">
        <div style="flex:1;min-width:200px;background:rgba(255,255,255,0.08);border-radius:12px;
                    padding:24px;text-align:center;">
          <div style="font-size:42px;font-weight:600;font-family:'Poppins',sans-serif;
                      color:{colors['cobalt'] if resolution_rate >= 10 else colors['coral']};">
            {resolution_rate}%
          </div>
          <div style="font-size:15px;font-weight:600;font-family:'Poppins',sans-serif;margin-top:6px;">
            Resolution Rate
          </div>
          <div style="font-size:12px;opacity:0.65;margin-top:4px;">
            {resolved_calls:,} resolved / {connected_calls:,} connected calls
          </div>
          <div style="font-size:11px;opacity:0.5;margin-top:2px;">
            Transferred + Payment Arranged
          </div>
        </div>
        <div style="flex:1;min-width:200px;background:rgba(255,255,255,0.08);border-radius:12px;
                    padding:24px;text-align:center;">
          <div style="font-size:42px;font-weight:600;font-family:'Poppins',sans-serif;">
            {connected_calls:,}
          </div>
          <div style="font-size:15px;font-weight:600;font-family:'Poppins',sans-serif;margin-top:6px;">
            Connected Calls
          </div>
          <div style="font-size:12px;opacity:0.65;margin-top:4px;">
            of {total:,} total conversations
          </div>
          <div style="font-size:11px;opacity:0.5;margin-top:2px;">
            Excludes voicemail &amp; no-answer
          </div>
        </div>
        <div style="flex:1;min-width:200px;background:rgba(255,255,255,0.08);border-radius:12px;
                    padding:24px;text-align:center;">
          <div style="font-size:42px;font-weight:600;font-family:'Poppins',sans-serif;">
            {voicemail_count + no_answer_count:,}
          </div>
          <div style="font-size:15px;font-weight:600;font-family:'Poppins',sans-serif;margin-top:6px;">
            Unreachable
          </div>
          <div style="font-size:12px;opacity:0.65;margin-top:4px;">
            {round((voicemail_count + no_answer_count) / total * 100, 1) if total else 0}% of total
          </div>
          <div style="font-size:11px;opacity:0.5;margin-top:2px;">
            Voicemail + no-answer
          </div>
        </div>
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RunGopher Conversation Evaluation Report</title>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Recursive:wght@400&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
  <style>
    body {{ font-family:'Recursive',Arial,sans-serif; margin:0; padding:0;
           background:{colors['white']}; color:{colors['black']}; line-height:1.6; }}
    h1,h2,h3 {{ font-family:'Poppins',Arial,sans-serif; font-weight:600; }}
    #pdf-btn {{
      position:fixed; bottom:28px; right:28px; z-index:999;
      background:{colors['cobalt']}; color:white; border:none; border-radius:10px;
      padding:12px 22px; font-size:15px; font-family:'Poppins',sans-serif;
      font-weight:600; cursor:pointer; box-shadow:0 4px 16px rgba(50,89,254,0.35);
    }}
    #pdf-btn:hover {{ opacity:0.9; }}
    @media print {{ #pdf-btn {{ display:none; }} }}
  </style>
</head>
<body>
  <button id="pdf-btn" onclick="downloadPDF()">&#8595; Download PDF</button>

  <div id="report-content">
    <div style="background:{colors['navy']};color:white;padding:56px 24px 32px;text-align:center;">
      <h1 style="margin:0 0 8px;font-size:34px;">RunGopher</h1>
      <p style="margin:0 0 6px;font-size:18px;opacity:0.85;font-family:'Poppins',sans-serif;">Conversation Evaluation Report</p>
      <p style="margin:0;opacity:0.65;font-size:14px;">{report_date} &nbsp;·&nbsp; {total:,} conversations analysed</p>
    </div>

    {resolution_card}

    <div style="max-width:920px;margin:0 auto;padding:52px 24px;">
      {section("Executive Summary", f"<p style='font-size:16px;line-height:1.8;'>{exec_summary}</p>")}
      {section("Outcome Distribution", bar_html)}
      {section("Cluster Detail", cluster_cards)}
      {section("Sample Conversations", sample_html or f"<p style='color:{colors['muted']};font-style:italic;'>No samples available.</p>")}
      {section("Top Tactics", f"<ol style='padding-left:24px;'>{tactics_html}</ol>")}
      {section("Compliance Flags", flags_html)}
      {section("Recommendations", recs_html)}
    </div>

    <footer style="text-align:center;padding:28px;background:{colors['sand']};
                   color:{colors['muted']};font-size:13px;">
      Powered by RunGopher
    </footer>
  </div>

  <script>
    function downloadPDF() {{
      const btn = document.getElementById('pdf-btn');
      btn.textContent = 'Generating...';
      btn.disabled = true;
      const element = document.getElementById('report-content');
      html2pdf().set({{
        margin: 0,
        filename: 'RunGopher_Evaluation_{safe_date}.pdf',
        image: {{ type: 'jpeg', quality: 0.98 }},
        html2canvas: {{ scale: 2, useCORS: true }},
        jsPDF: {{ unit: 'mm', format: 'a4', orientation: 'portrait' }}
      }}).from(element).save().then(() => {{
        btn.textContent = '↓ Download PDF';
        btn.disabled = false;
      }});
    }}
  </script>
</body>
</html>"""


# ─── Chat Endpoint ──────────────────────────────────────────────────────

@app.post("/api/jobs/{job_id}/chat")
async def chat(job_id: str, body: dict, _=Depends(require_auth)):
    """Chat with the evaluation report data."""
    from openai import OpenAI

    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "completed" or not job.get("results"):
        raise HTTPException(400, "Job not completed yet")

    message = body.get("message", "").strip()
    history = body.get("history", [])  # [{"role": "user"|"assistant", "content": "..."}]

    if not message:
        raise HTTPException(400, "Message is required")

    results = job["results"]
    cluster_data = results.get("cluster_data", {})
    summaries = job.get("summaries", [])

    # Build a compact index: ID, outcome, one-line summary (no transcripts yet)
    index_text = "\n".join(
        f"{s['id']}: [{s['outcome']}] {s['summary']}"
        for s in summaries
    ) if summaries else "(no individual conversation data available)"

    # Include up to 3 full transcripts per outcome category so the model
    # can actually show conversations when asked
    outcome_transcript_samples: dict = {}
    for s in summaries:
        outcome = s["outcome"]
        transcript = s.get("transcript", "").strip()
        if transcript and outcome not in outcome_transcript_samples:
            outcome_transcript_samples[outcome] = []
        if transcript and len(outcome_transcript_samples.get(outcome, [])) < 3:
            outcome_transcript_samples.setdefault(outcome, []).append(
                {"id": s["id"], "transcript": transcript}
            )

    transcript_samples_text = ""
    for outcome, samples in outcome_transcript_samples.items():
        transcript_samples_text += f"\n\n--- {outcome} ---"
        for sample in samples:
            transcript_samples_text += f"\n[ID: {sample['id']}]\n{sample['transcript']}\n"

    system_prompt = f"""You are an analyst assistant helping review a RunGopher debt collection voice agent evaluation report.
Answer questions based only on the data provided below.

EVALUATION SUMMARY:
- Total conversations analyzed: {results.get('total_conversations', 0)}
- PII instances stripped: {results.get('pii_total', 0)}
- Outcome distribution: {json.dumps(results.get('outcome_distribution', {}))}

CLUSTERS:
{json.dumps(cluster_data.get('clusters', []), indent=2)}

TOP TACTICS:
{json.dumps(cluster_data.get('top_tactics', []), indent=2)}

COMPLIANCE FLAGS:
{json.dumps(cluster_data.get('compliance_flags', []), indent=2)}

RECOMMENDATIONS:
{json.dumps(cluster_data.get('recommendations', []), indent=2)}

CONVERSATION INDEX ({len(summaries)} total — ID: [OUTCOME] summary):
{index_text}

SAMPLE FULL TRANSCRIPTS (PII-stripped, up to 3 per outcome category):
{transcript_samples_text if transcript_samples_text else "(none available)"}

When asked to show a conversation or give an example, quote the actual transcript from the samples above.
When asked about a specific outcome type, show the matching transcript sample(s) with their Participant IDs.
Be direct and specific. Use numbers and percentages from the data where relevant."""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-10:])  # Keep last 10 turns for context
    messages.append({"role": "user", "content": message})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )

    return {"reply": resp.choices[0].message.content.strip()}


# ─── Status & Results Endpoints ─────────────────────────────────────────

@app.get("/api/jobs")
async def list_jobs(_=Depends(require_auth)):
    """List all jobs."""
    return sorted(jobs.values(), key=lambda j: j["created_at"], reverse=True)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, _=Depends(require_auth)):
    """Get job status, logs, and results."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/jobs/{job_id}/report")
async def download_report(job_id: str, _=Depends(require_auth)):
    """Download the generated HTML report."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job["status"] != "completed" or not job.get("results", {}).get("report_file"):
        raise HTTPException(400, "Report not ready yet")

    report_path = REPORTS_DIR / job["results"]["report_file"]
    if not report_path.exists():
        raise HTTPException(404, "Report file not found")

    return FileResponse(report_path, media_type="text/html", filename=report_path.name)


@app.get("/api/health")
async def health():
    """Health check."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    ready = bool(api_key and not api_key.startswith("sk-your"))
    return {
        "status": "ok" if ready else "misconfigured",
        "ready": ready,
        "active_jobs": len([j for j in jobs.values() if j["status"] == "running"]),
    }


# ─── Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  RunGopher Conversation Evaluator")
    print("  http://localhost:8000")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
