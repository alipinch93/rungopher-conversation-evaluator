#!/usr/bin/env python3
"""
Phase 2: Summarization
Generates 1-sentence summaries for each PII-stripped conversation using OpenAI.
Each summary follows the format: OUTCOME_CATEGORY | description

Usage:
    python scripts/02_summarize.py --input output/stripped/calls_stripped.csv --output output/summaries/

Output:
    - CSV with original metadata + new "Summary" and "Outcome" columns
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.brand import CSV_COLUMNS, OUTCOME_CATEGORIES

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

SUMMARY_PROMPT = """You are analyzing a debt collection voice agent conversation.
The PII has been redacted with placeholders like [PERSON], [PHONE], [DOB], [AMOUNT], etc.
Speakers are labeled "AI" (the collection agent) and "User" (the debtor).

Generate exactly ONE sentence that captures:
- The call OUTCOME: one of [{outcomes}]
- The debtor's DISPOSITION (cooperative, hostile, confused, emotional, evasive, willing, unresponsive, etc.)
- Any NOTABLE event (escalation, compliance_issue, agent_error, successful_tactic, identity_not_verified, etc.)

Format your response EXACTLY as:
OUTCOME_CATEGORY | One sentence description

Example:
PAYMENT_ARRANGED | Cooperative debtor agreed to a 3-installment plan after agent offered a settlement discount.

Transcript:
{transcript}"""


def get_client() -> OpenAI:
    """Initialize OpenAI client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        print("ERROR: Set your OpenAI API key in .env file")
        print("  cp .env.example .env && edit .env")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def summarize_conversation(
    client: OpenAI, transcript: str, model: str = None
) -> tuple[str, str]:
    """
    Generate a 1-sentence summary for a single conversation.

    Returns:
        tuple: (outcome_category, full_summary_line)
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = SUMMARY_PROMPT.format(
        outcomes=", ".join(OUTCOME_CATEGORIES),
        transcript=transcript[:8000],  # Truncate very long transcripts
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        result = response.choices[0].message.content.strip()

        # Parse outcome category
        if "|" in result:
            parts = result.split("|", 1)
            outcome = parts[0].strip().upper().replace(" ", "_")
            summary = parts[1].strip()
        else:
            outcome = "OTHER"
            summary = result

        return outcome, f"{outcome} | {summary}"

    except Exception as e:
        return "ERROR", f"ERROR | Failed to summarize: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1-sentence summaries for PII-stripped conversations"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to PII-stripped CSV"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for summaries"
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="OpenAI model (default: from .env or gpt-4o-mini)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds between API calls (default: 0.5)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2: Summarization")
    print("=" * 60)

    # Load stripped conversations
    df = pd.read_csv(args.input, dtype={"Call ID": str}, keep_default_na=False)
    print(f"Loaded {len(df)} PII-stripped conversations")

    # Initialize OpenAI client
    client = get_client()
    model = args.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    print(f"Using model: {model}")

    # Generate summaries
    outcomes = []
    summaries = []
    errors = 0

    for idx, row in df.iterrows():
        transcript = row.get("Transcript", "")

        if not transcript.strip():
            outcomes.append("NO_TRANSCRIPT")
            summaries.append("NO_TRANSCRIPT | Empty or missing transcript")
            continue

        outcome, summary = summarize_conversation(client, transcript, model)
        outcomes.append(outcome)
        summaries.append(summary)

        if outcome == "ERROR":
            errors += 1

        if (idx + 1) % 50 == 0:
            print(f"  Summarized {idx + 1}/{len(df)}...")

        # Rate limiting
        time.sleep(args.rate_limit)

    # Add summary columns to DataFrame
    df["Outcome"] = outcomes
    df["Summary"] = summaries

    # Save full CSV with summaries
    input_name = Path(args.input).stem
    output_csv = output_dir / f"{input_name}_summaries.csv"
    df.to_csv(output_csv, index=False)

    # Also save a lightweight summaries-only file for clustering
    summary_df = df[["Call ID", "Outcome", "Summary", "Duration (Seconds)", "User Intent", "Direction"]].copy()
    summary_only_path = output_dir / f"{input_name}_summaries_only.csv"
    summary_df.to_csv(summary_only_path, index=False)

    # Print stats
    print("\n" + "=" * 60)
    print("Summarization Complete")
    print("=" * 60)
    print(f"  Conversations summarized: {len(df)}")
    print(f"  Errors: {errors}")
    print(f"  Full output: {output_csv}")
    print(f"  Summaries only: {summary_only_path}")

    print(f"\n  Outcome distribution:")
    outcome_counts = pd.Series(outcomes).value_counts()
    for outcome, count in outcome_counts.items():
        pct = count / len(outcomes) * 100
        print(f"    {outcome:25s} {count:5d}  ({pct:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
