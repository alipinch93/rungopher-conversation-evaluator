#!/usr/bin/env python3
"""
Phase 4: Branded Report Generation
Takes cluster analysis JSON and generates a RunGopher-branded HTML report.

Usage:
    python scripts/04_generate_report.py --input output/clusters/cluster_results.json --output output/reports/

Output:
    - Branded HTML report using RunGopher design system
    - Standalone file (all CSS inline, no external dependencies)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.brand import BRAND

load_dotenv(Path(__file__).parent.parent / ".env")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        print("ERROR: Set your OpenAI API key in .env file")
        sys.exit(1)
    return OpenAI(api_key=api_key)


REPORT_PROMPT = """Generate a complete, standalone HTML report for a debt collection conversation evaluation.
Use the data provided below to populate the report.

BRAND GUIDELINES (MANDATORY):
- Background: {white}
- Card backgrounds: {sand}
- Primary accent (highlights, alerts, negative outcomes): {coral}
- Secondary accent (charts, links, positive outcomes): {cobalt}
- Dark sections (hero header): {navy}
- Body text: {black}
- Muted text: {muted}
- Headings font: {heading_font}; font-weight: {heading_weight}
- Body font: {body_font}; font-weight: {body_weight}
- 50% white space rule — generous padding (40px+ sections)
- Subtle shadows on cards: box-shadow: 0 4px 20px rgba(0,0,0,0.06)
- Border-radius: 12px on cards
- Import Google Fonts Poppins and Recursive in the HTML <head>

REPORT STRUCTURE (follow exactly):
1. Full-width NAVY hero banner with white text:
   - "RunGopher" in top-left
   - Report title: "Conversation Evaluation Report"
   - Subtitle: date range + total conversations analyzed
2. Executive Summary section — 2-3 sentences on overall performance
3. Outcome Distribution — visual bar chart using CSS (no JS libraries):
   - Use {cobalt} for positive outcomes (payment, callback)
   - Use {coral} for negative outcomes (refused, hung up, compliance issue)
   - Use {sand} for neutral (voicemail, other)
   - Show count and percentage labels
4. Cluster Detail Cards — sand-colored cards with:
   - Cluster name as heading
   - Count + percentage badge
   - Common patterns as bullet list
   - Outliers in italic
5. Top Insights — numbered list with key findings
6. Compliance Flags — {coral}-bordered warning cards (if any flags exist)
7. Recommendations — {cobalt}-accented numbered action items
8. Footer — centered, muted text: "Powered by RunGopher | Generated [date]"

RULES:
- Output ONLY valid HTML. No markdown, no explanation, no code fences.
- All CSS must be inline in a <style> tag in the <head>. No external stylesheets except Google Fonts.
- No JavaScript required.
- The HTML must be completely self-contained and render correctly when opened in a browser.
- Use semantic HTML (section, article, header, footer).
- Make it look premium and polished — this is a client-facing report.

DATA:
{data_json}"""


def generate_report(client: OpenAI, cluster_data: dict, model: str = None) -> str:
    """
    Generate a branded HTML report from cluster analysis data.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    colors = BRAND["colors"]
    typo = BRAND["typography"]

    prompt = REPORT_PROMPT.format(
        white=colors["white"],
        sand=colors["sand"],
        coral=colors["coral"],
        cobalt=colors["cobalt"],
        navy=colors["navy"],
        black=colors["black"],
        muted=colors["muted"],
        heading_font=typo["heading"],
        heading_weight=typo["heading_weight"],
        body_font=typo["body"],
        body_weight=typo["body_weight"],
        data_json=json.dumps(cluster_data, indent=2),
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=8000,
    )

    html = response.choices[0].message.content.strip()

    # Clean up if the model wraps in code fences
    if html.startswith("```"):
        lines = html.split("\n")
        html = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate a RunGopher-branded HTML evaluation report"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to cluster_results.json"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for reports"
    )
    parser.add_argument(
        "--model", "-m", default=None, help="OpenAI model"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 4: Branded Report Generation")
    print("=" * 60)

    # Load cluster data
    with open(args.input) as f:
        cluster_data = json.load(f)

    total = cluster_data.get("total_conversations", 0)
    clusters = len(cluster_data.get("clusters", []))
    print(f"Loaded cluster data: {total} conversations, {clusters} clusters")

    # Generate report
    client = get_client()
    print("Generating branded HTML report...")
    html = generate_report(client, cluster_data, args.model)

    # Save HTML report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    html_path = output_dir / f"evaluation_report_{timestamp}.html"
    with open(html_path, "w") as f:
        f.write(html)

    print("\n" + "=" * 60)
    print("Report Generation Complete")
    print("=" * 60)
    print(f"  HTML report: {html_path}")
    print(f"  Open in browser: file://{html_path.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
