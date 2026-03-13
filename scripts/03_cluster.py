#!/usr/bin/env python3
"""
Phase 3: Clustering
Batches conversation summaries (200-500 per batch) and sends to OpenAI
for pattern analysis, trend identification, and compliance flagging.

Usage:
    python scripts/03_cluster.py --input output/summaries/calls_stripped_summaries_only.csv --output output/clusters/

Output:
    - JSON file with cluster analysis results
    - Markdown summary
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv(Path(__file__).parent.parent / ".env")

CLUSTER_PROMPT = """You are analyzing {count} debt collection conversation summaries.
Each summary is in the format: OUTCOME_CATEGORY | description

Perform the following analysis:

1. CLUSTER these conversations into categories with exact counts and percentages.
2. For each cluster, identify:
   - Common patterns or tactics observed
   - Notable outliers worth investigating
3. Identify TRENDS across all conversations:
   - Most common debtor dispositions
   - Most effective agent tactics (what led to payments or positive outcomes?)
   - Compliance concerns flagged
   - Opportunities for agent/prompt improvement

Return your analysis as valid JSON with this exact structure:
{{
  "total_conversations": {count},
  "clusters": [
    {{
      "name": "Category Name",
      "count": 0,
      "percentage": 0.0,
      "common_patterns": ["pattern 1", "pattern 2"],
      "outliers": ["outlier observation"]
    }}
  ],
  "disposition_breakdown": [
    {{"disposition": "cooperative", "count": 0, "percentage": 0.0}}
  ],
  "top_tactics": ["tactic 1", "tactic 2"],
  "compliance_flags": ["flag 1"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}

Summaries:
{summaries}"""


def get_client() -> OpenAI:
    """Initialize OpenAI client from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        print("ERROR: Set your OpenAI API key in .env file")
        sys.exit(1)
    return OpenAI(api_key=api_key)


def cluster_batch(
    client: OpenAI, summaries: list[str], model: str = None
) -> dict:
    """
    Send a batch of summaries to OpenAI for clustering analysis.

    Args:
        client: OpenAI client
        summaries: List of summary strings
        model: OpenAI model to use

    Returns:
        dict: Cluster analysis results
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    summaries_text = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(summaries)
    )

    prompt = CLUSTER_PROMPT.format(
        count=len(summaries),
        summaries=summaries_text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    result = response.choices[0].message.content.strip()

    try:
        return json.loads(result)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(result[start:end])
        raise


def merge_cluster_results(batch_results: list[dict]) -> dict:
    """
    Merge cluster analysis results from multiple batches into a
    single consolidated result.
    """
    if len(batch_results) == 1:
        return batch_results[0]

    merged = {
        "total_conversations": sum(b.get("total_conversations", 0) for b in batch_results),
        "clusters": [],
        "disposition_breakdown": [],
        "top_tactics": [],
        "compliance_flags": [],
        "recommendations": [],
        "batch_count": len(batch_results),
    }

    # Merge clusters by name
    cluster_map = {}
    for batch in batch_results:
        for cluster in batch.get("clusters", []):
            name = cluster["name"]
            if name in cluster_map:
                cluster_map[name]["count"] += cluster.get("count", 0)
                cluster_map[name]["common_patterns"].extend(
                    cluster.get("common_patterns", [])
                )
                cluster_map[name]["outliers"].extend(
                    cluster.get("outliers", [])
                )
            else:
                cluster_map[name] = {
                    "name": name,
                    "count": cluster.get("count", 0),
                    "common_patterns": list(cluster.get("common_patterns", [])),
                    "outliers": list(cluster.get("outliers", [])),
                }

    # Recalculate percentages
    total = merged["total_conversations"] or 1
    for cluster in cluster_map.values():
        cluster["percentage"] = round(cluster["count"] / total * 100, 1)
        # Deduplicate patterns
        cluster["common_patterns"] = list(set(cluster["common_patterns"]))[:5]
        cluster["outliers"] = list(set(cluster["outliers"]))[:3]

    merged["clusters"] = sorted(
        cluster_map.values(), key=lambda x: -x["count"]
    )

    # Merge other fields (deduplicate)
    for batch in batch_results:
        merged["top_tactics"].extend(batch.get("top_tactics", []))
        merged["compliance_flags"].extend(batch.get("compliance_flags", []))
        merged["recommendations"].extend(batch.get("recommendations", []))

    merged["top_tactics"] = list(set(merged["top_tactics"]))[:10]
    merged["compliance_flags"] = list(set(merged["compliance_flags"]))[:10]
    merged["recommendations"] = list(set(merged["recommendations"]))[:10]

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Cluster conversation summaries into patterns and trends"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to summaries CSV"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for cluster results"
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=300,
        help="Summaries per batch (default: 300, range: 200-500)",
    )
    parser.add_argument(
        "--model", "-m", default=None, help="OpenAI model"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 3: Clustering")
    print("=" * 60)

    # Load summaries
    df = pd.read_csv(args.input, dtype={"Participant ID": str}, keep_default_na=False)
    summaries = df["Summary"].tolist()
    summaries = [s for s in summaries if s and "ERROR" not in s and "NO_TRANSCRIPT" not in s]
    print(f"Loaded {len(summaries)} valid summaries")

    # Batch
    batch_size = max(200, min(500, args.batch_size))
    batches = [
        summaries[i : i + batch_size]
        for i in range(0, len(summaries), batch_size)
    ]
    print(f"Split into {len(batches)} batches of ~{batch_size}")

    # Cluster each batch
    client = get_client()
    batch_results = []

    for i, batch in enumerate(batches):
        print(f"\n  Clustering batch {i + 1}/{len(batches)} ({len(batch)} summaries)...")
        try:
            result = cluster_batch(client, batch, args.model)
            batch_results.append(result)
            print(f"    → Found {len(result.get('clusters', []))} clusters")
        except Exception as e:
            print(f"    ⚠ Error on batch {i + 1}: {e}")

    if not batch_results:
        print("ERROR: No batches produced results")
        sys.exit(1)

    # Merge batch results
    print("\nMerging batch results...")
    merged = merge_cluster_results(batch_results)

    # Save JSON
    json_path = output_dir / "cluster_results.json"
    with open(json_path, "w") as f:
        json.dump(merged, f, indent=2)

    # Save Markdown summary
    md_path = output_dir / "cluster_summary.md"
    with open(md_path, "w") as f:
        f.write(f"# Conversation Cluster Analysis\n\n")
        f.write(f"**Total Conversations:** {merged['total_conversations']}\n\n")
        f.write("## Outcome Clusters\n\n")
        f.write("| Category | Count | % |\n|----------|-------|---|\n")
        for c in merged["clusters"]:
            f.write(f"| {c['name']} | {c['count']} | {c['percentage']}% |\n")
        f.write("\n## Top Tactics\n\n")
        for t in merged.get("top_tactics", []):
            f.write(f"- {t}\n")
        f.write("\n## Compliance Flags\n\n")
        for flag in merged.get("compliance_flags", []):
            f.write(f"- ⚠️ {flag}\n")
        f.write("\n## Recommendations\n\n")
        for i, rec in enumerate(merged.get("recommendations", []), 1):
            f.write(f"{i}. {rec}\n")

    print("\n" + "=" * 60)
    print("Clustering Complete")
    print("=" * 60)
    print(f"  Clusters found: {len(merged['clusters'])}")
    print(f"  JSON output: {json_path}")
    print(f"  Markdown output: {md_path}")
    for c in merged["clusters"][:5]:
        print(f"    {c['name']:30s} {c['count']:5d}  ({c['percentage']}%)")
    if len(merged["clusters"]) > 5:
        print(f"    ... and {len(merged['clusters']) - 5} more")
    print("=" * 60)


if __name__ == "__main__":
    main()
