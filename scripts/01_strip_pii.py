#!/usr/bin/env python3
"""
Phase 1: PII Stripping
Strips all PII from RunGopher conversation CSV exports using Microsoft Presidio.
Runs 100% locally — no data leaves your machine.

Usage:
    python scripts/01_strip_pii.py --input data/calls.csv --output output/stripped/

Output:
    - CSV with phone columns redacted and transcript PII replaced with typed placeholders
    - Summary stats of PII entities found
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils.csv_loader import load_conversations, extract_transcript_text
from scripts.utils.pii_patterns import get_all_custom_recognizers, ENTITY_PLACEHOLDERS
from scripts.utils.brand import PII_METADATA_COLUMNS


def build_analyzer() -> AnalyzerEngine:
    """
    Build a Presidio AnalyzerEngine with custom debt-collection recognizers.
    Downloads spaCy model if not present.
    """
    # Ensure spaCy model is available
    try:
        import spacy
        spacy.load("en_core_web_lg")
    except OSError:
        print("Downloading spaCy model (one-time)...")
        os.system("python -m spacy download en_core_web_lg")

    analyzer = AnalyzerEngine()

    # Register custom recognizers for debt collection PII
    for recognizer in get_all_custom_recognizers():
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


def build_anonymizer() -> AnonymizerEngine:
    """Build a Presidio AnonymizerEngine."""
    return AnonymizerEngine()


def strip_transcript_pii(
    transcript: str,
    analyzer: AnalyzerEngine,
    anonymizer: AnonymizerEngine,
    score_threshold: float = 0.4,
) -> tuple[str, dict]:
    """
    Strip PII from a single transcript and replace with typed placeholders.

    Args:
        transcript: Raw transcript text
        analyzer: Presidio AnalyzerEngine
        anonymizer: Presidio AnonymizerEngine
        score_threshold: Minimum confidence score to consider a PII match

    Returns:
        tuple: (anonymized_text, entity_counts)
    """
    if not transcript or not isinstance(transcript, str) or not transcript.strip():
        return "", {}

    text = extract_transcript_text(transcript)

    # Analyze for PII entities
    entities_to_detect = list(ENTITY_PLACEHOLDERS.keys())
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=entities_to_detect,
        score_threshold=score_threshold,
    )

    if not results:
        return text, {}

    # Build operator config: map each entity type to its placeholder
    operators = {}
    for entity_type, placeholder in ENTITY_PLACEHOLDERS.items():
        operators[entity_type] = OperatorConfig(
            "replace", {"new_value": placeholder}
        )

    # Anonymize
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )

    # Count entities found
    entity_counts = {}
    for result in results:
        entity_counts[result.entity_type] = (
            entity_counts.get(result.entity_type, 0) + 1
        )

    return anonymized.text, entity_counts


def strip_metadata_pii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redact PII columns in metadata (phone numbers).
    Replaces entire column values with placeholders.
    """
    for col in PII_METADATA_COLUMNS:
        if col in df.columns:
            # Keep a hash for grouping if needed, but remove actual number
            df[col] = df[col].apply(
                lambda x: "[PHONE]" if x and str(x).strip() else ""
            )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Strip PII from RunGopher conversation CSV exports"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for stripped CSV"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.4,
        help="PII detection confidence threshold (0.0-1.0, default: 0.4)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load conversations
    print("=" * 60)
    print("Phase 1: PII Stripping")
    print("=" * 60)
    df = load_conversations(args.input)

    # Build Presidio engines
    print("Initializing PII detection engine...")
    analyzer = build_analyzer()
    anonymizer = build_anonymizer()

    # Strip metadata PII (phone columns)
    print("Stripping metadata PII (phone numbers)...")
    df = strip_metadata_pii(df)

    # Strip transcript PII
    print(f"Stripping transcript PII from {len(df)} conversations...")
    total_entities = {}
    stripped_transcripts = []
    errors = 0

    for idx, row in df.iterrows():
        try:
            stripped, counts = strip_transcript_pii(
                row["Transcript"], analyzer, anonymizer, args.threshold
            )
            stripped_transcripts.append(stripped)

            for entity, count in counts.items():
                total_entities[entity] = total_entities.get(entity, 0) + count

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} conversations...")

        except Exception as e:
            print(f"  ⚠ Error on row {idx} (Call ID: {row.get('Call ID', '?')}): {e}")
            stripped_transcripts.append(row["Transcript"])
            errors += 1

    df["Transcript"] = stripped_transcripts

    # Save stripped CSV
    input_name = Path(args.input).stem
    output_path = output_dir / f"{input_name}_stripped.csv"
    df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("PII Stripping Complete")
    print("=" * 60)
    print(f"  Conversations processed: {len(df)}")
    print(f"  Errors: {errors}")
    print(f"  Output: {output_path}")
    print(f"\n  PII entities found & replaced:")
    for entity, count in sorted(total_entities.items(), key=lambda x: -x[1]):
        placeholder = ENTITY_PLACEHOLDERS.get(entity, f"[{entity}]")
        print(f"    {entity:25s} → {placeholder:15s} ({count:,} instances)")

    total = sum(total_entities.values())
    print(f"\n  Total PII instances replaced: {total:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
