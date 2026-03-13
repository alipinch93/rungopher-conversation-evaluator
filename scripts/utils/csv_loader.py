"""
CSV Loader — reads RunGopher conversation export CSVs.
Handles the specific column structure and normalizes transcript format.
"""

import pandas as pd
from pathlib import Path


def load_conversations(csv_path: str) -> pd.DataFrame:
    """
    Load a RunGopher conversation CSV export.
    
    Expected columns:
        Participant ID, Assistant, Assistant Phone, Customer Phone, Direction,
        User Intent, Platform Status, Tool Status, Duration (Seconds),
        Connect Time, Transcript
    
    Returns a DataFrame with all columns preserved.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(
        csv_path,
        dtype={"Participant ID": str, "Duration (Seconds)": float},
        keep_default_na=False,
    )

    # Validate expected columns exist
    expected = {"Participant ID", "Transcript"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Drop rows with empty transcripts
    df = df[df["Transcript"].str.strip().astype(bool)].copy()
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df)} conversations from {path.name}")
    return df


def extract_transcript_text(transcript: str) -> str:
    """
    Normalize a transcript string for processing.
    Preserves AI/User speaker labels.
    """
    if not transcript or not isinstance(transcript, str):
        return ""
    
    # Clean up common formatting issues
    text = transcript.strip()
    # Normalize line breaks
    text = text.replace("\\n", "\n")
    return text
