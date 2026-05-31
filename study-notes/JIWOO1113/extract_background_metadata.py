"""
Extract background-file metadata rows from UrbanSound8K.

UrbanSound8K metadata uses:
    salience = 1 for foreground sounds
    salience = 2 for background sounds

Run:
    python study-notes/JIWOO1113/extract_background_metadata.py

Default input:
    UrbanSound8K/metadata/UrbanSound8K.csv

Default output:
    study-notes/JIWOO1113/UrbanSound8K_background_metadata.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "UrbanSound8K"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "UrbanSound8K_background_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract only background-file rows from UrbanSound8K metadata."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the UrbanSound8K directory.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional direct path to UrbanSound8K.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV path where extracted background metadata will be saved.",
    )
    return parser.parse_args()


def metadata_path_from_args(dataset_root: Path, metadata_csv: Path | None) -> Path:
    if metadata_csv is not None:
        return metadata_csv

    candidates = [
        dataset_root / "metadata" / "UrbanSound8K.csv",
        dataset_root / "metadate" / "UrbanSound8K.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "UrbanSound8K metadata CSV was not found. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def extract_background_metadata(metadata_csv: Path, dataset_root: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)

    required_columns = {"slice_file_name", "start", "end", "salience", "fold"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required metadata column(s): {missing}")

    background = metadata.loc[metadata["salience"] == 2].copy()
    background["duration_sec"] = background["end"] - background["start"]
    background["relative_audio_path"] = background.apply(
        lambda row: str(
            Path("UrbanSound8K")
            / "audio"
            / f"fold{int(row['fold'])}"
            / str(row["slice_file_name"])
        ),
        axis=1,
    )
    background["audio_exists"] = background["relative_audio_path"].map(
        lambda relative_path: (dataset_root.parent / relative_path).exists()
    )

    return background


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    metadata_csv = metadata_path_from_args(dataset_root, args.metadata_csv)

    background = extract_background_metadata(metadata_csv, dataset_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    background.to_csv(args.output_csv, index=False)

    print(f"Input metadata: {metadata_csv}")
    print(f"Background rows: {len(background)}")
    print(f"Output CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
