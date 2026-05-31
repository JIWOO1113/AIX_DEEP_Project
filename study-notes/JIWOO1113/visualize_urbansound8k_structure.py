"""
Visualize the original UrbanSound8K dataset structure before preprocessing.

This script does not resample, pad, truncate, or normalize audio. It reads:
1. UrbanSound8K/metadata/UrbanSound8K.csv
2. WAV header information with soundfile.info()
3. Optional raw waveform values for audio-quality statistics

Outputs are written under study-notes/JIWOO1113/image/urbansound8k_structure/.

Run:
    python study-notes/JIWOO1113/visualize_urbansound8k_structure.py

Required packages:
    pip install pandas matplotlib numpy soundfile
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

MPL_CACHE_DIR = Path("/tmp") / "matplotlib-urbansound8k"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = SCRIPT_DIR.parents[1] / "UrbanSound8K"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "image" / "urbansound8k_structure"
DEFAULT_PROFILE_CSV = SCRIPT_DIR / "UrbanSound8K_audio_profile.csv"
DEFAULT_QUALITY_CSV = SCRIPT_DIR / "UrbanSound8K_audio_quality.csv"
FILENAME_PATTERN = r"^(?P<fsID>\d+)-(?P<classID>\d+)-(?P<occurrenceID>\d+)-(?P<sliceID>\d+)\.wav$"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize UrbanSound8K metadata and original WAV file properties."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the UrbanSound8K directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figure files will be saved.",
    )
    parser.add_argument(
        "--profile-csv",
        type=Path,
        default=DEFAULT_PROFILE_CSV,
        help="CSV path for per-file audio header summary.",
    )
    parser.add_argument(
        "--reuse-profile",
        action="store_true",
        help="Reuse profile CSV if it already exists.",
    )
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=DEFAULT_QUALITY_CSV,
        help="CSV path for per-file RMS, peak, clipping, and silence statistics.",
    )
    parser.add_argument(
        "--reuse-quality",
        action="store_true",
        help="Reuse quality CSV if it already exists.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests.",
    )
    parser.add_argument(
        "--skip-waveforms",
        action="store_true",
        help="Skip waveform example figure.",
    )
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="Skip RMS, peak, clipping, and silence statistics.",
    )
    parser.add_argument(
        "--duration-tolerance-sec",
        type=float,
        default=0.01,
        help="Tolerance for duration mismatch reports.",
    )
    parser.add_argument(
        "--clipping-threshold",
        type=float,
        default=0.999,
        help="Absolute amplitude threshold used to flag clipping.",
    )
    parser.add_argument(
        "--silence-dbfs-threshold",
        type=float,
        default=-60.0,
        help="Frame RMS dBFS threshold used to estimate silence.",
    )
    parser.add_argument(
        "--silence-frame-sec",
        type=float,
        default=0.05,
        help="Frame size in seconds for silence estimation.",
    )
    return parser.parse_args()


def load_metadata(dataset_root: Path) -> pd.DataFrame:
    metadata_path = dataset_root / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    metadata["metadata_duration_sec"] = metadata["end"] - metadata["start"]
    return metadata


def class_order(metadata: pd.DataFrame) -> list[str]:
    return (
        metadata.sort_values("classID")
        .drop_duplicates("classID")["class"]
        .astype(str)
        .tolist()
    )


def relative_to_repo(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root.parent))
    except ValueError:
        return str(path)


def build_audio_profile(
    metadata: pd.DataFrame, dataset_root: Path, max_files: int | None = None
) -> pd.DataFrame:
    rows = []
    selected = metadata.head(max_files).copy() if max_files else metadata.copy()
    total = len(selected)

    for number, (_, row) in enumerate(selected.iterrows(), start=1):
        audio_path = (
            dataset_root
            / "audio"
            / f"fold{int(row['fold'])}"
            / str(row["slice_file_name"])
        )

        profile = row.to_dict()
        profile["relative_audio_path"] = relative_to_repo(audio_path, dataset_root)
        profile["exists"] = audio_path.exists()
        profile["sample_rate"] = np.nan
        profile["channels"] = np.nan
        profile["frames"] = np.nan
        profile["actual_duration_sec"] = np.nan
        profile["format"] = pd.NA
        profile["subtype"] = pd.NA
        profile["read_error"] = pd.NA

        if audio_path.exists():
            try:
                info = sf.info(audio_path)
                profile["sample_rate"] = int(info.samplerate)
                profile["channels"] = int(info.channels)
                profile["frames"] = int(info.frames)
                profile["actual_duration_sec"] = float(info.duration)
                profile["format"] = str(info.format)
                profile["subtype"] = str(info.subtype)
            except Exception as exc:  # noqa: BLE001
                profile["read_error"] = repr(exc)

        rows.append(profile)

        if number % 1000 == 0 or number == total:
            print(f"Scanned {number}/{total} audio headers")

    profile_df = pd.DataFrame(rows)
    profile_df["duration_diff_sec"] = (
        profile_df["actual_duration_sec"] - profile_df["metadata_duration_sec"]
    )
    return profile_df


def audio_path_from_row(row: pd.Series, dataset_root: Path) -> Path:
    return (
        dataset_root
        / "audio"
        / f"fold{int(row['fold'])}"
        / str(row["slice_file_name"])
    )


def save_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    return path


def expected_audio_paths(metadata: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    expected = metadata.copy()
    expected["expected_audio_path"] = expected.apply(
        lambda row: relative_to_repo(audio_path_from_row(row, dataset_root), dataset_root),
        axis=1,
    )
    return expected


def build_fsid_cross_fold_report(metadata: pd.DataFrame) -> pd.DataFrame:
    fold_lists = metadata.groupby("fsID")["fold"].apply(
        lambda values: ",".join(str(int(value)) for value in sorted(values.unique()))
    )
    file_counts = metadata.groupby("fsID").size()
    leaky_fsids = fold_lists[fold_lists.str.contains(",")].index

    if len(leaky_fsids) == 0:
        return pd.DataFrame(
            columns=[
                "fsID",
                "folds_for_fsID",
                "files_for_fsID",
                "slice_file_name",
                "fold",
                "classID",
                "class",
                "start",
                "end",
                "salience",
            ]
        )

    report = metadata[metadata["fsID"].isin(leaky_fsids)].copy()
    report["folds_for_fsID"] = report["fsID"].map(fold_lists)
    report["files_for_fsID"] = report["fsID"].map(file_counts)
    columns = [
        "fsID",
        "folds_for_fsID",
        "files_for_fsID",
        "slice_file_name",
        "fold",
        "classID",
        "class",
        "start",
        "end",
        "salience",
    ]
    return report[columns].sort_values(["fsID", "fold", "slice_file_name"])


def build_filename_parse_report(metadata: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(FILENAME_PATTERN)
    rows = []

    for row_index, row in metadata.reset_index().iterrows():
        match = pattern.match(str(row["slice_file_name"]))
        if not match:
            rows.append(
                {
                    "row_index": int(row["index"]),
                    "slice_file_name": row["slice_file_name"],
                    "issue": "pattern_mismatch",
                    "csv_fsID": int(row["fsID"]),
                    "parsed_fsID": pd.NA,
                    "csv_classID": int(row["classID"]),
                    "parsed_classID": pd.NA,
                    "parsed_occurrenceID": pd.NA,
                    "parsed_sliceID": pd.NA,
                }
            )
            continue

        parsed = {key: int(value) for key, value in match.groupdict().items()}
        issues = []
        if parsed["fsID"] != int(row["fsID"]):
            issues.append("fsID_mismatch")
        if parsed["classID"] != int(row["classID"]):
            issues.append("classID_mismatch")

        if issues:
            rows.append(
                {
                    "row_index": int(row["index"]),
                    "slice_file_name": row["slice_file_name"],
                    "issue": "+".join(issues),
                    "csv_fsID": int(row["fsID"]),
                    "parsed_fsID": parsed["fsID"],
                    "csv_classID": int(row["classID"]),
                    "parsed_classID": parsed["classID"],
                    "parsed_occurrenceID": parsed["occurrenceID"],
                    "parsed_sliceID": parsed["sliceID"],
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "row_index",
            "slice_file_name",
            "issue",
            "csv_fsID",
            "parsed_fsID",
            "csv_classID",
            "parsed_classID",
            "parsed_occurrenceID",
            "parsed_sliceID",
        ],
    )


def build_file_inventory_reports(
    metadata: pd.DataFrame, dataset_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = expected_audio_paths(metadata, dataset_root)
    expected_paths = set(expected["expected_audio_path"])
    actual_paths = {
        relative_to_repo(path, dataset_root)
        for path in sorted((dataset_root / "audio").glob("fold*/*.wav"))
    }

    missing = expected[~expected["expected_audio_path"].isin(actual_paths)].copy()
    missing = missing[
        [
            "expected_audio_path",
            "slice_file_name",
            "fsID",
            "fold",
            "classID",
            "class",
        ]
    ]

    extra_rows = []
    for path_string in sorted(actual_paths - expected_paths):
        path = Path(path_string)
        extra_rows.append(
            {
                "relative_audio_path": path_string,
                "fold_folder": path.parent.name,
                "slice_file_name": path.name,
            }
        )
    extra = pd.DataFrame(
        extra_rows,
        columns=["relative_audio_path", "fold_folder", "slice_file_name"],
    )
    return missing, extra


def write_class_fold_tables(
    metadata: pd.DataFrame, ordered_classes: list[str], output_dir: Path
) -> list[Path]:
    table_dir = output_dir / "tables"
    count_table = pd.crosstab(metadata["class"], metadata["fold"]).reindex(ordered_classes)
    count_table = count_table.reindex(columns=range(1, 11), fill_value=0)
    count_table.insert(0, "class", count_table.index)
    count_table = count_table.reset_index(drop=True)

    ratio_values = count_table.drop(columns=["class"]).div(
        count_table.drop(columns=["class"]).sum(axis=1), axis=0
    )
    ratio_table = ratio_values.round(4)
    ratio_table.insert(0, "class", count_table["class"])

    return [
        save_csv(count_table, table_dir / "class_by_fold_count.csv"),
        save_csv(ratio_table, table_dir / "class_by_fold_ratio.csv"),
    ]


def write_validation_reports(
    metadata: pd.DataFrame,
    profile: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
    duration_tolerance_sec: float,
) -> tuple[list[Path], dict[str, int]]:
    validation_dir = output_dir / "validation"

    fsid_cross_fold = build_fsid_cross_fold_report(metadata)
    filename_parse_errors = build_filename_parse_report(metadata)
    missing_wavs, extra_wavs = build_file_inventory_reports(metadata, dataset_root)

    read_errors = profile[profile["read_error"].notna()].copy()
    read_error_columns = [
        "relative_audio_path",
        "slice_file_name",
        "fsID",
        "fold",
        "classID",
        "class",
        "read_error",
    ]
    read_errors = read_errors[
        [column for column in read_error_columns if column in read_errors.columns]
    ]

    duration_mismatch = profile[
        profile["duration_diff_sec"].abs() > duration_tolerance_sec
    ].copy()
    duration_columns = [
        "relative_audio_path",
        "slice_file_name",
        "fsID",
        "fold",
        "classID",
        "class",
        "metadata_duration_sec",
        "actual_duration_sec",
        "duration_diff_sec",
        "sample_rate",
        "channels",
        "frames",
    ]
    duration_mismatch = duration_mismatch[
        [column for column in duration_columns if column in duration_mismatch.columns]
    ].sort_values("duration_diff_sec", key=lambda values: values.abs(), ascending=False)

    paths = [
        save_csv(fsid_cross_fold, validation_dir / "fsid_cross_fold_leakage.csv"),
        save_csv(filename_parse_errors, validation_dir / "filename_parse_errors.csv"),
        save_csv(missing_wavs, validation_dir / "missing_wavs.csv"),
        save_csv(extra_wavs, validation_dir / "extra_wavs.csv"),
        save_csv(read_errors, validation_dir / "read_errors.csv"),
        save_csv(duration_mismatch, validation_dir / "duration_mismatch.csv"),
    ]

    summary = {
        "fsids_crossing_folds": int(fsid_cross_fold["fsID"].nunique())
        if not fsid_cross_fold.empty
        else 0,
        "files_from_cross_fold_fsids": int(len(fsid_cross_fold)),
        "filename_parse_errors": int(len(filename_parse_errors)),
        "missing_wavs": int(len(missing_wavs)),
        "extra_wavs": int(len(extra_wavs)),
        "read_errors": int(len(read_errors)),
        "duration_mismatch": int(len(duration_mismatch)),
    }
    return paths, summary


def dbfs(value: float, eps: float = 1e-12) -> float:
    return float(20.0 * np.log10(max(value, eps)))


def frame_rms(samples: np.ndarray, frame_length: int) -> np.ndarray:
    if samples.size == 0:
        return np.array([], dtype=np.float32)

    mono = samples.mean(axis=1)
    frame_length = max(1, frame_length)
    frame_count = int(np.ceil(len(mono) / frame_length))
    padded_length = frame_count * frame_length
    if padded_length > len(mono):
        mono = np.pad(mono, (0, padded_length - len(mono)))

    frames = mono.reshape(frame_count, frame_length)
    return np.sqrt(np.mean(np.square(frames), axis=1))


def build_audio_quality_stats(
    profile: pd.DataFrame,
    dataset_root: Path,
    clipping_threshold: float,
    silence_dbfs_threshold: float,
    silence_frame_sec: float,
) -> pd.DataFrame:
    rows = []
    readable = profile[profile["exists"] & profile["read_error"].isna()].copy()
    total = len(readable)
    silence_threshold = 10.0 ** (silence_dbfs_threshold / 20.0)

    for number, (_, row) in enumerate(readable.iterrows(), start=1):
        audio_path = dataset_root.parent / str(row["relative_audio_path"])
        quality = {
            "relative_audio_path": row["relative_audio_path"],
            "slice_file_name": row["slice_file_name"],
            "fsID": int(row["fsID"]),
            "fold": int(row["fold"]),
            "classID": int(row["classID"]),
            "class": row["class"],
            "sample_rate": int(row["sample_rate"]),
            "channels": int(row["channels"]),
            "frames": int(row["frames"]),
            "actual_duration_sec": float(row["actual_duration_sec"]),
            "rms": np.nan,
            "rms_dbfs": np.nan,
            "peak_abs": np.nan,
            "peak_dbfs": np.nan,
            "clipping_sample_count": np.nan,
            "clipping_fraction": np.nan,
            "silent_frame_count": np.nan,
            "total_frame_count": np.nan,
            "silent_frame_fraction": np.nan,
            "quality_read_error": pd.NA,
        }

        try:
            samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            abs_samples = np.abs(samples)
            peak = float(abs_samples.max()) if abs_samples.size else 0.0
            rms_value = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
            clipping_mask = abs_samples >= clipping_threshold
            frame_length = max(1, int(round(sample_rate * silence_frame_sec)))
            rms_by_frame = frame_rms(samples, frame_length)
            silent_mask = rms_by_frame <= silence_threshold

            quality["sample_rate"] = int(sample_rate)
            quality["channels"] = int(samples.shape[1])
            quality["frames"] = int(samples.shape[0])
            quality["actual_duration_sec"] = float(samples.shape[0] / sample_rate)
            quality["rms"] = rms_value
            quality["rms_dbfs"] = dbfs(rms_value)
            quality["peak_abs"] = peak
            quality["peak_dbfs"] = dbfs(peak)
            quality["clipping_sample_count"] = int(clipping_mask.sum())
            quality["clipping_fraction"] = (
                float(clipping_mask.sum() / abs_samples.size) if abs_samples.size else 0.0
            )
            quality["silent_frame_count"] = int(silent_mask.sum())
            quality["total_frame_count"] = int(len(rms_by_frame))
            quality["silent_frame_fraction"] = (
                float(silent_mask.sum() / len(rms_by_frame))
                if len(rms_by_frame)
                else 0.0
            )
        except Exception as exc:  # noqa: BLE001
            quality["quality_read_error"] = repr(exc)

        rows.append(quality)

        if number % 1000 == 0 or number == total:
            print(f"Scanned {number}/{total} audio files for quality stats")

    return pd.DataFrame(rows)


def write_audio_quality_reports(
    quality: pd.DataFrame, output_dir: Path
) -> tuple[list[Path], dict[str, int]]:
    quality_dir = output_dir / "audio_quality"
    readable = quality[quality["quality_read_error"].isna()].copy()
    suspicious_clipping = readable[readable["clipping_sample_count"] > 0].copy()
    suspicious_silence = readable[readable["silent_frame_fraction"] >= 0.95].copy()
    very_quiet = readable[readable["rms_dbfs"] <= -50.0].copy()

    paths = [
        save_csv(
            suspicious_clipping.sort_values("clipping_fraction", ascending=False),
            quality_dir / "suspicious_clipping.csv",
        ),
        save_csv(
            suspicious_silence.sort_values("silent_frame_fraction", ascending=False),
            quality_dir / "suspicious_silence.csv",
        ),
        save_csv(
            very_quiet.sort_values("rms_dbfs"),
            quality_dir / "very_quiet_files.csv",
        ),
    ]
    summary = {
        "quality_rows": int(len(quality)),
        "quality_read_errors": int(quality["quality_read_error"].notna().sum()),
        "files_with_clipping": int(len(suspicious_clipping)),
        "mostly_silent_files": int(len(suspicious_silence)),
        "very_quiet_files": int(len(very_quiet)),
    }
    return paths, summary


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, values: list[int] | np.ndarray) -> None:
    ymax = max(values) if len(values) else 0
    offset = ymax * 0.015 if ymax else 1
    for index, value in enumerate(values):
        ax.text(index, value + offset, str(int(value)), ha="center", va="bottom", fontsize=8)


def plot_metadata_overview(
    metadata: pd.DataFrame, ordered_classes: list[str], output_dir: Path
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("UrbanSound8K metadata overview", fontsize=16, fontweight="bold")

    class_counts = metadata["class"].value_counts().reindex(ordered_classes)
    axes[0, 0].barh(class_counts.index, class_counts.values, color="#4C78A8")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_title("Files per class")
    axes[0, 0].set_xlabel("count")

    fold_counts = metadata["fold"].value_counts().sort_index()
    axes[0, 1].bar(fold_counts.index.astype(str), fold_counts.values, color="#F58518")
    axes[0, 1].set_title("Files per fold")
    axes[0, 1].set_xlabel("fold")
    axes[0, 1].set_ylabel("count")
    annotate_bars(axes[0, 1], fold_counts.values)

    salience_counts = metadata["salience"].value_counts().sort_index()
    salience_labels = [
        "1 foreground" if value == 1 else "2 background"
        for value in salience_counts.index
    ]
    axes[1, 0].bar(salience_labels, salience_counts.values, color="#54A24B")
    axes[1, 0].set_title("Salience distribution")
    axes[1, 0].set_ylabel("count")
    annotate_bars(axes[1, 0], salience_counts.values)

    slices_per_recording = metadata.groupby("fsID").size()
    axes[1, 1].hist(slices_per_recording, bins=30, color="#B279A2", edgecolor="white")
    axes[1, 1].set_title("Slices per original Freesound recording")
    axes[1, 1].set_xlabel("number of slices from one fsID")
    axes[1, 1].set_ylabel("number of fsIDs")

    output_path = output_dir / "01_metadata_overview.png"
    save_figure(fig, output_path)
    return output_path


def plot_class_by_fold_heatmap(
    metadata: pd.DataFrame, ordered_classes: list[str], output_dir: Path
) -> Path:
    matrix = pd.crosstab(metadata["class"], metadata["fold"]).reindex(ordered_classes)
    matrix = matrix.reindex(columns=range(1, 11), fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 7))
    image = ax.imshow(matrix.values, cmap="YlGnBu", aspect="auto")
    ax.set_title("Class x fold count matrix")
    ax.set_xlabel("fold")
    ax.set_ylabel("class")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(matrix.iloc[row_index, col_index])
            text_color = "white" if value > matrix.values.max() * 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                value,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    fig.colorbar(image, ax=ax, label="count")
    output_path = output_dir / "02_class_by_fold_heatmap.png"
    save_figure(fig, output_path)
    return output_path


def plot_duration_distribution(
    profile: pd.DataFrame, ordered_classes: list[str], output_dir: Path
) -> Path:
    valid = profile[profile["exists"] & profile["actual_duration_sec"].notna()].copy()
    valid["actual_duration_sec"] = pd.to_numeric(valid["actual_duration_sec"])
    valid["metadata_duration_sec"] = pd.to_numeric(valid["metadata_duration_sec"])
    valid["duration_diff_sec"] = pd.to_numeric(valid["duration_diff_sec"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Original clip duration checks", fontsize=16, fontweight="bold")

    bins = np.linspace(0, max(4.1, valid["actual_duration_sec"].max()), 45)
    axes[0, 0].hist(
        valid["actual_duration_sec"],
        bins=bins,
        color="#4C78A8",
        edgecolor="white",
    )
    axes[0, 0].set_title("Actual WAV duration")
    axes[0, 0].set_xlabel("seconds")
    axes[0, 0].set_ylabel("files")

    grouped = [
        valid.loc[valid["class"] == label, "actual_duration_sec"].values
        for label in ordered_classes
    ]
    try:
        axes[0, 1].boxplot(
            grouped, tick_labels=ordered_classes, vert=True, showfliers=False
        )
    except TypeError:
        axes[0, 1].boxplot(
            grouped, labels=ordered_classes, vert=True, showfliers=False
        )
    axes[0, 1].set_title("Actual duration by class")
    axes[0, 1].set_ylabel("seconds")
    axes[0, 1].tick_params(axis="x", rotation=35)

    max_duration = max(
        valid["metadata_duration_sec"].max(),
        valid["actual_duration_sec"].max(),
    )
    axes[1, 0].scatter(
        valid["metadata_duration_sec"],
        valid["actual_duration_sec"],
        s=8,
        alpha=0.35,
        color="#F58518",
    )
    axes[1, 0].plot([0, max_duration], [0, max_duration], color="#444444", linewidth=1)
    axes[1, 0].set_title("Metadata duration vs actual WAV duration")
    axes[1, 0].set_xlabel("metadata end-start seconds")
    axes[1, 0].set_ylabel("actual WAV seconds")

    axes[1, 1].hist(
        valid["duration_diff_sec"],
        bins=50,
        color="#54A24B",
        edgecolor="white",
    )
    axes[1, 1].set_title("Actual - metadata duration")
    axes[1, 1].set_xlabel("seconds")
    axes[1, 1].set_ylabel("files")

    output_path = output_dir / "03_duration_distribution.png"
    save_figure(fig, output_path)
    return output_path


def value_counts_bar(
    ax: plt.Axes,
    series: pd.Series,
    title: str,
    xlabel: str,
    color: str,
    rotate: bool = False,
) -> None:
    counts = series.value_counts(dropna=False).sort_index()
    labels = [str(label) for label in counts.index]
    ax.bar(labels, counts.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if rotate:
        ax.tick_params(axis="x", rotation=35)
    annotate_bars(ax, counts.values)


def plot_audio_format_distribution(profile: pd.DataFrame, output_dir: Path) -> Path:
    valid = profile[profile["exists"] & profile["read_error"].isna()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Original WAV format distribution", fontsize=16, fontweight="bold")

    value_counts_bar(
        axes[0, 0],
        valid["sample_rate"].astype("Int64"),
        "Sample rate distribution",
        "Hz",
        "#4C78A8",
    )
    value_counts_bar(
        axes[0, 1],
        valid["channels"].astype("Int64"),
        "Channel count distribution",
        "channels",
        "#F58518",
    )
    value_counts_bar(
        axes[1, 0],
        valid["subtype"],
        "WAV subtype distribution",
        "subtype",
        "#54A24B",
        rotate=True,
    )
    value_counts_bar(
        axes[1, 1],
        valid["format"],
        "Container format distribution",
        "format",
        "#B279A2",
        rotate=True,
    )

    output_path = output_dir / "04_audio_format_distribution.png"
    save_figure(fig, output_path)
    return output_path


def plot_audio_quality_distribution(quality: pd.DataFrame, output_dir: Path) -> Path:
    valid = quality[quality["quality_read_error"].isna()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Original audio quality statistics", fontsize=16, fontweight="bold")

    axes[0, 0].hist(valid["rms_dbfs"], bins=50, color="#4C78A8", edgecolor="white")
    axes[0, 0].set_title("RMS level")
    axes[0, 0].set_xlabel("dBFS")
    axes[0, 0].set_ylabel("files")

    axes[0, 1].hist(valid["peak_dbfs"], bins=50, color="#F58518", edgecolor="white")
    axes[0, 1].set_title("Peak level")
    axes[0, 1].set_xlabel("dBFS")
    axes[0, 1].set_ylabel("files")

    axes[1, 0].hist(
        valid["clipping_fraction"],
        bins=50,
        color="#E45756",
        edgecolor="white",
    )
    axes[1, 0].set_title("Clipping sample fraction")
    axes[1, 0].set_xlabel("fraction")
    axes[1, 0].set_ylabel("files")

    axes[1, 1].hist(
        valid["silent_frame_fraction"],
        bins=50,
        color="#54A24B",
        edgecolor="white",
    )
    axes[1, 1].set_title("Silent frame fraction")
    axes[1, 1].set_xlabel("fraction")
    axes[1, 1].set_ylabel("files")

    output_path = output_dir / "06_audio_quality_distribution.png"
    save_figure(fig, output_path)
    return output_path


def select_waveform_examples(
    profile: pd.DataFrame, ordered_classes: list[str]
) -> pd.DataFrame:
    valid = profile[profile["exists"] & profile["read_error"].isna()].copy()
    valid = valid[valid["actual_duration_sec"].notna()]
    examples = []

    for label in ordered_classes:
        class_rows = valid[valid["class"] == label].copy()
        if class_rows.empty:
            continue
        median_duration = class_rows["actual_duration_sec"].median()
        class_rows["duration_distance"] = (
            class_rows["actual_duration_sec"] - median_duration
        ).abs()
        examples.append(class_rows.sort_values("duration_distance").iloc[0])

    return pd.DataFrame(examples)


def plot_waveform_examples(
    examples: pd.DataFrame, dataset_root: Path, output_dir: Path
) -> Path:
    fig, axes = plt.subplots(5, 2, figsize=(16, 12), sharex=False)
    axes_flat = axes.ravel()
    fig.suptitle("One original waveform example per class", fontsize=16, fontweight="bold")

    for ax, (_, row) in zip(axes_flat, examples.iterrows()):
        audio_path = dataset_root.parent / str(row["relative_audio_path"])
        samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        time_axis = np.arange(samples.shape[0]) / float(sample_rate)

        for channel_index in range(samples.shape[1]):
            ax.plot(
                time_axis,
                samples[:, channel_index],
                linewidth=0.6,
                alpha=0.8,
                label=f"ch{channel_index + 1}",
            )

        title = (
            f"{row['class']} | {sample_rate} Hz | "
            f"{samples.shape[1]} ch | {row['actual_duration_sec']:.2f}s"
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("seconds")
        ax.set_ylabel("amplitude")
        ax.grid(True, linewidth=0.3, alpha=0.4)
        if samples.shape[1] > 1:
            ax.legend(loc="upper right", fontsize=7)

    for ax in axes_flat[len(examples) :]:
        ax.axis("off")

    output_path = output_dir / "05_waveform_examples_by_class.png"
    save_figure(fig, output_path)
    return output_path


def write_summary(
    profile: pd.DataFrame,
    output_dir: Path,
    validation_summary: dict[str, int] | None = None,
    quality_summary: dict[str, int] | None = None,
) -> Path:
    summary_path = output_dir / "00_dataset_structure_summary.txt"
    valid = profile[profile["exists"] & profile["read_error"].isna()].copy()
    missing_count = int((~profile["exists"]).sum())
    error_count = int(profile["read_error"].notna().sum())

    lines = [
        "UrbanSound8K original structure summary",
        "=" * 42,
        f"Rows in metadata: {len(profile)}",
        f"Existing audio files: {len(valid)}",
        f"Missing audio files: {missing_count}",
        f"Files with read errors: {error_count}",
        "",
        "Class counts:",
        profile["class"].value_counts().sort_index().to_string(),
        "",
        "Fold counts:",
        profile["fold"].value_counts().sort_index().to_string(),
        "",
        "Sample rate counts:",
        valid["sample_rate"].astype("Int64").value_counts().sort_index().to_string(),
        "",
        "Channel counts:",
        valid["channels"].astype("Int64").value_counts().sort_index().to_string(),
        "",
        "Subtype counts:",
        valid["subtype"].value_counts().sort_index().to_string(),
        "",
        "Actual duration seconds:",
        valid["actual_duration_sec"].describe().to_string(),
    ]

    if validation_summary:
        lines.extend(
            [
                "",
                "Validation checks:",
                pd.Series(validation_summary).to_string(),
            ]
        )

    if quality_summary:
        lines.extend(
            [
                "",
                "Audio quality checks:",
                pd.Series(quality_summary).to_string(),
            ]
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    profile_csv = args.profile_csv.resolve()
    quality_csv = args.quality_csv.resolve()

    metadata = load_metadata(dataset_root)
    ordered_classes = class_order(metadata)

    if args.reuse_profile and profile_csv.exists():
        print(f"Reusing profile CSV: {profile_csv}")
        profile = pd.read_csv(profile_csv)
    else:
        profile = build_audio_profile(metadata, dataset_root, args.max_files)
        profile_csv.parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(profile_csv, index=False)
        print(f"Wrote profile CSV: {profile_csv}")

    table_paths = write_class_fold_tables(metadata, ordered_classes, output_dir)
    validation_paths, validation_summary = write_validation_reports(
        metadata,
        profile,
        dataset_root,
        output_dir,
        args.duration_tolerance_sec,
    )

    quality = None
    quality_paths: list[Path] = []
    quality_summary: dict[str, int] = {}
    if not args.skip_quality:
        if args.reuse_quality and quality_csv.exists():
            print(f"Reusing quality CSV: {quality_csv}")
            quality = pd.read_csv(quality_csv)
        else:
            quality = build_audio_quality_stats(
                profile,
                dataset_root,
                args.clipping_threshold,
                args.silence_dbfs_threshold,
                args.silence_frame_sec,
            )
            quality_csv.parent.mkdir(parents=True, exist_ok=True)
            quality.to_csv(quality_csv, index=False)
            print(f"Wrote quality CSV: {quality_csv}")

        quality_paths, quality_summary = write_audio_quality_reports(
            quality, output_dir
        )

    created_files = [
        write_summary(profile, output_dir, validation_summary, quality_summary),
        *table_paths,
        *validation_paths,
        plot_metadata_overview(metadata, ordered_classes, output_dir),
        plot_class_by_fold_heatmap(metadata, ordered_classes, output_dir),
        plot_duration_distribution(profile, ordered_classes, output_dir),
        plot_audio_format_distribution(profile, output_dir),
    ]

    if quality is not None:
        created_files.extend(quality_paths)
        created_files.append(plot_audio_quality_distribution(quality, output_dir))

    if not args.skip_waveforms:
        examples = select_waveform_examples(profile, ordered_classes)
        created_files.append(plot_waveform_examples(examples, dataset_root, output_dir))

    print("\nCreated:")
    print(profile_csv)
    if quality is not None:
        print(quality_csv)
    for path in created_files:
        print(path)


if __name__ == "__main__":
    main()
