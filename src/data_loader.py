"""Dataset loading utilities for the runnable CKD pipeline."""

from __future__ import annotations

import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
UCI_CKD_URL = (
    "https://archive.ics.uci.edu/static/public/336/chronic+kidney+disease.zip"
)


def download_uci_ckd(
    destination: str | Path = RAW_DIR / "chronic_kidney_disease",
    url: str = UCI_CKD_URL,
) -> Path:
    """Download and extract the UCI CKD file if it is not already available."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)

    existing_files = list(destination.rglob("*.arff")) + list(destination.rglob("*.csv"))
    if existing_files:
        return existing_files[0]

    archive_path = destination / "chronic_kidney_disease.zip"
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
    archive_path.write_bytes(content)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)

    extracted_files = list(destination.rglob("*.arff")) + list(destination.rglob("*.csv"))
    if not extracted_files:
        raise FileNotFoundError("Downloaded UCI CKD archive did not contain ARFF or CSV data.")
    return extracted_files[0]


def _parse_arff_text(text: str) -> pd.DataFrame:
    """Parse the simple ARFF format used by the UCI CKD dataset."""
    columns: list[str] = []
    data_lines: list[str] = []
    in_data = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        lower_line = line.lower()
        if lower_line.startswith("@attribute"):
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                columns.append(parts[1].strip("'\""))
        elif lower_line.startswith("@data"):
            in_data = True
        elif in_data:
            data_lines.append(line)

    reader = csv.reader(io.StringIO("\n".join(data_lines)), quotechar="'", skipinitialspace=True)
    rows = [[value.strip().strip("'").replace("\t", "") for value in row] for row in reader]
    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe = dataframe.replace({"?": pd.NA, "": pd.NA})
    return dataframe


def load_uci_ckd_from_path(path: str | Path) -> pd.DataFrame:
    """Load the UCI CKD dataset from a local ARFF or CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path)
    else:
        dataframe = _parse_arff_text(path.read_text(encoding="utf-8", errors="replace"))
    return dataframe


def load_uci_ckd_from_ucimlrepo() -> pd.DataFrame:
    """Load UCI CKD using the official ucimlrepo package."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "ucimlrepo is required when the direct UCI archive cannot be parsed. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    dataset = fetch_ucirepo(id=336)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    target_column = targets.columns[0]
    return pd.concat([features, targets.rename(columns={target_column: "class"})], axis=1)


def load_uci_ckd(download: bool = True, path: str | Path | None = None) -> pd.DataFrame:
    """Load the UCI CKD dataset, downloading it when needed."""
    dataset_path = Path(path) if path is not None else RAW_DIR / "chronic_kidney_disease"
    if path is None:
        cached_csv = RAW_DIR / "uci_ckd.csv"
        if cached_csv.exists():
            return pd.read_csv(cached_csv)
        try:
            dataframe = load_uci_ckd_from_ucimlrepo()
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(cached_csv, index=False)
            return dataframe
        except Exception as exc:
            print(f"Official UCI loader unavailable ({exc}). Trying direct archive download.")
    if download:
        dataset_path = download_uci_ckd(dataset_path)
    return load_uci_ckd_from_path(dataset_path)
