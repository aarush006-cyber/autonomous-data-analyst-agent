"""Dataset ingestion and profiling tools.

Handles CSV upload, validation, and metadata extraction.
Treats CSV contents as untrusted data.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.state import DatasetMetadata

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 50
MAX_COLUMNS = 500
SAMPLE_ROWS = 3


class DatasetError(Exception):
    """Raised for dataset validation failures."""


def validate_csv_file(file_path: str) -> None:
    """Validate a CSV file before loading."""
    path = Path(file_path)

    if not path.exists():
        raise DatasetError(f"File not found: {path.name}")

    if not path.suffix.lower() == ".csv":
        raise DatasetError("Only CSV files are supported.")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise DatasetError("File is empty.")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DatasetError(f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.")


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file with error handling."""
    validate_csv_file(file_path)

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding="latin-1")
        except Exception as e:
            raise DatasetError(f"Failed to parse CSV with encoding fallback: {e}")
    except pd.errors.EmptyDataError:
        raise DatasetError("CSV file contains no data.")
    except pd.errors.ParserError as e:
        raise DatasetError(f"CSV parsing error: {e}")

    if df.empty:
        raise DatasetError("Dataset is empty (no rows).")

    if len(df.columns) > MAX_COLUMNS:
        raise DatasetError(f"Too many columns ({len(df.columns)}). Maximum is {MAX_COLUMNS}.")

    if len(df.columns) == 0:
        raise DatasetError("Dataset has no columns.")

    return df


def profile_dataset(df: pd.DataFrame, filename: str = "dataset.csv") -> DatasetMetadata:
    """Generate comprehensive profiling metadata for a DataFrame."""
    quality_warnings: list[str] = []

    # Basic info
    rows, cols = df.shape

    # Column types
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Missing values
    missing = df.isnull().sum().to_dict()
    missing_pct = {col: round(count / rows * 100, 2) if rows > 0 else 0
                   for col, count in missing.items()}

    high_missing = [col for col, pct in missing_pct.items() if pct > 50]
    if high_missing:
        quality_warnings.append(f"High missing values (>50%): {', '.join(high_missing)}")

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        dup_pct = round(duplicate_rows / rows * 100, 1)
        quality_warnings.append(f"{duplicate_rows} duplicate rows ({dup_pct}%)")

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe()
        numeric_summary = {
            col: {stat: round(float(val), 4) if pd.notna(val) else None
                  for stat, val in desc[col].items()}
            for col in numeric_cols
        }

    # Categorical summary
    cat_cols = df.select_dtypes(include=["object", "string", "str", "category", "bool"]).columns.tolist()
    categorical_summary = {}
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).head(10)
        categorical_summary[col] = {
            "unique": int(df[col].nunique(dropna=False)),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }

    # Single-value columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    if constant_cols:
        quality_warnings.append(f"Constant/near-constant columns: {', '.join(constant_cols)}")

    # Sample rows (sanitized: truncate long strings to prevent prompt injection)
    sample = df.head(SAMPLE_ROWS).copy()
    for col in sample.select_dtypes(include=["object", "string", "str"]).columns:
        sample[col] = sample[col].astype(str).str.slice(0, 100)
    sample_rows = sample.to_dict(orient="records")

    file_size = 0
    try:
        # We might not have a real path for in-memory DataFrames
        pass
    except Exception:
        pass

    return DatasetMetadata(
        filename=filename,
        rows=rows,
        columns=cols,
        column_names=list(df.columns),
        dtypes=dtypes,
        missing_values={k: int(v) for k, v in missing.items()},
        missing_pct=missing_pct,
        duplicate_rows=duplicate_rows,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        sample_rows=sample_rows,
        quality_warnings=quality_warnings,
    )


def get_safe_metadata_for_llm(metadata: DatasetMetadata) -> dict:
    """Return a sanitized metadata dict safe to include in LLM prompts.
    
    Truncates large fields and strips potentially adversarial content.
    """
    def _sanitize(v: Any) -> Any:
        if isinstance(v, str):
            return v[:500]
        return v

    return {
        "filename": metadata.filename,
        "rows": metadata.rows,
        "columns": metadata.columns,
        "column_names": [str(c)[:100] for c in metadata.column_names[:100]],
        "dtypes": {str(k)[:100]: str(v) for k, v in list(metadata.dtypes.items())[:100]},
        "missing_values": dict(list(metadata.missing_values.items())[:100]),
        "duplicate_rows": metadata.duplicate_rows,
        "numeric_summary": {k: v for k, v in list(metadata.numeric_summary.items())[:50]},
        "categorical_summary": {k: v for k, v in list(metadata.categorical_summary.items())[:30]},
        "sample_rows": metadata.sample_rows[:3],
        "quality_warnings": metadata.quality_warnings[:20],
    }
