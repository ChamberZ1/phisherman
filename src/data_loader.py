import pandas as pd
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from numbers import Real


# Schema structure
STANDARD_COLUMNS = [
    "source",
    "from_address",
    "subject",
    "body",
    "text_combined",
    "label",
    "has_url",
]


def normalize_label(raw: Any) -> Optional[int]:
    """Return 0/1 label or ``None`` if it cannot be normalized.

    The function handles a variety of common representations such as
    ``"Phishing"``, ``"spam"``, numeric strings, booleans, etc.  If the
    input is missing or unrecognised the result is ``None`` which allows the
    caller to drop the row later.
    """

    # Checks if the value is missing (NaN, None, etc.);
    if pd.isna(raw):
        return None

    # Convert True -> 1, False -> 0
    if isinstance(raw, bool):
        return int(raw)

    if isinstance(raw, Real) and raw in (0, 1):
        return int(raw)

    # Convert raw to string, remove leading/trailing whitespace, and convert to lowercase
    text = str(raw).strip().lower()
    # Check if text is empty after stripping
    if not text:
        return None

    # Sets of words that translate to phishing (1) and safe (0) labels.
    positives = {
        "phishing email",
        "phishing",
        "phish",
        "spam",
        "malicious",
        "1",
        "1.0",
        "yes",
        "true",
        "t",
        "y",
    }
    negatives = {
        "safe email",
        "safe",
        "ham",
        "legit",
        "benign",
        "0",
        "0.0",
        "no",
        "false",
        "f",
        "n",
    }

    if text in positives:
        return 1
    if text in negatives:
        return 0

    # If none of the above cases matched, the label cannot be normalized
    return None


def normalize_has_url(raw: Any) -> Optional[int]:
    """Return 0/1 for URL-presence flag or ``None`` if unknown/invalid."""

    if pd.isna(raw):
        return None

    if isinstance(raw, bool):
        return int(raw)

    if isinstance(raw, Real) and raw in (0, 1):
        return int(raw)

    text = str(raw).strip().lower()
    if not text:
        return None

    positives = {"1", "1.0", "yes", "true", "t", "y"}
    negatives = {"0", "0.0", "no", "false", "f", "n"}

    if text in positives:
        return 1
    if text in negatives:
        return 0

    return None


def _prepare_dataframe(
    df: pd.DataFrame, source: str
) -> pd.DataFrame:
    """Standardise a dataframe returned by one of the structure-specific load_fns.

    - make sure all required columns exist
    - make text columns strings and fill missing values
    - normalise ``label``
    - drop rows without a valid label or an empty body
    - add ``source`` and ``text_combined``
    - trim the output to the canonical columns
    """

    df = df.copy()
    # Loop through each required column name
    for col in ("from_address", "subject", "body", "label", "has_url"):
        if col not in df.columns:
            # Add the column: empty string for text fields, None for label and has_url
            df[col] = "" if (col != "label" and col != "has_url") else None

    # Process text columns: convert missing values and ensure they are strings
    for col in ("from_address", "subject", "body"):
        # .fillna("") replaces NaN/None with empty string; .astype(str) converts all values to string type
        df[col] = df[col].fillna("").astype(str)

    # Apply normalization helpers to canonical binary fields
    df["label"] = df["label"].apply(normalize_label)
    df["has_url"] = df["has_url"].apply(normalize_has_url)

    # Keep only rows where label was successfully normalized (not None)
    df = df[df["label"].notna()]

    # Remove rows where the body is empty or contains only whitespace
    # .str.strip() removes leading/trailing whitespace; .astype(bool) converts to boolean (empty string = False)
    df = df[df["body"].str.strip().astype(bool)]

    # Add a column with the source identifier for tracking which dataset each row came from
    df["source"] = source

    # Create a combined text field by concatenating from_address, subject, and body with labels
    # String concatenation using + operator on pandas Series aligns rows automatically
    df["text_combined"] = (
        "FROM: " + df["from_address"]
        + " SUBJECT: "
        + df["subject"]
        + " BODY: "
        + df["body"]
    )

    # Keep only the canonical columns
    cols = ["source", "from_address", "subject", "body", "label", "text_combined", "has_url"]
    # .loc[:, [...]] selects all rows and only the specified columns; filtered to columns that exist
    df = df.loc[:, [c for c in cols if c in df.columns]]

    # Return the standardized dataframe
    return df

def _select_loader(path) -> Callable:
    """Select the appropriate loader function based on the columns in the dataset.

    """

    if hasattr(path, "seek"):
        try:
            path.seek(0)
        except Exception:
            pass

    df = pd.read_csv(path, nrows=0)  # Read only the header row to get column names
    cols = set(df.columns.str.lower())  # Convert column names to lowercase for case-ins
    
    if "email text" in cols and "email type" in cols:
        return load_email_text_type
    elif "subject" in cols and "body" in cols and "label" in cols and len(cols) == 3:
        return load_subject_body_label
    elif "text_combined" in cols and "label" in cols:
        return load_text_combined
    else:
        return load_7col

def load_7col(
    path: Union[str, Path],
    source: str,
    label_col: str = "label",
) -> pd.DataFrame:
    """Load files with sender, receiver, date, subject, body, urls, and label columns.

    ``label_col`` may be used if the dataset has an extra column containing the
    label (for instance ``"is_phish"``).  If no label column is present the
    returned frame will be empty since we can't perform supervised training.
    ``source`` is a short identifier stored in the ``source`` column of the
    resulting frame.
    """

    df = pd.read_csv(path)
    df["from_address"] = df.get("sender", "")
    df["subject"] = df.get("subject", "")
    df["body"] = df.get("body", "")
    df["label"] = df.get(label_col, None)
    df["has_url"] = df.get("urls", None)

    # Pass the dataframe to _prepare_dataframe to standardize it and apply transformations
    return _prepare_dataframe(df, source)


def load_email_text_type(
    path: Union[str, Path],
    source: str,
    text_col: str = "Email Text",
    label_col: str = "Email Type",
) -> pd.DataFrame:
    """Load two-column datasets with email text and an email type column."""

    df = pd.read_csv(path)
    df["from_address"] = ""
    df["subject"] = ""
    df["body"] = df.get(text_col, "")
    df["label"] = df.get(label_col, None)
    return _prepare_dataframe(df, source)


def load_subject_body_label(
    path: Union[str, Path],
    source: str,
    subj_col: str = "subject",
    body_col: str = "body",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load datasets that already have ``subject``, ``body`` and ``label``."""

    df = pd.read_csv(path)
    df["from_address"] = ""
    df["subject"] = df.get(subj_col, "")
    df["body"] = df.get(body_col, "")
    df["label"] = df.get(label_col, None)
    return _prepare_dataframe(df, source)

def load_text_combined(
    path: Union[str, Path],
    source: str,
    text_col: str = "text_combined",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load datasets where all textual information is in a single column."""

    df = pd.read_csv(path)
    df["from_address"] = ""
    df["subject"] = ""
    df["body"] = df.get(text_col, "")
    df["label"] = df.get(label_col, None)
    return _prepare_dataframe(df, source)


def load_all(
    configs: List[Dict[str, Any]], dedupe: bool = True, 
) -> pd.DataFrame:
    """Load a collection of datasets and concatenate them.

    ``configs`` - a list of dictionaries containing the
    the keys ``"load_fn"`` (a callable such as :func:`load_7col`),
    ``"path"`` (a filename) and ``"source"`` (string).  Additional kwargs are
    passed through to the load_fn.

    Example::

        datasets = [
            {
                "load_fn": load_7col,
                "path": "data/ceas08.csv",
                "source": "kaggle_bundle_ceas_08",
                "label_col": "is_phish",
            },
            {
                "load_fn": load_email_text_type,
                "path": "data/simple_phish.csv",
                "source": "kaggle_simple_phish",
            },
        ]
        df = load_all(datasets)

    The returned dataframe will already have ``text_combined`` and a normalized
    0/1 ``label``.  ``dedupe`` controls whether duplicate
    ``text_combined`` rows are removed.
    """

    # Initialize an empty list to hold all dataframes to be concatenated
    frames: List[pd.DataFrame] = []
    for cfg in configs:
        # Work on a copy to avoid mutating user-provided configs
        cfg_local = dict(cfg)
        # .pop(key) removes the key from the dictionary and returns its value, default value is None
        load_fn: Callable = cfg_local.pop("load_fn", None)
        path = cfg_local.pop("path")
        if load_fn is None:
            load_fn = _select_loader(path)

        source = cfg_local.pop("source")

        # if path appears to be a file-like object, rewind it so it can be read again
        if hasattr(path, "seek"):
            try:
                path.seek(0)
            except Exception:
                pass
        # Call the load_fn function with the path, source, and remaining kwargs
        df = load_fn(path, source=source, **cfg_local)
        # If returned dataframe is not empty, add to list of dataframes
        if not df.empty:
            frames.append(df)

    # Return an empty df with the correct schema if no data was loaded from any dataset
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # pd.concat() joins multiple dataframes vertically (stacks them on top of each other)
    # ignore_index=True resets the row index to 0, 1, 2, ... (instead of keeping original indices)
    result = pd.concat(frames, ignore_index=True)
    if dedupe and "text_combined" in result.columns:
        result = result.drop_duplicates(subset=["text_combined"])
    return result



# Exotic exports
__all__ = [
    "load_7col",
    "load_email_text_type",
    "load_subject_body_label",
    "load_text_combined",
    "load_all",
    "normalize_label",
    "normalize_has_url",
]
