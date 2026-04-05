import pandas as pd
from typing import Tuple, Optional, Dict


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe produced by data_loader.py.

    1. drop rows where ``label`` is missing (``NaN`` or ``None``).
    2. drop rows where ``text_combined`` is empty after stripping whitespace.
    3. normalise whitespace in ``text_combined`` (collapse ``\\s+`` to single
       space and strip at ends).

    The returned frame is a copy; the original is not modified.
    """
    df = df.copy()

    df = df[df["label"].notna()]  # drop rows with missing labels

    df = df[df["label"].isin([0, 1])]

    df["label"] = df["label"].astype(int)  # ensure label is int type

    # ensure text_combined is string and normalise whitespace
    df["text_combined"] = df["text_combined"].fillna("").astype(str)

    df["text_combined"] = (
        df["text_combined"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # drop rows where text_combined ended up empty
    df = df[df["text_combined"].astype(bool)]

    return df


def sanity_check(
    df: pd.DataFrame, before_count: Optional[int] = None
) -> Dict[str, object]:
    """Print a brief report about *df* and return the statistics.

    ``before_count`` can be supplied in order to compute the number of rows
    removed during cleaning.

    The returned dictionary contains keys ``total``, ``label_counts``,
    ``ratio`` and ``removed``.  
    
    ``ratio`` is ``count1 / count0`` (``inf`` if
    there are no zero-label rows).
    """
    total = len(df)
    counts = df["label"].value_counts().to_dict()
    zeros = counts.get(0, 0)
    ones = counts.get(1, 0)
    ratio = ones / zeros if zeros > 0 else float("inf")
    removed = None
    if before_count is not None:
        removed = before_count - total

    print(f"Total rows: {total}")
    print(f"Label distribution: 0_count={zeros}, 1_count={ones} (ratio {ratio:.2f})")
    if removed is not None:
        print(f"Rows removed: {removed}")

    if "source" in df.columns:
        print("\nCounts by source:")
        print(df.groupby("source")["label"].value_counts())

        print("\nProportions by source:")
        print(df.groupby("source")["label"].value_counts(normalize=True))


    stats = {"total": total, "label_counts": counts, "ratio": ratio, "removed": removed}
    return stats

def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split using scikit-learn.

    This performs a two-stage stratified split:

    1. Split the full dataframe into ``train`` and ``temp`` where
       ``temp_size = val_size + test_size`` using ``stratify=df['label']``.
    2. Split ``temp`` into ``val`` and ``test`` where
       ``test_frac_of_temp = test_size / (val_size + test_size)`` again using
       stratification on the labels.

    Parameters
    - ``val_size`` and ``test_size`` are fractions of the original dataset
      (both must be > 0 and their sum must be < 1.0).
    - ``random_state`` is forwarded to scikit-learn to make splits repeatable.

    Returns ``(train, val, test)`` as three new dataframes with indices reset
    (``reset_index(drop=True)``).  All columns and row contents are preserved.

    Raises ``ValueError`` for invalid size arguments or when scikit-learn's
    stratified splitting fails due to insufficient per-class examples.
    """
    # validate sizes
    if val_size <= 0:
        raise ValueError("val_size must be > 0")
    if test_size <= 0:
        raise ValueError("test_size must be > 0")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0")

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for stratified_split; install it with 'pip install scikit-learn'"
        ) from e

    temp_size = val_size + test_size

    try:
        train, temp = train_test_split(  # first split into train and temp using scikit-learn
            df,
            test_size=temp_size,
            stratify=df["label"],
            random_state=random_state,
        )
        # fraction of temp that should be the final test set
        test_frac_of_temp = test_size / temp_size
        val, test = train_test_split(
            temp,
            test_size=test_frac_of_temp,
            stratify=temp["label"],
            random_state=random_state,
        )
    except ValueError as e:
        # sklearn complains when classes are too small for stratification
        raise ValueError(
            "Stratified split failed: each class needs enough members to be present in each split. "
            "Ensure each label has at least two samples (and enough samples for the requested fractions)."
        ) from e

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def preprocess_and_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: Optional[int] = None,
    save_paths: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full preprocessing pipeline and return train/val/test splits.

    ``save_paths`` may be a mapping with keys ``"train"``, ``"val"`` and
    ``"test"`` giving filenames; the corresponding datasets will be written
    as CSV if provided.
    """
    before = len(df)
    print("Before cleaning and deduplication:")
    sanity_check(df, before_count=before)
    clean = basic_cleaning(df)
    deduped = clean.drop_duplicates(subset=["text_combined"])
    print("\nAfter cleaning and deduplication:")
    sanity_check(deduped, before_count=len(clean))
    train, val, test = stratified_split(
        deduped, val_size=val_size, test_size=test_size, random_state=random_state
    )
    if save_paths:
        if "train" in save_paths:
            train.to_csv(save_paths["train"], index=False)
        if "val" in save_paths:
            val.to_csv(save_paths["val"], index=False)
        if "test" in save_paths:
            test.to_csv(save_paths["test"], index=False)
    return train, val, test
