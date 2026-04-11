import argparse
from pathlib import Path

from src.data_loader import load_all
from src.dataset_config import discover_csv_configs
from src.preprocessing import preprocess_and_split


def build_processed_splits(
    raw_dir: Path,
    processed_dir: Path,
    val_size: float,
    test_size: float,
    random_state: int | None,
    dedupe: bool,
) -> tuple[Path, Path, Path]:
    configs = discover_csv_configs(raw_dir)
    if not configs:
        raise FileNotFoundError(f"No CSV files found under: {raw_dir}")

    print(f"Discovered {len(configs)} raw CSV file(s).")
    df = load_all(configs, dedupe=dedupe)
    print(f"Loaded normalized rows: {len(df)}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train, val, test = preprocess_and_split(
        df,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        save_paths={
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    )

    print(f"Saved train: {train_path} ({len(train)} rows)")
    print(f"Saved val:   {val_path} ({len(val)} rows)")
    print(f"Saved test:  {test_path} ({len(test)} rows)")

    return train_path, val_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load raw phishing datasets, preprocess/split, and save processed splits.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable exact deduplication in data_loader.load_all",
    )
    return parser.parse_args()


def main() -> None:
    '''
    Example usage:
    .\.venv\Scripts\python main.py
    .\.venv\Scripts\python main.py --raw-dir data/raw --processed-dir data/processed
    '''
    args = parse_args()
    build_processed_splits(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        dedupe=not args.no_dedupe,
    )


if __name__ == "__main__":
    main()
