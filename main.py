import argparse
from types import SimpleNamespace
from pathlib import Path
from typing import Any

from src.data_loader import load_all
from src.preprocessing import preprocess_and_split


def discover_csv_configs(raw_dir: Path) -> list[dict[str, Any]]:
    """Build load configs for all CSV files under raw_dir.

    Uses data_loader.load_all auto-loader selection by omitting load_fn.
    """
    csv_paths = sorted(raw_dir.rglob("*.csv"))
    configs: list[dict[str, Any]] = []

    for path in csv_paths:
        rel_parent = path.parent.relative_to(raw_dir)
        prefix = str(rel_parent).replace("\\", "_").replace("/", "_")
        stem = path.stem
        source = f"{prefix}_{stem}" if prefix != "." else stem
        configs.append({
            "path": str(path),
            "source": source,
        })

    return configs


def run_pipeline(
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
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train baseline model after preprocessing using src/layers/classical/train_ml.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "linear_svm"],
        help="Which baseline model to train when --train-model is set",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="text_combined",
        help="Which text column to use for TF-IDF when --train-model is set",
    )
    return parser.parse_args()



def main() -> None:
    '''
    Example usage:
    .\.venv\Scripts\python main.py --train-model --model logreg
    .\.venv\Scripts\python main.py --train-model --model linear_svm
    .\.venv\Scripts\python main.py --train-model --model logreg --text-col body

    
    '''
    args = parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        dedupe=not args.no_dedupe,
    )
    if args.train_model:
        from src.layers.classical import train_ml
        train_args = SimpleNamespace(
            processed_dir=args.processed_dir,
            models_dir=Path("models"),
            train_file="train.csv",
            val_file="val.csv",
            test_file="test.csv",
            target_col="label",
            text_col=args.text_col,
            max_iter=2000,
            class_weight="balanced",
            random_state=args.random_state,
            model=args.model,
        )
        train_ml.run_training(train_args)


if __name__ == "__main__":
    main()
