from pathlib import Path

from src.dataset_config import discover_csv_configs
from src.data_loader import load_all


def build_benign_dataset(
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("data/processed/benign_only.csv"),
) -> Path:
    configs = discover_csv_configs(raw_dir)
    if not configs:
        raise FileNotFoundError(f"No CSV files found under: {raw_dir}")

    df = load_all(configs, dedupe=True)
    benign = df[df["label"] == 0].reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    benign.to_csv(output_path, index=False)

    print(f"Benign rows: {len(benign)}")
    print(f"Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    build_benign_dataset()
