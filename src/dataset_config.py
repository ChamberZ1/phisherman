from pathlib import Path
from typing import Any


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
