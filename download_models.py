"""Download pre-trained model weights from Hugging Face.

Usage:
    python download_models.py
"""
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "zchf420/phisherman"

DISTILBERT_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "training_args.bin",
]

JOBLIB_FILES = [
    "textcombined_svm.joblib",
    "isolation_forest.joblib",
]


def main() -> None:
    Path("models/distilbert").mkdir(parents=True, exist_ok=True)

    print("Downloading DistilBERT weights...")
    for filename in DISTILBERT_FILES:
        hf_hub_download(repo_id=REPO_ID, filename=filename,
                        local_dir="models/distilbert")
        print(f"  {filename}")

    print("Downloading sklearn models...")
    for filename in JOBLIB_FILES:
        hf_hub_download(repo_id=REPO_ID, filename=filename,
                        local_dir="models")
        print(f"  {filename}")

    print("\nDone. Run `python app.py` to start the web app.")


if __name__ == "__main__":
    main()
