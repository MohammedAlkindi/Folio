"""
One-time script to download all-MiniLM-L6-v2 via hf-mirror.com using plain
requests (bypasses huggingface_hub's domain-validation check).

Run:  python download_model.py
Then: folio ingest   (uses the local model path automatically)
"""

import sys
from pathlib import Path

import requests

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MIRROR = "https://hf-mirror.com"
LOCAL_DIR = Path("models/all-MiniLM-L6-v2")

FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
    "special_tokens_map.json",
    "sentence_bert_config.json",
    "modules.json",
    "model.safetensors",
    "1_Pooling/config.json",
]


def download(url: str, dest: Path) -> None:
    print(f"  {dest.name} ...", end="", flush=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
    mb = dest.stat().st_size / 1_048_576
    print(f"  {mb:.1f} MB")


def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_DIR / "1_Pooling").mkdir(exist_ok=True)

    print(f"Downloading {MODEL_ID} → {LOCAL_DIR}/\n")

    failed = []
    for rel in FILES:
        dest = LOCAL_DIR / rel
        if dest.exists():
            print(f"  {rel}  (already exists, skipping)")
            continue
        url = f"{MIRROR}/{MODEL_ID}/resolve/main/{rel}"
        try:
            download(url, dest)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failed.append(rel)

    if failed:
        print(f"\nWarning: {len(failed)} file(s) failed: {failed}")
        sys.exit(1)

    print(f"\nDone. Model saved to: {LOCAL_DIR.resolve()}")
    print("\nUpdate config/folio_config.yaml — change the model line to:")
    print(f'  model: "{LOCAL_DIR.resolve()}"')


if __name__ == "__main__":
    main()
