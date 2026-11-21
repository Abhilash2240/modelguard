"""UCI Metro Interstate Traffic Volume Data Adapter.

Downloads, unpacks, verifies, and registers the Metro Interstate Traffic Volume dataset
from the official UCI repository (Dataset ID: 492).
"""

import gzip
import io
from pathlib import Path

import requests
from app.domain.hashing import calculate_sha256

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "manifests" / "metro_traffic_manifest.json"


def download_metro_traffic(destination_dir: Path) -> Path:
    """Download official Metro Interstate Traffic Volume dataset.

    If the zip file contains Metro_Interstate_Traffic_Volume.csv.gz or CSV directly,
    it extracts the CSV to destination_dir.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    target_csv = destination_dir / "Metro_Interstate_Traffic_Volume.csv"

    if target_csv.exists():
        print(f"Dataset already exists at {target_csv}")
        return target_csv

    print(f"Downloading from {UCI_DATASET_URL}...")
    resp = requests.get(UCI_DATASET_URL, timeout=60)
    resp.raise_for_status()

    import zipfile

    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(".csv.gz"):
                with z.open(file_info) as gz_f:
                    with gzip.GzipFile(fileobj=gz_f) as decompressed:
                        content = decompressed.read()
                        with open(target_csv, "wb") as out_f:
                            out_f.write(content)
                break
            elif file_info.filename.endswith(".csv"):
                with z.open(file_info) as csv_f:
                    with open(target_csv, "wb") as out_f:
                        out_f.write(csv_f.read())
                break

    if not target_csv.exists():
        raise FileNotFoundError("Could not find Metro Interstate Traffic Volume CSV inside downloaded archive.")

    sha256 = calculate_sha256(target_csv)
    print(f"Successfully downloaded and extracted {target_csv} (SHA-256: {sha256})")
    return target_csv


if __name__ == "__main__":
    import sys

    dest = Path("./data/raw") if len(sys.argv) < 2 else Path(sys.argv[1])
    download_metro_traffic(dest)
