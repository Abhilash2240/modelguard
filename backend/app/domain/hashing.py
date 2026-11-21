import hashlib
from pathlib import Path


def calculate_sha256(file_path_or_bytes: str | Path | bytes, chunk_size: int = 65536) -> str:
    """Calculate the cryptographic SHA-256 hex digest of a file or bytes.

    Uses streaming chunks to avoid loading large files into memory.
    """
    hasher = hashlib.sha256()

    if isinstance(file_path_or_bytes, bytes):
        hasher.update(file_path_or_bytes)
        return hasher.hexdigest()

    path = Path(file_path_or_bytes)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def calculate_schema_hash(columns: list) -> str:
    """Calculate deterministic SHA-256 hash of a column structure."""
    normalized = "|".join(f"{col['name']}:{col['dtype']}" for col in sorted(columns, key=lambda x: x["name"]))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
