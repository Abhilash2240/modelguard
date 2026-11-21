import hashlib
import tempfile
from pathlib import Path

from app.domain.hashing import calculate_schema_hash, calculate_sha256


def test_sha256_bytes():
    payload = b"modelguard_deterministic_payload_2026"
    expected = hashlib.sha256(payload).hexdigest()
    assert calculate_sha256(payload) == expected


def test_sha256_streaming_file():
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(b"line1\nline2\nline3\n")
        f_path = Path(f.name)

    try:
        hash1 = calculate_sha256(f_path)
        # Modify a single byte
        with open(f_path, "wb") as f:
            f.write(b"line1\nline2\nline4\n")
        hash2 = calculate_sha256(f_path)

        assert hash1 != hash2
    finally:
        if f_path.exists():
            f_path.unlink()


def test_schema_hash_deterministic():
    cols1 = [{"name": "temp", "dtype": "float64"}, {"name": "holiday", "dtype": "object"}]
    cols2 = [{"name": "holiday", "dtype": "object"}, {"name": "temp", "dtype": "float64"}]
    # Ordering should not affect normalized schema hash
    assert calculate_schema_hash(cols1) == calculate_schema_hash(cols2)
