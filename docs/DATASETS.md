# ModelGuard — Dataset Register & Lineage Documentation

## 1. Primary Dataset: Metro Interstate Traffic Volume

- **Publisher:** UCI Machine Learning Repository
- **Dataset ID:** 492
- **Citation:** Hogue, J. (2019). *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. DOI: [10.24432/C5X60B](https://doi.org/10.24432/C5X60B).
- **Licence:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Task:** Hourly traffic volume regression.
- **Total Expected Observations:** 48,204.
- **Period:** 2012-10-02 to 2018-09-30.

### 1.1 Schema & Feature Contract (v1)

| Feature | Type | Nullable | Valid Bounds / Constraints | Description |
|---|---|---|---|---|
| `holiday` | String | Yes | US holiday name or "None" | Holiday indicator |
| `temp` | Float | No | `(150.0, 350.0)` Kelvin | Hourly ambient temperature |
| `rain_1h` | Float | No | `[0.0, 500.0)` mm | Hourly rainfall |
| `snow_1h` | Float | No | `[0.0, 50.0)` mm | Hourly snowfall |
| `clouds_all` | Integer | No | `[0, 100]` % | Cloud coverage |
| `weather_main` | Categorical | No | Clear, Clouds, Rain, Drizzle, Mist, Haze, Fog, Thunderstorm, Snow, Squall, Smoke | Weather summary category |
| `weather_description`| String | No | Non-empty string | Detailed weather conditions |
| `date_time` | DateTime string | No | `YYYY-MM-DD HH:MM:SS` | Timestamp in CST |
| `traffic_volume` | Integer | No | `[0, 12000]` | **Target:** Hourly I-94 westbound volume |

### 1.2 Data Integrity & Provenance Rules
1. **Cryptographic Checksum:** Every dataset snapshot uploaded is fingerprinted with streaming SHA-256 before ingestion.
2. **Immutability:** Modifying or replacing an existing version tag (e.g. `v1.0.0`) is prohibited with HTTP 409 Conflict.
3. **Contract Validation:** Pandera DataFrameSchema evaluates column existence, type coercion, nullability, and sensor range filters.
4. **Validation Evidence:** All schema errors and quality warnings are persisted to the database as structured JSON reports.

---

## 2. Ingestion & Local Fixtures

- `datasets/manifests/metro_traffic_manifest.json`: Machine-readable metadata and citation manifest.
- `datasets/adapters/uci_metro_traffic.py`: Script to download and decompress the official UCI dataset archive.
- `datasets/fixtures/sample_metro_valid.csv`: Clean test fixture for test verification.
- `datasets/fixtures/sample_metro_invalid.csv`: Deterministic failure fixture (out-of-bounds Kelvin temperature, illegal weather category, negative traffic volume).
