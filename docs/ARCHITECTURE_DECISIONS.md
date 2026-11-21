# ModelGuard — Architecture Decision Records (ADRs)

## ADR 001: Modular Monolith with Asynchronous Worker
- **Status:** Accepted
- **Context:** An enterprise MLOps control plane requires decoupled long-running pipelines (training, drift reporting) from low-latency HTTP request handlers, without unnecessary distributed microservice complexity in the initial release.
- **Decision:** Build ModelGuard as a modular monolith in FastAPI, sharing domain models and database transactions, with Redis + RQ worker processes handling asynchronous execution.
- **Consequences:** Low operational overhead for local and Docker Compose environments, clean boundaries, easily split into microservices if needed later.

## ADR 002: Pandera as DataFrame Contract Adapter
- **Status:** Accepted
- **Context:** Tabular dataset validation requires declarative schema checks, nullability constraints, and domain-specific bounds (e.g. temperature physics ranges).
- **Decision:** Adopt Pandera for DataFrame-level schema definitions while abstracting validation through ModelGuard's `BaseDataContract` domain interface.
- **Consequences:** Highly expressive, vectorized validation. Business rules remain ModelGuard-owned, isolating the domain layer from specific validation libraries.

## ADR 003: SQLite / PostgreSQL Dual Persistence Strategy
- **Status:** Accepted
- **Context:** Developers need fast, zero-dependency unit and API test execution, while production local Compose deployment requires durable multi-process PostgreSQL.
- **Decision:** Use SQLAlchemy 2.x declarative models and session factories compatible with both in-memory SQLite (using `StaticPool` for test isolation) and PostgreSQL (for Docker Compose and local development).
- **Consequences:** Pytest test suite runs in under 1 second without requiring an active PostgreSQL container, while preserving strict transactional foreign keys.

## ADR 004: Immutable Dataset Versioning via SHA-256 Fingerprinting
- **Status:** Accepted
- **Context:** Model reproducibility requires exact tracking of training data. Overwriting or silently mutating raw datasets invalidates model auditability.
- **Decision:** Calculate streaming SHA-256 checksums and normalized schema hashes upon snapshot registration. Forbid version tag mutation.
- **Consequences:** Guaranteed data provenance. Before training or promotion, ModelGuard verifies that the physical file digest matches the database record.
