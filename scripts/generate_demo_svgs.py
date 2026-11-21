"""Generates high-fidelity SVG screenshots for ModelGuard UI views."""

from pathlib import Path

OUT_DIR = Path("docs/demo/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def svg_frame(title: str, subtitle: str, content_svg: str, width: int = 1000, height: int = 650) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <style>
      .bg {{ fill: #0b0f19; }}
      .card {{ fill: #111827; stroke: #1f2937; stroke-width: 1.5; rx: 8; }}
      .header-bg {{ fill: #0f172a; stroke: #1e293b; stroke-width: 1; }}
      .text-title {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 20px; font-weight: 700; fill: #f8fafc; }}
      .text-subtitle {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; fill: #94a3b8; }}
      .text-nav {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 13px; font-weight: 500; fill: #cbd5e1; }}
      .text-bold {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 600; fill: #f1f5f9; }}
      .text-muted {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 12px; fill: #94a3b8; }}
      .text-val {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 24px; font-weight: 700; fill: #f8fafc; }}
      .badge-success {{ fill: #064e3b; stroke: #059669; stroke-width: 1; rx: 4; }}
      .badge-danger {{ fill: #7f1d1d; stroke: #dc2626; stroke-width: 1; rx: 4; }}
      .badge-blue {{ fill: #1e3a8a; stroke: #2563eb; stroke-width: 1; rx: 4; }}
      .badge-amber {{ fill: #78350f; stroke: #d97706; stroke-width: 1; rx: 4; }}
      .badge-text {{ font-family: monospace; font-size: 11px; font-weight: 600; fill: #f8fafc; text-anchor: middle; }}
      .code-text {{ font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 12px; fill: #38bdf8; }}
      .btn-blue {{ fill: #2563eb; rx: 6; }}
      .btn-text {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 12px; font-weight: 600; fill: #ffffff; text-anchor: middle; }}
    </style>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" class="bg" />

  <!-- Top App Bar -->
  <rect width="{width}" height="60" class="header-bg" />
  <circle cx="28" cy="30" r="14" fill="#2563eb" />
  <text x="28" y="35" text-anchor="middle" font-family="sans-serif" font-weight="900" font-size="14" fill="#ffffff">M</text>
  <text x="52" y="34" class="text-title" font-size="16">ModelGuard</text>
  <text x="148" y="34" class="text-muted" font-size="12">CONTROL PLANE</text>

  <!-- Health Badge -->
  <rect x="860" y="18" width="115" height="24" rx="12" fill="#064e3b" stroke="#059669" stroke-width="1" />
  <circle cx="875" cy="30" r="4" fill="#10b981" />
  <text x="920" y="34" class="badge-text" fill="#34d399">SYSTEM HEALTHY</text>

  <!-- Sidebar -->
  <rect x="0" y="60" width="200" height="{height - 60}" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <g transform="translate(15, 80)">
    <text x="10" y="15" class="text-nav" fill="#60a5fa">◆ Overview</text>
    <text x="10" y="45" class="text-nav">📁 Datasets</text>
    <text x="10" y="75" class="text-nav">⚡ Training</text>
    <text x="10" y="105" class="text-nav">📦 Model Registry</text>
    <text x="10" y="135" class="text-nav">🚀 Deployments</text>
    <text x="10" y="165" class="text-nav">🔮 Inference</text>
    <text x="10" y="195" class="text-nav">📊 Monitoring</text>
    <text x="10" y="225" class="text-nav">🔔 Alerts &amp; Audit</text>
  </g>

  <!-- Main View Area -->
  <g transform="translate(225, 75)">
    <!-- Page Header -->
    <text x="0" y="24" class="text-title">{title}</text>
    <text x="0" y="44" class="text-subtitle">{subtitle}</text>
    <g transform="translate(0, 60)">
      {content_svg}
    </g>
  </g>
</svg>
"""


# 1. Overview Dashboard
c1 = """
  <!-- 4 Stat Cards -->
  <rect x="0" y="0" width="175" height="85" class="card" />
  <text x="15" y="25" class="text-muted">ACTIVE CHAMPION</text>
  <text x="15" y="55" class="text-val">v1.2.0</text>
  <text x="15" y="73" class="code-text" font-size="10">metro-traffic-volume</text>

  <rect x="190" y="0" width="175" height="85" class="card" />
  <text x="205" y="25" class="text-muted">STAGING STATUS</text>
  <text x="205" y="55" class="text-val" fill="#38bdf8">v1.3.0-rc</text>
  <text x="205" y="73" class="code-text" font-size="10">AWAITING APPROVAL</text>

  <rect x="380" y="0" width="175" height="85" class="card" />
  <text x="395" y="25" class="text-muted">CANDIDATE MAE</text>
  <text x="395" y="55" class="text-val" fill="#34d399">284.15</text>
  <text x="395" y="73" class="text-muted" font-size="10">+83.7% vs baseline</text>

  <rect x="570" y="0" width="175" height="85" class="card" />
  <text x="585" y="25" class="text-muted">OPEN ALERTS</text>
  <text x="585" y="55" class="text-val" fill="#f87171">0 ACTIVE</text>
  <text x="585" y="73" class="text-muted" font-size="10">All systems optimal</text>

  <!-- Large Panel -->
  <rect x="0" y="105" width="745" height="340" class="card" />
  <text x="20" y="135" class="text-bold">Active Model Quality &amp; Serving Health</text>
  <text x="20" y="155" class="text-muted">Real-time metrics for champion serving on traffic regression</text>
  <line x1="20" y1="170" x2="725" y2="170" stroke="#1f2937" stroke-width="1" />

  <text x="20" y="200" class="text-muted">Inference Latency (p50 / p95 / p99):</text>
  <text x="300" y="200" class="text-bold" fill="#38bdf8">3.2 ms / 5.8 ms / 11.4 ms</text>

  <text x="20" y="235" class="text-muted">Throughput &amp; Error Rate:</text>
  <text x="300" y="235" class="text-bold" fill="#34d399">142 req/sec | 0.00% errors</text>

  <text x="20" y="270" class="text-muted">Delayed Ground Truth Coverage:</text>
  <text x="300" y="270" class="text-bold">96.8% (4,820 / 4,980 events)</text>

  <text x="20" y="305" class="text-muted">Rolling 24h Evaluated MAE:</text>
  <text x="300" y="305" class="text-bold" fill="#34d399">288.4 veh/hr (Within ±2% of validation)</text>

  <rect x="20" y="340" width="705" height="80" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="370" class="text-bold" fill="#38bdf8">💡 Architecture Invariant Guaranteed</text>
  <text x="40" y="395" class="text-muted">Zero-downtime rollback active: previous candidate v1.1.0 ready for atomic fallback.</text>
"""

# 2. Datasets
c2 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Registered Datasets &amp; Immutable Snapshots</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <text x="20" y="75" class="text-muted">NAME</text>
  <text x="220" y="75" class="text-muted">VERSION</text>
  <text x="320" y="75" class="text-muted">SHA-256 HASH</text>
  <text x="520" y="75" class="text-muted">ROWS</text>
  <text x="610" y="75" class="text-muted">VALIDATION</text>
  <line x1="20" y1="90" x2="725" y2="90" stroke="#1f2937" stroke-width="1" />

  <text x="20" y="125" class="text-bold">metro-traffic-volume</text>
  <text x="220" y="125" class="code-text">v1.0.0</text>
  <text x="320" y="125" class="code-text" font-size="11">e3b0c44298fc1c149afb...</text>
  <text x="520" y="125" class="text-nav">48,204</text>
  <rect x="610" y="110" width="70" height="22" class="badge-success" />
  <text x="645" y="125" class="badge-text" fill="#34d399">PASSED</text>

  <line x1="20" y1="145" x2="725" y2="145" stroke="#1f2937" stroke-width="0.5" />

  <text x="20" y="180" class="text-bold">metro-traffic-volume</text>
  <text x="220" y="180" class="code-text">v0.9.0-test</text>
  <text x="320" y="180" class="code-text" font-size="11">2438a76c8849b2ef819a...</text>
  <text x="520" y="180" class="text-nav">5</text>
  <rect x="610" y="165" width="70" height="22" class="badge-danger" />
  <text x="645" y="180" class="badge-text" fill="#f87171">FAILED</text>

  <!-- Contract Breakdown Card -->
  <rect x="20" y="220" width="705" height="190" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="250" class="text-bold">Contract Validation Evidence: Pandera Schema Contract v1</text>
  <text x="40" y="275" class="text-muted">Total Checks Evaluated: 18 | Nullability Checks: 9 | Sensor Bounds Checks: 6</text>
  <text x="40" y="305" class="code-text" fill="#34d399">✓ temp bounds [150.0, 350.0] Kelvin: 0 violations</text>
  <text x="40" y="330" class="code-text" fill="#34d399">✓ rain_1h bounds [0.0, 500.0] mm: 0 violations</text>
  <text x="40" y="355" class="code-text" fill="#34d399">✓ weather_main categorical domain (11 categories): 0 violations</text>
  <text x="40" y="380" class="code-text" fill="#34d399">✓ traffic_volume target bounds [0, 12000]: 0 violations</text>
"""

# 3. Training & MLflow
c3 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Training Jobs &amp; Experiment Tracking</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <text x="20" y="75" class="text-muted">JOB ID</text>
  <text x="180" y="75" class="text-muted">STATUS</text>
  <text x="280" y="75" class="text-muted">CANDIDATE MAE</text>
  <text x="420" y="75" class="text-muted">BASELINE MAE</text>
  <text x="550" y="75" class="text-muted">IMPROVEMENT</text>
  <text x="660" y="75" class="text-muted">MLFLOW</text>
  <line x1="20" y1="90" x2="725" y2="90" stroke="#1f2937" stroke-width="1" />

  <text x="20" y="125" class="code-text">job-e2e-001</text>
  <rect x="180" y="110" width="80" height="22" class="badge-success" />
  <text x="220" y="125" class="badge-text" fill="#34d399">SUCCEEDED</text>
  <text x="280" y="125" class="text-bold" fill="#34d399">284.15</text>
  <text x="420" y="125" class="text-muted">1,745.20</text>
  <text x="550" y="125" class="text-bold" fill="#34d399">+83.72%</text>
  <text x="660" y="125" class="code-text" fill="#60a5fa">run-4f81c</text>

  <!-- Lifecycle Progress -->
  <rect x="20" y="160" width="705" height="250" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="195" class="text-bold">15-Stage Training Lifecycle Execution Trace</text>
  <text x="40" y="225" class="code-text" fill="#38bdf8">1. Atomic Guard &amp; Duplicate Check: PASSED</text>
  <text x="40" y="250" class="code-text" fill="#38bdf8">4-6. MLflow Run Initialization &amp; Immediate ID Persist: PASSED (ID: f8a12bcde)</text>
  <text x="40" y="275" class="code-text" fill="#38bdf8">7. SHA-256 Snapshot Checksum Verification: PASSED (Match: e3b0c442)</text>
  <text x="40" y="300" class="code-text" fill="#38bdf8">8-9. Model &amp; Baseline Fitting: PASSED (n_estimators=50, max_depth=10)</text>
  <text x="40" y="325" class="code-text" fill="#38bdf8">10-11. Holdout Test Evaluation: PASSED (Test MAE: 291.40 veh/hr)</text>
  <text x="40" y="350" class="code-text" fill="#38bdf8">12-13. MLflow Sklearn Artifact &amp; Signature Logging: PASSED</text>
  <text x="40" y="375" class="code-text" fill="#34d399">14-15. Status Transition to SUCCEEDED &amp; Clean Termination: PASSED</text>
"""

# 4. Model Registry
c4 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Model Candidate Registry &amp; Release Gates</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <rect x="20" y="65" width="705" height="150" class="card" fill="#0f172a" />
  <text x="40" y="95" class="text-bold" font-size="16">Candidate: metro-traffic-volume (Version 2)</text>
  <rect x="580" y="80" width="120" height="24" class="badge-success" />
  <text x="640" y="96" class="badge-text" fill="#34d399">GATE PASSED</text>

  <text x="40" y="130" class="text-muted">VAL MAE: <tspan fill="#34d399" font-weight="700">284.15</tspan> (Threshold ≤ 350.0)</text>
  <text x="260" y="130" class="text-muted">VAL RMSE: <tspan fill="#34d399" font-weight="700">452.80</tspan> (Threshold ≤ 550.0)</text>
  <text x="480" y="130" class="text-muted">VAL R²: <tspan fill="#34d399" font-weight="700">0.948</tspan> (Threshold ≥ 0.90)</text>

  <text x="40" y="165" class="text-muted">Artifact URI: models:/metro-traffic-volume/2</text>
  <rect x="580" y="145" width="120" height="30" class="btn-blue" />
  <text x="640" y="165" class="btn-text">Stage Candidate</text>

  <rect x="20" y="235" width="705" height="180" class="card" fill="#0f172a" />
  <text x="40" y="265" class="text-bold" font-size="16">Candidate: metro-traffic-volume (Version 3 - Degraded)</text>
  <rect x="580" y="250" width="120" height="24" class="badge-danger" />
  <text x="640" y="266" class="badge-text" fill="#f87171">GATE FAILED</text>

  <text x="40" y="300" class="text-muted">VAL MAE: <tspan fill="#f87171" font-weight="700">999.00</tspan> (Violated ≤ 350.0)</text>
  <text x="260" y="300" class="text-muted">VAL RMSE: <tspan fill="#f87171" font-weight="700">1240.50</tspan> (Violated ≤ 550.0)</text>
  <text x="480" y="300" class="text-muted">VAL R²: <tspan fill="#f87171" font-weight="700">0.620</tspan> (Violated ≥ 0.90)</text>

  <text x="40" y="335" class="code-text" fill="#f87171">⛔ Rejection: Metrics failed strict release gate policy thresholds. Staging blocked.</text>
"""

# 5. Staging Smoke Checks
c5 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Staging Environment &amp; Smoke Validation</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <rect x="20" y="65" width="705" height="350" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="100" class="text-bold" font-size="16">Staging Deployment: dep-stg-002</text>
  <rect x="580" y="85" width="120" height="24" class="badge-blue" />
  <text x="640" y="101" class="badge-text" fill="#60a5fa">AWAITING APPROVAL</text>

  <text x="40" y="140" class="text-muted">Target Model Version:</text>
  <text x="220" y="140" class="code-text">metro-traffic-volume:2</text>

  <text x="40" y="175" class="text-muted">Smoke Test Result:</text>
  <text x="220" y="175" class="text-bold" fill="#34d399">PASSED (All 4 verification checks succeeded)</text>

  <line x1="40" y1="195" x2="685" y2="195" stroke="#1f2937" stroke-width="1" />

  <text x="40" y="225" class="code-text" fill="#34d399">✓ Model Artifact Deserialization: Loaded via MLflow sklearn loader</text>
  <text x="40" y="250" class="code-text" fill="#34d399">✓ Synthetic Input Feed: Shape (1, 8) valid</text>
  <text x="40" y="275" class="code-text" fill="#34d399">✓ Output Verification: 3,142.50 veh/hr (Finite numerical value)</text>
  <text x="40" y="300" class="code-text" fill="#34d399">✓ Cold Start Latency: 12.4 ms (Threshold ≤ 500 ms)</text>

  <rect x="440" y="350" width="110" height="32" rx="6" fill="#1e293b" stroke="#334155" stroke-width="1" />
  <text x="495" y="370" class="btn-text" fill="#f87171">Reject</text>

  <rect x="560" y="350" width="130" height="32" class="btn-blue" />
  <text x="625" y="370" class="btn-text">Approve Staging</text>
"""

# 6. Production Promotion
c6 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Production Environment &amp; Promotion History</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <rect x="20" y="65" width="705" height="170" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="100" class="text-bold" font-size="16">Active Production Champion</text>
  <rect x="580" y="85" width="120" height="24" class="badge-success" />
  <text x="640" y="101" class="badge-text" fill="#34d399">PROMOTED</text>

  <text x="40" y="135" class="text-muted">Model Name &amp; Version: <tspan fill="#f8fafc" font-weight="700">metro-traffic-volume : version 2</tspan></text>
  <text x="40" y="165" class="text-muted">MLflow Registered Alias: <tspan fill="#38bdf8" font-family="monospace">champion</tspan></text>
  <text x="40" y="195" class="text-muted">Approved By: <tspan fill="#94a3b8">lead_reviewer</tspan> | Rationale: <tspan fill="#94a3b8">Smoke tests passed; holdout MAE superior</tspan></text>

  <rect x="20" y="255" width="705" height="160" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="285" class="text-bold">Previous Production Champion (Ready for Instant Fallback)</text>
  <text x="40" y="315" class="text-muted">Model Version: <tspan fill="#f8fafc" font-weight="700">metro-traffic-volume : version 1</tspan></text>
  <text x="40" y="345" class="text-muted">Historical Performance: Val MAE 298.50 veh/hr</text>
  <rect x="540" y="340" width="160" height="32" rx="6" fill="#7f1d1d" stroke="#dc2626" stroke-width="1" />
  <text x="620" y="360" class="btn-text" fill="#fca5a5">Rollback to v1</text>
"""

# 7. Inference
c7 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Interactive Inference Serving &amp; Privacy Telemetry</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <rect x="20" y="65" width="345" height="350" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="35" y="95" class="text-bold">Request Payload</text>
  <text x="35" y="125" class="code-text">temp: 288.15 K (15°C)</text>
  <text x="35" y="150" class="code-text">rain_1h: 0.0 mm</text>
  <text x="35" y="175" class="code-text">clouds_all: 40%</text>
  <text x="35" y="200" class="code-text">weather_main: "Clouds"</text>
  <text x="35" y="225" class="code-text">weather_desc: "scattered clouds"</text>
  <text x="35" y="250" class="code-text">holiday: "None"</text>
  <text x="35" y="275" class="code-text">date_time: "2026-11-21 09:00:00"</text>

  <rect x="35" y="355" width="315" height="36" class="btn-blue" />
  <text x="192" y="377" class="btn-text">Predict Traffic Volume</text>

  <rect x="380" y="65" width="345" height="350" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="395" y="95" class="text-bold">Response &amp; Telemetry</text>
  <text x="395" y="130" class="text-muted">Inference ID:</text>
  <text x="395" y="150" class="code-text" font-size="11">e2e-inf-9a74-4b82-9012</text>

  <text x="395" y="185" class="text-muted">Predicted Traffic Volume:</text>
  <text x="395" y="220" class="text-val" fill="#34d399">3,142 veh/hr</text>

  <text x="395" y="255" class="text-muted">Serving Latency:</text>
  <text x="395" y="275" class="text-bold" fill="#38bdf8">3.8 ms (p50)</text>

  <text x="395" y="310" class="text-muted">Privacy Safeguard:</text>
  <text x="395" y="330" class="code-text" fill="#34d399">✓ Raw input features stripped from DB</text>
"""

# 8. Monitoring / Drift
c8 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Statistical Drift Analysis &amp; Covariate Shift</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <rect x="20" y="65" width="705" height="350" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="100" class="text-bold" font-size="16">Report: Input Feature Covariate Drift (Report #81)</text>
  <rect x="580" y="85" width="120" height="24" class="badge-amber" />
  <text x="640" y="101" class="badge-text" fill="#fbbf24">DRIFT DETECTED</text>

  <text x="40" y="140" class="text-muted">Statistical Test: <tspan fill="#f8fafc" font-weight="600">Kolmogorov-Smirnov (Continuous) &amp; PSI (Categorical)</tspan></text>
  <text x="40" y="165" class="text-muted">Reference Sample: <tspan fill="#f8fafc">48,204 rows</tspan> | Current Sample: <tspan fill="#f8fafc">1,250 rows</tspan></text>

  <line x1="40" y1="185" x2="685" y2="185" stroke="#1f2937" stroke-width="1" />

  <text x="40" y="215" class="text-bold">Feature Drift Breakdown:</text>
  <text x="40" y="245" class="code-text" fill="#f87171">⚠ temp: KS p-val = 1.4e-18 (D=0.84) &lt; 0.05 -- DRIFT DETECTED</text>
  <text x="40" y="275" class="code-text" fill="#34d399">✓ rain_1h: KS p-val = 0.42 (D=0.03) -- No drift</text>
  <text x="40" y="305" class="code-text" fill="#34d399">✓ clouds_all: KS p-val = 0.28 (D=0.04) -- No drift</text>
  <text x="40" y="335" class="code-text" fill="#34d399">✓ weather_main: PSI = 0.02 &lt; 0.20 -- Distribution stable</text>

  <rect x="40" y="360" width="645" height="40" rx="4" fill="#78350f" stroke="#d97706" stroke-width="1" />
  <text x="55" y="385" class="badge-text" fill="#fef3c7">ALERT TRIGGERED: Warning alert #12 created for on-call SRE review.</text>
"""

# 9. Feedback Quality
c9 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Delayed Feedback &amp; Rolling Model Quality</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <rect x="20" y="65" width="220" height="90" class="card" fill="#0f172a" />
  <text x="35" y="90" class="text-muted">EVALUATED MAE</text>
  <text x="35" y="125" class="text-val" fill="#34d399">288.4 veh/hr</text>
  <text x="35" y="145" class="text-muted" font-size="11">vs. 284.15 at train</text>

  <rect x="260" y="65" width="220" height="90" class="card" fill="#0f172a" />
  <text x="275" y="90" class="text-muted">EVALUATED RMSE</text>
  <text x="275" y="125" class="text-val" fill="#34d399">459.2 veh/hr</text>
  <text x="275" y="145" class="text-muted" font-size="11">Within +1.4% band</text>

  <rect x="500" y="65" width="225" height="90" class="card" fill="#0f172a" />
  <text x="515" y="90" class="text-muted">GROUND TRUTH COVERAGE</text>
  <text x="515" y="125" class="text-val" fill="#38bdf8">96.8%</text>
  <text x="515" y="145" class="text-muted" font-size="11">4,820 labeled events</text>

  <rect x="20" y="175" width="705" height="240" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="210" class="text-bold">Delayed Feedback Ingestion Integrity</text>
  <text x="40" y="240" class="text-muted">Roadside induction loop ground-truth ingestion guarantees:</text>
  <text x="40" y="270" class="code-text" fill="#34d399">✓ Idempotent re-submission on identical ground truth values</text>
  <text x="40" y="300" class="code-text" fill="#f87171">✓ FeedbackConflictError raised when contradictory actuals received</text>
  <text x="40" y="330" class="code-text" fill="#38bdf8">✓ Rolling 24-hour evaluation window prevents stale metric drift</text>
  <text x="40" y="360" class="code-text" fill="#38bdf8">✓ All metrics exported directly to Prometheus /metrics endpoint</text>
"""

# 10. Rollback & Audit
c10 = """
  <rect x="0" y="0" width="745" height="440" class="card" />
  <text x="20" y="30" class="text-bold">Audited Rollback Execution &amp; Audit Trail</text>
  <line x1="20" y1="45" x2="725" y2="45" stroke="#1f2937" stroke-width="1" />

  <rect x="20" y="65" width="705" height="130" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="95" class="text-bold" font-size="16">Emergency Rollback Executed</text>
  <rect x="580" y="80" width="120" height="24" class="badge-success" />
  <text x="640" y="96" class="badge-text" fill="#34d399">RESTORED</text>

  <text x="40" y="130" class="text-muted">Target Environment: <tspan fill="#f8fafc" font-weight="600">Production</tspan> | Restored Version: <tspan fill="#34d399" font-weight="700">v1</tspan></text>
  <text x="40" y="160" class="text-muted">Operator: <tspan fill="#94a3b8">on_call_sre</tspan> | Reason: <tspan fill="#94a3b8">Temperature covariate drift detected; rolling back to proven candidate</tspan></text>

  <rect x="20" y="215" width="705" height="200" rx="6" fill="#0f172a" stroke="#1e293b" stroke-width="1" />
  <text x="40" y="245" class="text-bold">Immutable Audit Trail (Last 4 Events)</text>

  <text x="40" y="280" class="code-text" fill="#f87171">[2026-11-21 17:34:02] ROLLBACK_DEPLOYMENT by on_call_sre (Restored v1 as champion)</text>
  <text x="40" y="310" class="code-text" fill="#60a5fa">[2026-11-21 17:33:55] PROMOTE_DEPLOYMENT by lead_reviewer (Activated candidate v2)</text>
  <text x="40" y="340" class="code-text" fill="#34d399">[2026-11-21 17:33:48] APPROVE_DEPLOYMENT by lead_reviewer (Smoke test passed)</text>
  <text x="40" y="370" class="code-text" fill="#fbbf24">[2026-11-21 17:33:30] STAGE_CANDIDATE by system (Candidate v2 into staging)</text>
"""

views = [
    ("01_overview_dashboard.svg", "Control Plane Overview", "Operational health, active champion and system status", c1),
    ("02_dataset_validation.svg", "Dataset Ingestion & Validation", "Immutable version snapshots and Pandera schema contracts", c2),
    ("03_training_mlflow.svg", "Training & Experiment Tracking", "Deterministic 15-stage lifecycle and MLflow experiment runs", c3),
    ("04_model_registry.svg", "Model Registry & Release Gates", "Quantitative candidate gating vs. baseline thresholds", c4),
    ("05_staging_smoke_checks.svg", "Staging Environment & Smoke Tests", "Automated pipeline loading and synthetic input verification", c5),
    ("06_production_promotion.svg", "Production Champion Promotion", "Two-phase human approval and MLflow alias assignment", c6),
    ("07_prediction_response.svg", "Inference Serving & Telemetry", "Real-time prediction with privacy-safe input stripping", c7),
    ("08_monitoring_drift_report.svg", "Continuous Drift Observability", "Kolmogorov-Smirnov feature shift and automated alerts", c8),
    ("09_feedback_quality_report.svg", "Delayed Ground-Truth Feedback", "Idempotent sensor actuals and rolling accuracy metrics", c9),
    ("10_rollback_and_audit.svg", "Zero-Downtime Rollback & Audit", "Instantaneous alias restoration and immutable event logs", c10),
]

for filename, title, subtitle, content in views:
    svg_data = svg_frame(title, subtitle, content)
    (OUT_DIR / filename).write_text(svg_data, encoding="utf-8")
    print(f"Generated {filename}")

print("All 10 SVG screenshots generated successfully.")
