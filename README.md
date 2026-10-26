# Ransomware AI Detection and Response Simulation

A Python research prototype for classifying normalized behavioral observations, scoring anomalies, and demonstrating a safe response workflow. Existing models are Random Forest, XGBoost, and Isolation Forest; FastAPI, Prometheus, Grafana, and Streamlit expose results.

**This is a lab simulation, not a production endpoint agent.** No ransomware is executed. No real process termination, host isolation, network blocking, or file encryption is implemented. Synthetic examples test pipeline behavior and cannot establish model accuracy or zero-day detection.

The [foundation audit](docs/FOUNDATION_AUDIT.md) records baseline failures, changes, verification results, and remaining work.

## Setup

Use Python 3.13 or 3.14. The pinned scientific dependencies require newer Python versions than the previous README's 3.10 claim; verification for this change used Python 3.14.7. CI covers 3.13 and 3.14.

```bash
git clone https://github.com/Wolfsenpai123/Ransomware_AI_Dectection.git
cd Ransomware_AI_Dectection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
export PYTHONPATH=src
python -m unittest discover -s tests -v
```

`httpx` in `requirements-dev.txt` supports FastAPI's test client. No new runtime dependency was introduced. Run commands from the repository root. `.env` is ignored by Git; Python services read exported environment variables rather than implicitly loading this file.

## Safe behavior pipeline

These commands generate CSV events only, aggregate windows, and demonstrate the three existing models:

```bash
python generate_safe_behavior_logs.py
python build_behavior_features.py
python train_behavior_models.py --allow-row-split
python mitre_mapping.py
streamlit run streamlit_app.py
```

Paths now agree: `data/raw/behavior/safe_behavior.csv` → `data/processed/behavior/behavior_features.csv` → `reports/behavior_model_metrics.csv` and `reports/behavior_scored_windows.csv`.

The generator creates one synthetic session containing benign and ransomware-like phases. Training without `--allow-row-split` deliberately fails because one session cannot support independent train/validation/test groups. The override explicitly ignores correlation metadata for a demonstration, still keeps identical feature rows together, and records the limitation in the report. Repeated copies of the same generated session are not independent validation data.

Metrics in the JSON training report describe the held-out partition under the selected split. All-window scores in the CSV include training observations and are marked exploratory. Detection lead time is left unavailable rather than estimated using those training observations. `if_score` is an anomaly score, not a malware probability; `iso_*` aliases remain available.

## Dataset ingestion and evaluation

Raw external datasets and trained artifacts are not shipped. `data/raw/dataset_registry.json` lists expected paths, not proof that data is installed. Inventory available files with:

```bash
python -m symptom_ai.dataset_ingestion.dataset_scanner
```

After obtaining authorized feature-only data and checking its labels/schema:

```bash
python -m symptom_ai.dataset_ingestion.parser_csu_ransomware
python -m symptom_ai.dataset_ingestion.parser_ransomset
python -m symptom_ai.dataset_ingestion.ingest_extended_available_datasets
python -m symptom_ai.models.train_symptom_models
python -m symptom_ai.models.build_training_reference
```

Missing parsers' inputs produce diagnostics; the unifier requires at least one input. Unrecognized, missing, conflicting, or generic malware labels remain `unknown`. They must be reviewed before binary ransomware training; the trainer rejects them. General malware is not automatically ransomware. The RISS adapter is disabled because its previous features were constructed from target labels. UGRansome's `Prediction` and `Threats` fields are not accepted as verified target/observation evidence.

Training uses an observation allowlist. It excludes source identifiers, labels, scenario fields, response policy, analyst/retraining flags, similarity outputs, and anomaly/unknown decision fields. Existing raw-feature proxy mappings remain approximate: PE sizes, API counts, and network statistics are not direct evidence of encryption, persistence, or C2.

Raw numeric proxy measurements use the fixed transform `x / (1 + x)` to avoid fitting normalization to held-out data. Missing observations default to zero; negative/nonfinite inputs fail validation. This proxy scale needs domain-specific replacement and validation before operational use. **Re-ingest data and retrain legacy artifacts**: schema version 2 is required by the API, and old min/max-normalized CSVs must not be mixed with regenerated data.

The default split joins all available shared source, session, source-file, family, sample-ID, and identical-feature relationships into disjoint groups. It searches deterministic group splits for both classes in train, validation, and test, and fails when independent groups are insufficient. Source grouping is deliberately conservative and can join whole datasets; missing provenance and near duplicates remain limitations. Reports record grouping columns, group IDs, seeds, partition sizes, and dataset SHA-256.

Random Forest versus XGBoost selection uses **validation macro F1**, followed by held-out test reporting. Reports include precision, recall, F1, accuracy, per-class support/results, confusion matrices, and benign false-positive rate. An unavailable FPR is `null`, not zero. Test class proportions are not artificially balanced before splitting.

Isolation Forest trains only on benign training observations. Its score threshold is calibrated using benign validation observations at a nominal 5% FPR, with calibration sample count and observed FPR reported. Small samples and tied scores limit calibration; test FPR may differ. Anomalies require investigation and do not establish malware identity or zero-day detection.

## API and response states

```bash
# Generate a local key; use the same exported value for the watcher/demo clients.
export SYMPTOM_AI_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn symptom_ai.api.main:app --host 127.0.0.1 --port 8000
```

`GET /health` is a liveness check; model readiness is separate. Missing/incompatible models return HTTP 503. Predictions require a nonempty `symptoms` dictionary containing recognized feature names and finite numeric values in `[0, 1]`; unknown names, coercible strings, booleans, extra request fields, and invalid ranges return 422.

`/predict`, `/health`, and model information support local read-only demonstrations. `/respond`, `/explain`, `/prepare-demo-docs`, `/unlock-demo`, `/cases/unknown`, and `/backups` require `X-API-Key`. Sensitive routes return 503 until `SYMPTOM_AI_API_KEY` is set, and 401 for an invalid key.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/respond \
  -H "X-API-Key: $SYMPTOM_AI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"symptoms":{"file_write_burst":0.9,"high_entropy_write":0.9,"backup_disable_attempt":0.9}}'
```

An isolated anomaly or rename symptom produces a review warning rather than automatic containment. Correlated file evidence or destructive-action indicators can recommend containment. A file-impact indicator in a log is an observation, not confirmation that ransomware damaged a real file.

Outputs distinguish `suspicious`, `impact_observed`, `containment_recommended`, `containment_executed`, `simulated_stop_requested`, and `simulated_containment_executed`. Actual containment remains false. The watcher calls `/predict`, writes a cooperative `STOP_SIGNAL.json` for qualifying simulated activity, and does not claim the generator has acknowledged the request. `/respond` may create lab backup copies and a pending case; its lockdown records which demo files would be locked without changing contents or permissions. `/unlock-demo` remains a compatibility operation without permission changes.

These defaults are for a trusted local lab. Shared API keys are not enterprise identity/RBAC; TLS, rate limits, deployment isolation, durable queues, artifact integrity, and operational auditing remain future work. Load only trusted local joblib artifacts.

## Live demonstrations and reviewed learning

With trained symptom artifacts and API running:

```bash
python -m symptom_ai.live_monitoring.live_log_ai_watcher --window-events 20 --poll 1 --reset-offset
# In another shell with the same environment:
python -m symptom_ai.live_monitoring.demo_scenario_runner --help
```

Benign, suspicious, known-like, and novel-pattern scenarios remain simulations. Novel/impact cases are saved as `pending_review`, with no confirmed label. Repeating a scenario does **not** automatically confirm ransomware or create an approved signature. Legacy automatically learned signature files are ignored. Approved signatures compare observable event rates and symptom magnitudes with multiple impact indicators; `scenario_type` and a few common event names cannot trigger a match by themselves.

Review a case locally using an explicit benign/ransomware label or rejection, analyst identity, and evidence notes:

```bash
python -m symptom_ai.auto_learning.review data/learning_queue/learning_case_EXAMPLE.json \
  --label benign --reviewer 'local-analyst' --notes 'Reviewed evidence and confirmed legitimate workload'
python -m symptom_ai.auto_learning.auto_retraining_pipeline
```

Approval records include an evidence digest; changes to events/symptoms invalidate approval. This local JSON workflow assumes a trusted reviewer and filesystem, not cryptographic reviewer authentication. Approved benign cases support false-positive correction. Pending, rejected, malformed, and legacy cases cannot enter candidate training.

Retraining creates a dataset, models, logs, and reports under a unique `models/versions/candidate_*` directory, preserving the subprocess environment and archiving active artifacts. It never overwrites the active dataset/models on success or failure. Candidate reports require manual review and promotion; no automatic production promotion/rollback mechanism is claimed. If a manual promotion later fails, stop serving and restore the complete matching archived artifact set before restarting. Include preprocessing/version metadata and references in that review.

## Monitoring

```bash
cp .env.example .env
# Set GRAFANA_ADMIN_USER and a unique GRAFANA_ADMIN_PASSWORD in .env.
docker compose --env-file .env -f monitoring/docker-compose.yml up -d
python -m symptom_ai.live_monitoring.grafana_prometheus_exporter --port 9108 --poll 1
```

Grafana (`127.0.0.1:3000`) and Prometheus (`127.0.0.1:9090`) are bound to loopback. Compose refuses unset Grafana credentials. Configure the Prometheus datasource and import `monitoring/grafana/ransomware_ai_dashboard.json`. Linux containers scrape the exporter via `host.docker.internal`; restrict host exporter access to the local/container lab network. Dashboards reflect simulation indicators, not independently verified infection or containment. Image versions/digests should be pinned and reviewed before a deployment.

Email alerts are optional and default to dry-run. Export the SMTP settings from `.env.example` only if intentionally using the notifier. Never commit `.env`, tokens, datasets, or malware samples.

## Scope and remaining work

This hardening task preserves the current model families and simulation workflow. It does not implement fileless malware detection, real endpoint collection, process termination, host isolation, or malware execution. Priority follow-up work is reviewed source schemas/labels, independent benign workloads and held-out families/sessions, realistic feature scales, calibration uncertainty, near-duplicate analysis, and a reviewed artifact release process. Enterprise agent functionality requires separate design and authorization.

## Integrated family, workload and ablation evaluation

PR #1's reports, figures and scripts remain available. The reports already in
`reports/evaluation/` and `reports/evaluation_v2/` are **historical legacy results**,
not verification of the schema-2 models. Their raw source datasets and original
artifacts are unavailable here. See [integration verification](docs/INTEGRATION_VERIFICATION.md)
for conflict resolution, actual checks and migration limits.

After re-ingesting reviewed source measurements, the within-source family protocol is:

```bash
python -m symptom_ai.models.prepare_heldout_family_v2 --family Akira
python -m symptom_ai.models.evaluate_heldout_family_v2 --family Akira
python -m symptom_ai.models.prepare_heldout_family_v2 --family LockBit
python -m symptom_ai.models.evaluate_heldout_family_v2 --family LockBit
python -m symptom_ai.models.build_heldout_family_summary
python -m symptom_ai.models.evaluate_isolation_forest_benign_only
python draw_figure_4_2.py
python draw_figure_4_3.py
```

This explicit protocol holds a named family out within one source; it does not
claim the default trainer's cross-source independence. Sessions, source files,
sample IDs, other shared families and exact observations cannot cross partitions.
Preparation manifests bind split CSV checksums; legacy or changed splits must be regenerated.
Benign fit/calibration/test groups must also be independent. The scripts refuse
insufficient provenance or groups rather than splitting correlated observations.
Selected class counts differ from deployment prevalence. Defaults require large
source datasets; use smaller counts only for an explicitly limited experiment.

Both evaluators share the observation allowlist and range checks. Offline IF
calibration uses the same threshold function as active training, with equality
at the threshold treated as an inlier. It writes a separate schema-2 shadow
candidate bundle and calibration report. The API checks feature **names**, scale,
schema, and the artifact/report checksum before emitting shadow telemetry.
Legacy or mismatched artifacts produce an explicit `unavailable` shadow status;
they cannot change `unknown_risk` or response policy. The active IF is already
benign-calibrated; `unknown_detector_mode` now describes that fact.

For log-only workloads, with the API running:

```bash
python -m symptom_ai.live_monitoring.benign_workload_runner \
  --scenario backup_like --duration 8 --delay 0 --reset-log
python -m symptom_ai.live_monitoring.ablation_evaluator \
  --truth benign --case-id backup_like --output-dir reports/benign_verification
```

Other workload profiles are `cloud_sync_like`, `compression_like` and
`software_update_like`. Run the ablation evaluator on each retained JSONL input
in `reports/evaluation/ablation_inputs/`, with its corresponding truth and case ID,
then run `symptom_ai.live_monitoring.build_ablation_summary`. The full-hybrid
metric counts actionable warnings/recommendations; an impact-only review or API
error does not count as successful detection. These scenario logs are simulations,
not an independent benign enterprise population or proof of pre-impact efficacy.
Schema-2 ablations name the model methods `classifier_only` and
`classifier_plus_isolation_forest`, and record `model_used` per window. This
correctly describes either validation-selected Random Forest or XGBoost.
The summary reader also supports historical XGBoost-specific result keys,
but refuses to mix schema versions.

The PR #1 analyst CLI remains available and delegates to the same evidence-bound
review used by retraining:

```bash
python -m symptom_ai.live_monitoring.analyst_review --list
python -m symptom_ai.live_monitoring.analyst_review \
  --case-file learning_case_EXAMPLE.json --decision approve \
  --analyst local-analyst --note 'Reviewed positive behavioral evidence'
```

Use `--decision benign` for a reviewed false positive and `reject` to revoke
approval. Both consumers immediately honor rejection or evidence changes. Legacy
`--signature-file` arguments are accepted for compatibility but no separate
signature database is written. Legacy approvals require re-review with explicit
notes; there is one approval authority. The watcher exposes schema 2, early alert
and impact timing, and shadow telemetry. Its legacy `containment_triggered` field
is always false; `sandbox_safety_stop` describes only the cooperative stop request.
Use `simulated_stop_requested` and `containment_recommended` for current consumers.

Run all demonstrations/evaluators against a disposable synthetic lab with:

```bash
PYTHONPATH=src python scripts/verify_integration.py --summary /tmp/integration-summary.json
```

This starts and stops its own localhost API and leaves active data/models/reports
untouched. `SYMPTOM_AI_API_BASE_URL` optionally selects a different API base URL
for watcher and ablation clients. The verification script uses synthetic fixtures
and validates interfaces only. Monitoring containers are configuration-checked;
container startup and dashboard rendering remain outside this verification.

## Detection-quality baseline (Task 02)

The offline baseline measures the current detector without deploying rule changes.
It separates **rule-only, classifier-only, anomaly-only and hybrid** results and
uses independently seeded synthetic endpoint sessions for six benign workloads:
backup, compression, updates, cloud sync, development tools and administration.
There are also three scripted positive profiles. No workload operation, malware,
response API, stop signal, host isolation or process termination is executed.

Read the [fixed protocol](docs/DETECTION_QUALITY_PROTOCOL.md),
[findings and verification](docs/DETECTION_QUALITY_BASELINE.md) and
[recorded synthetic report](reports/detection_quality_baseline/report.md).
The classifier's success on shared templates is not a real-world accuracy claim.
The hybrid's substantial false alarms are retained as baseline findings.

Use a fresh output directory for each reproducibility run:

```bash
export PYTHONPATH=src
python -m symptom_ai.benchmark.baseline prepare --output /tmp/quality-run --seed 20261027
python -m symptom_ai.benchmark.baseline calibrate --output /tmp/quality-run
# Inspect validation_diagnostics.json and frozen.json; no test-driven tuning.
python -m symptom_ai.benchmark.baseline evaluate --output /tmp/quality-run
```

`run` performs all three steps with the same defaults (30 sessions/profile,
eight simulated one-minute windows/session). Checksums bind the input partitions,
code, model, diagnostics and dependency versions. Calibration parses development
rows only. Recalibration and overwriting final results are refused. After reading
a final result, treat that holdout as retired for future optimization.

JSON reports include confusion counts, positive-class precision/recall/F1/FPR,
per-profile results, session-cluster bootstrap intervals, and explicit timing
availability. **Real endpoint-hour and latency fields remain null.** Simulated
rates use recorded simulated exposure, with one alert per alerting window; latency
uses scripted onset and window-end detection, including missed-session counts.
There is no incident deduplication or measured enterprise exposure.

No genuine reviewed external data is installed in this environment. The source
inventory records that limitation. Verify the ingestion/training contract with:

```bash
python scripts/verify_quality_baseline.py
```

This uses 15 explicitly synthetic raw fixture CSVs, fixed normalization,
unification, the actual grouped foundation trainer and API-compatible candidate
artifacts. It never reports those fixtures as genuine source performance.

To admit genuine feature data after source-specific mapping/schema/label review,
create a local JSON manifest with `schema_version: 2`,
`normalization: "fixed_nonnegative_saturation_v1"`,
`source_kind: "captured_feature_data"`, `reviewer`, `reviewed_at`, `review_notes`,
and a `sources` list. Each item needs `path` (relative to the manifest), `sha256`,
`adapter: "normalized_symptoms_v2"`, `provenance` and `label_definition`. Each CSV
must include unique `sample_id`, `dataset_source`, explicit reviewed `label`,
grouping provenance and recognized observation columns. Then run:

```bash
python -m symptom_ai.benchmark.baseline reviewed-sources \
  --review-manifest /path/to/reviewed-sources.json --output /tmp/reviewed-candidate
```

This creates an isolated unified dataset and training candidate. It refuses
unverified labels, changed inputs and insufficient independent groups. Source
presence and a local reviewer string do not authenticate reviewers or prove label
quality. Raw source adapters require their own schema review; no automatic download
or real malware execution is implemented. Identical feature names across fixed
raw-proxy transforms and watcher count-derived symptoms do not prove compatible
units or observation semantics. Candidate promotion remains manual.
