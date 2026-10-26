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
