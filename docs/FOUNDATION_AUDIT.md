# Foundation hardening audit — TASK 01

Repository baseline: `9b158b6f72a54ce76f3ab9de8177c5c2f3ec85eb` (use `git rev-parse main` for the full baseline revision). Work branch: `fix/foundation-hardening`. Local verification: October 26, 2026; Python 3.14.7, scikit-learn 1.8.0, pandas 3.0.3, XGBoost 3.4.1. Main was not modified.

## Scope and baseline

Reviewed the training and ingestion pipelines, feature/schema definitions, prediction and explanation consumers, watcher, learning pipeline, response engine, safe generators/replay scripts, notification/exporter, dashboards, shell helpers, requirements, and monitoring configuration. Python syntax/definition inspection covered 51 original Python files. No raw malware datasets or pretrained symptom artifacts were present. No ransomware, process termination, host isolation, or live malware execution was performed. Existing safe demos ran only against temporary lab folders.

A clean export of `main` was used to reproduce failures independently of working changes. `pip install -r requirements.txt` succeeded in a fresh `.venv`; the original requirements were retained. `httpx` was added only as a development dependency for FastAPI tests.

| Baseline command/check | Actual baseline outcome |
| --- | --- |
| `python -m unittest discover -v` | No tests existed: 0 tests, exit 5 on Python 3.14. |
| `python generate_safe_behavior_logs.py` | Exit 1: output directory `data/raw/behavior_logs` did not exist. |
| `python build_behavior_features.py` | Exit 1: looked for a different path, `data/raw/behavior/safe_behavior.csv`. |
| `python train_behavior_models.py` | Exit 1: missing processed behavior features. |
| `python -m symptom_ai.models.train_symptom_models` | Exit 1: unified dataset absent. |
| `python -m symptom_ai.dataset_ingestion.ingest_extended_available_datasets` | Missing external datasets followed by unhandled `ValueError: No objects to concatenate`. |
| Dataset inventory | All 14 configured external source paths absent; two registry statuses nevertheless claimed `imported`. |
| API/monitoring review | Sensitive routes lacked authentication; Grafana used hard-coded `admin/admin` credentials. |

## Confirmed issues and fixes

| Area | Confirmed bug/risk | Change and practical effect |
| --- | --- | --- |
| Splitting | Random row splits allowed correlated windows/samples into both sets. | Shared evaluation joins source, session, source-file, family, sample IDs, and identical observations into connected groups. Train/validation/test are disjoint. Insufficient provenance/groups fail explicitly; a named demonstration override records its limitations. |
| Model selection | Primary classifier chosen by test accuracy, then the same test score reported. | Select using validation macro F1; report held-out test results afterward. Natural holdout class distribution is preserved. |
| Metrics | No consistent FPR or per-class results across pipelines; all-window scoring was used for a lead-time claim. | Shared precision/recall/F1, confusion matrices, per-class support and FPR, benign FPR, accuracy, seeds/partition IDs and dataset fingerprint. All-window scores marked exploratory; training-contaminated lead time removed. |
| Anomaly baseline | Symptom Isolation Forest trained on mixed benign/malicious data; behavior baseline used an uncalibrated 20% contamination assumption. | Benign-only fitting; threshold calibrated on benign validation at nominal 5% FPR, with actual calibration count/FPR and test metrics. Calibrated offset survives joblib serialization and existing prediction interfaces. |
| Feature leakage | Every numeric column could be a model input, including review/retraining, similarity and anomaly outcome flags. RISS symptoms were mathematical functions of the target. | Observation allowlist shared by training and references. RISS ingestion disabled until genuine observation mapping is available. Scenario IDs, outcomes, labels and metadata excluded. UGRansome `Threats`/`Prediction` no longer serve as observation/verified target. |
| Preprocessing | Normalization used extrema from the complete source before splitting, allowing holdout information into features; sparse CSU inputs could call `.fillna` on a scalar. | Fixed nonnegative `x/(1+x)` transform avoids fitted extrema, handles scalar defaults, and rejects negative/nonfinite measurements. Sparse CSU columns align to rows; unavailable inverse extension features stay absent rather than suspicious. Schema migration requires re-ingestion and retraining. |
| Labels | Missing RansomSet binary labels used `fillna(1)`; several adapters classified all unknown/default/generic malware as ransomware. | Explicit text and verified binary mappings; unknown, conflicting or missing labels remain `unknown`. Training refuses these until reviewed. Generic malware labels do not establish ransomware. |
| Unification | Missing metadata could become numeric zero; empty extended ingestion crashed. | Missing observation columns can default to zero while metadata stays missing. Empty/missing-label inputs get actionable errors. Duplicate shadowed SILRAD/Android definitions removed. |
| Behavior data | Generator, builder, trainer, Streamlit and MITRE disagreed on file paths/columns. Misordered service-stop arguments corrupted label/family values. Window grouping could mix sessions. | Paths and `if_*` score columns aligned, `iso_*` aliases retained, output parents created, service-stop argument corrected, session IDs preserved, validation and per-session windows added. Importing generator/builder/trainer no longer runs those pipelines. |
| Automatic learning | File-impact observations were assigned confirmed ransomware labels, immediately reused as signatures and appended to active training data. | Cases start pending without confirmed labels. Local review requires explicit benign/positive/rejected decision, reviewer and evidence notes, bound to an evidence digest. Unapproved/legacy/rejected/malformed cases cannot train. |
| Signatures | Sharing a `scenario_type`, or three common event names, could trigger a malware decision. | Legacy automatic signature files ignored. Only reviewed positive cases can match; matching checks event-rate and symptom distances with multiple impact indicators. Scenario names are not evidence. |
| Retraining | Subprocess discarded the entire environment. Training temporarily overwrote the production CSV; partial model failures had no artifact rollback. | Preserve environment, set absolute package path, run the current interpreter with timeout, train under a unique candidate directory, archive active joblib/metadata. Success and failure both leave active data/models intact. Repeated reviewed rows deduplicate by stable evidence ID; base sample IDs must be unique/nonmissing. |
| Risk/false positives | Normalizing only by active symptom weights made one strong ordinary symptom score near 1. Any anomaly/rename could demand lockdown. | Use a fixed denominator and explicit correlated/destructive evidence for containment recommendations. An isolated anomaly or rename remains a warning/review case. Heuristic scores are not calibrated probabilities. |
| Response truthfulness | Alert names claimed infection or containment before an action/acknowledgement occurred. | Separate suspicion, impact indicators, recommendations, actual execution, simulation requests and simulation results. Actual containment remains false; STOP_SIGNAL is a cooperative request, not proof of generator acknowledgement. |
| Safe actions | Lockdown changed file permissions; progressive lock simulation deleted original dummy documents. | Lockdown records simulated locks without permission/content changes. Progressive simulation creates separate markers and preserves originals; external/symlink files are rejected in that path. Backups/case records remain lab operations. |
| API | Unauthenticated state changes, unbounded/unvalidated symptom dictionaries, missing-model errors. A `NaN` validation error itself caused a 500. | Environment key with constant-time comparison on sensitive routes, bounded recognized feature names and finite `[0,1]` values, 422 input errors that do not echo nonfinite data, 503 for absent/incompatible model artifacts. Public local liveness/prediction retained. |
| Monitoring | Default Grafana credentials and externally published container ports; stale exporter label series survived later windows. | Required environment credentials, loopback Grafana/Prometheus ports, stale host/evidence series cleared and host cardinality bounded, new alert states recognized. Registry stops claiming absent data is imported. |
| Documentation | Claims implied automatic confirmed learning/zero-day detection and real containment; quickstart paths and supported Python misleading. | README rewritten around actual simulation, evaluation protocol, review flow, migration, setup and deployment limits. Generated summaries and UI labels explain exploratory results and recommendations. |

## Verification results

Commands were run locally after implementation; synthetic data is strictly for pipeline verification.

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` with `PYTHONPATH=src` | **37 tests passed**. Includes actual Random Forest/XGBoost/Isolation Forest training, serialization and FastAPI inference using 300 synthetic observations from 15 synthetic source groups. |
| `python -m compileall -q src tests *.py` | Passed. |
| `python -m pip check` | Passed: no broken requirements. |
| `git diff --check` | Passed. |
| Generate → build → behavior train with `--allow-row-split` → MITRE | Passed: 2,700 safe CSV events, 54 windows, all three model artifacts and expected CSV/JSON reports. Metrics are deliberately not presented as real detection accuracy. |
| Behavior train without override on that single session | Expected validation failure: insufficient independent groups. |
| Missing external datasets | Inventory reports absent paths; extended unifier exits with an actionable validation message rather than a concat traceback. |
| `python -m symptom_ai.safe_demo` and `python -m symptom_ai.unknown_demo` | Both exited 0 using temporary lab folders. |
| Compose with empty Grafana credentials | Expected configuration failure. |
| Compose with nonsecret validation placeholders | `docker compose ... config --quiet` passed. Container services were not launched; Grafana dashboard rendering and container scrape connectivity remain unverified. |
| Retraining success/failure | Regression tests simulate subprocess outcomes and verify original dataset/model bytes remain unchanged, inherited environment is retained, and candidate state is explicit. No external raw dataset training or production promotion performed. |

The suite covers normalization, feature extraction, malformed/missing labels and values, session grouping, duplicate/family relationships, reproducible holdouts, FPR reporting, benign calibration, decision/false-positive behavior, approval/rejection/tampering, scenario-independent signatures, stop-signal state, real synthetic model inference, API authentication/validation/readiness, and non-destructive simulation behavior. One upstream Starlette warning recommends a future `httpx2` test-client migration; tests pass with the existing `httpx` interface.

CI adds Python 3.13/3.14 dependency checks, compilation, tests, and monitoring configuration validation. Local results are recorded above; remote workflow status is available on the pull request.

## Files changed and why

- `src/symptom_ai/models/evaluation.py` (new): central feature boundary, group partitions, metrics, benign-only calibration. Both trainers use it; reference builder and demo prediction decoding stay compatible.
- `src/symptom_ai/dataset_ingestion/validation.py` (new), existing ingestion/parser/unifier modules, and registry: explicit label contracts, normalization, sparse schema handling, honest missing-source state, removal of target-derived features and duplicate definitions.
- Root generator/builder/trainer, `mitre_mapping.py`, `streamlit_app.py`: working safe pipeline paths, validated per-session features, consistent score outputs and honest UI/lead-time behavior.
- `src/symptom_ai/auto_learning/review.py` (new), retraining pipeline and watcher: evidence-bound local review, pending cases, reviewed matching and isolated candidates. Existing subprocess execution uses inherited environment.
- Risk scorer, response policy and protective lockdown: fixed risk denominator, corroborated recommendations, safe recorded lockdown and pending cases. Progressive simulation preserves originals; sandbox/progressive API clients send the configured key.
- API, `.env.example`, Compose, exporter and notifier: input validation/readiness/authentication, environment credentials, restricted container ports, correct telemetry states and stale-series cleanup.
- Reporting summary and `check_demo_result.sh`: consume the corrected result schema and describe recommendations honestly.
- `tests/test_foundation.py`, `requirements-dev.txt`, `.github/workflows/tests.yml`: regression coverage, justified test dependency and CI.
- `README.md`, this audit and `.gitignore`: reproducible setup, migration, review instructions, limitations and generated-output exclusions. Removed the tracked `.pyc` cache; source interfaces remain the maintained artifact.

## Limitations and next steps

1. **Dataset quality is unverified.** No external raw datasets were installed. Confirm each adapter's source schema, binary conventions, family labels and licensing. Review generic malware separately from ransomware, and replace coarse PE/API/network symptom proxies with defensible mappings. Missing observations remain indistinguishable from observed absence.
2. **Provenance can be insufficient.** Source grouping intentionally holds entire datasets together, and shared families/identical features may connect most rows. Two sources or one session may be insufficient for three independent partitions. Collect independent sessions/families/sources; never silently weaken groups to obtain better scores. Near duplicates, incomplete identity metadata and domain shift require further analysis.
3. **No real accuracy or early-detection claim.** Synthetic results only establish code behavior. Some allowed inputs are impact observations; retrospective classification is not proof of detection before damage. Evaluate independent benign workloads and held-out families with temporal protocols, uncertainty intervals and deployment-specific costs/FPR targets. Isolation Forest anomalies are not confirmed zero-day ransomware.
4. **Feature migration is mandatory.** Re-ingest original measurements using the new fixed transform and retrain; legacy model/CSV scales are incompatible. A schema-version gate cannot establish the provenance of a manually supplied CSV. More suitable physical scales or train-fitted transforms need separate validation.
5. **Review and artifact release are local workflows.** JSON reviewer identity/digests detect accidental evidence changes but are not cryptographic approval, role enforcement, or durable transaction logs. Candidate promotion remains manual. The pipeline avoids active artifact mutation, so failed training needs no active rollback; stop serving and restore the full archived artifact set if a later manual release fails. Signed manifests, validation gates, atomic releases and tested operational rollback remain future work.
6. **Lab security defaults are limited.** Shared key authentication is not RBAC. Public prediction can consume compute; add deployment-level TLS, identity, access controls, rate limits and resource limits before any shared exposure. Only load trusted local joblib models. Container images still need deployment-specific version/digest pinning and vulnerability review; the exporter is reachable for container scraping and needs host firewall restrictions.
7. **Simulation compatibility is intentional.** Existing alert names that asserted infection/containment are replaced with review/recommendation names; consumers should use explicit state fields. Repeating the novel scenario no longer auto-learns it. Lock/unlock no longer change permissions, and progressive markers preserve originals. Update downstream presentation expectations accordingly.
8. **No new endpoint features.** Fileless telemetry, real process/host/network enforcement, production EDR integrations and additional model architectures were deliberately outside this task.
