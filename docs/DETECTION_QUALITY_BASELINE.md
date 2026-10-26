# Task 02 — Detection Quality Baseline

Branch: `feat/detection-quality-baseline`, based on integrated foundation commit
`af0634b`. The new PR targets `fix/foundation-hardening` while PR #2 is unmerged,
so the Task 02 delta can be reviewed separately. Main is not modified.

## What this establishes

A versioned, offline evaluation process with separate preparation, development-only
fitting/diagnostics, frozen decisions and final scoring. It measures the existing
hybrid policy without adjusting its thresholds against the final test. The
[protocol](DETECTION_QUALITY_PROTOCOL.md) specifies the recorded run in advance.

Only synthetic data is available. All 14 configured external datasets are absent.
Consequently **no genuine dataset accuracy, enterprise alert rate, real detection
latency, zero-day efficacy or fileless detection result is claimed**. No external
source was downloaded, and no real process/host/destructive action was enabled.

## Recorded experiment and provenance

- Seed `20261027`, generator version 1, 270 independently seeded endpoint sessions,
  30 per profile, eight nonoverlapping simulated 60-second windows/session.
- Six benign profiles: backup, compression, updates, sync, development tools and
  routine administration. Three positive profiles express known-like progressive,
  novel impact-token and mixed activity. Positive sessions have two benign warmup
  windows, followed by a scripted onset; impact begins later.
- Training: 162 sessions / 1,296 windows (990 benign, 306 scripted positive).
  Validation: 54 / 432 (312 benign, 120 positive).
  Final test: 54 / 432 (318 benign, 114 positive).
- Whole sessions/endpoints and connected exact observation duplicates stay in one
  partition. Source/template sharing is explicit: this is within-generator testing,
  not independent-family or source validation. Profile counts vary by group split.
- RF/XGBoost share fixed specifications with the foundation trainer. Validation
  macro-F1 selects Random Forest. IF fits benign training only; the fixed nominal
  5% calibration target produced 5.13% benign validation FPR. No final-test threshold
  search or deployed decision change was performed.
- [JSON report](../reports/detection_quality_baseline/report.json),
  [manifest](../reports/detection_quality_baseline/split_manifest.json),
  [frozen decisions](../reports/detection_quality_baseline/frozen.json) and
  [validation diagnostics](../reports/detection_quality_baseline/validation_diagnostics.json)
  retain checksums, actual package versions, feature names, groups and provenance.
  Generated input logs and model bundles are ignored; the documented commands
  regenerate them. All features use the current watcher aggregation; no metadata,
  scenario identifiers, labels, review flags or outcome columns enter the models.

## Why the earlier four-window check was not an accuracy baseline

The integration verifier trained on `synthetic_samples`: benign write values were
uniformly below 0.4 and positive write values above 0.5, with only write and entropy
features. It then passed live count-derived workload vectors into those artifacts.
A normal backup/update/sync write burst could therefore fall into the fixture's
scripted positive region or outside its IF benign range. That was an integration
contract check, not representative workload training. The previous local result
had classifier false alarms on three of four profiles; adding anomaly evidence
flagged all four. Hybrid ordinary-activity warnings provided another route.

The new benchmark trains and holds out *sessions from the same declared watcher
representation*, while retaining rule/anomaly/hybrid decomposition. Its different
sample population and feature coverage mean that a lower classifier FPR cannot
be interpreted as a measured real-world improvement over that four-window check.

## Final test results — synthetic only

| Method | TP | FP | TN | FN | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule-only | 54 | 46 | 272 | 60 | 0.5400 | 0.4737 | 0.5047 | 0.1447 |
| Classifier-only | 114 | 0 | 318 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Anomaly-only | 34 | 18 | 300 | 80 | 0.6538 | 0.2982 | 0.4096 | 0.0566 |
| Hybrid | 79 | 261 | 57 | 35 | 0.2324 | 0.6930 | 0.3480 | 0.8208 |

The classifier result reflects easy distinctions in shared scripted templates.
It is not evidence of perfect ransomware detection. The hybrid can underperform
both individual signals because its impact-review branch suppresses some model
signals while ordinary active symptoms independently trigger warnings.

Per-workload benign false alarms / complete test windows:

| Profile | Sessions | Windows | Rule | Classifier | Anomaly | Hybrid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Backup | 7 | 56 | 0 | 0 | 0 | 56 |
| Compression | 5 | 40 | 0 | 0 | 0 | 40 |
| Software updates | 5 | 40 | 0 | 0 | 0 | 39 |
| Cloud sync | 3 | 24 | 0 | 0 | 0 | 24 |
| Development tools | 7 | 56 | 46 | 0 | 13 | 2 |
| Routine administration | 8 | 64 | 0 | 0 | 5 | 63 |

The overall 318-window benign denominator also includes 38 pre-onset windows from
positive sessions; these count as benign false-positive opportunities. An alert
before scripted attack onset is never treated as successful attack detection.

There are 5.3 **simulated benign endpoint-hours** in the final partition. Counting
one alert per alerting window, rule/classifier/anomaly/hybrid false-alert rates are
8.68 / 0.00 / 3.40 / 49.25 per simulated benign endpoint-hour. **Real endpoint-hour
rates are null** because no captured exposure is available. There is no hidden
alert cooldown or incident deduplication.

Conditional median simulated latency is 180 seconds for rules and 60 seconds for
the other methods, at one-minute window resolution. Rules detected 14/19 positive
sessions, anomaly-only 18/19, and classifier/hybrid 19/19. Missed sessions remain
censored; they are never assigned zero latency. The perfect hybrid session hit
rate does not hide its 35 missed positive windows or 261 benign false alarms.
Real latency is unavailable. The JSON also records p95, impact lead and deterministic
95% session-bootstrap intervals; intervals are conditional on these templates.

## False-alarm contributors and actions

Findings below use the development partition and code inspection. Final counts
are reported above without using them to choose new thresholds.

1. **Ordinary activity becomes a warning.** `recommend_response` emits early warning
   for *any* active symptom at 0.7, including normal reads/writes and touched paths.
   Of 256 validation hybrid false alarms, 200 used `early_warning`, 26 used
   `isolate_and_backup` and 30 used `protective_lockdown`. These are recommendations,
   not executed actions. Future work should validate separate review/warning and
   high-confidence alert endpoints on development data with explicit costs.
2. **Path cardinality is mislabeled as impact.** `multi_directory_impact` counts
   unique paths, not directories or damaged files. It was active in 255/256
   validation hybrid false alarms; `user_document_impact_high` was active in 241.
   A regression demonstrates that 15 benign reads in one directory yield a maximum
   directory-impact symptom. Measure actual directory/operation scope and separate
   touched paths from verified impact; missing observations need explicit semantics.
3. **Saturated counts and overlapping signals are broad proxies.** File writes,
   storage spikes, entropy, mass modification and reads occur during legitimate
   workloads. Validation false alarms included active writes in 152 windows,
   reads in 108, storage spikes in 82 and mass modification in 75. Single-feature
   median-replacement probes report removed/introduced alarms, but correlation
   means they cannot justify dropping a feature or changing a threshold alone.
   Replacing process-spawn activity removed eight IF validation false alarms but
   only one hybrid false alarm; other ordinary active symptoms still triggered
   the hybrid. This is sensitivity evidence, not an approved detector change.
4. **Rename rules flag build tools.** Rule-only false alarms in validation came
   from development-tool windows with ordinary rename/write/process events. The
   existing rule counts rename events as impact and accepts three such events.
   Replace inferred damage with measured operation/evidence semantics before
   claiming infection. Use benign builds and bulk legitimate renames as challenge
   cases; do not add process-name allowlists as proof of safety.
5. **Network proxies conflate intent.** A periodic `network_beacon` token maps to
   both C2 and exfiltration symptoms; routine admin polling can therefore satisfy
   the network containment recommendation. Richer destination/process/context
   evidence is needed; event frequency alone cannot establish C2 or data theft.
   Entropy extras are also aggregated without checking whether the operation was
   actually a file write. Resolve those measurement meanings before endpoint work.
6. **IF errors depend on workload.** Its final false positives concentrate in
   development and administration, rather than all profiles. Validation diagnostics
   expose feature ranges and sensitivity instead of assuming every outlier is
   ransomware. Collect sufficiently diverse reviewed benign training/calibration
   sessions and report profile-specific FPR, sample sizes and domain shifts.
7. **Hybrid impact handling suppresses evidence.** `decide_alert_type` returns
   `file_impact_needs_review` before considering classifier/anomaly results when no
   containment policy is recommended. Actionable hybrid metrics exclude that state.
   Diagnostics found 52 validation windows where impact-only review suppressed
   classifier or anomaly evidence. This explains the low developer alert count
   despite frequent rule/anomaly hits.
   Review the intended distinction between a review case, warning, incident and
   recommendation using development evidence; do not silently redefine success.
8. **Feature names are not compatible units.** Fixed raw-proxy normalization and
   live count-based symptoms share names but differ in meaning. This baseline fits
   the watcher representation directly and never promotes its experimental bundle.
   Reviewed source mappings and future telemetry need explicit measurement/version
   contracts; apparent synthetic classifier success cannot validate that migration.

## Ingestion/training validation and limits

The feature-source workflow requires a checksummed source-review manifest with
schema/normalization, provenance, label definitions, reviewer/date and notes.
It unifies admitted observations and invokes the actual foundation grouped trainer
without changing active data/models. It refuses changed hashes, unsupported raw
adapters, unverified labels, duplicate/missing IDs or insufficient independent
groups. Local review identity is an assertion, not authenticated reviewer access.

With no genuine external data, `scripts/verify_quality_baseline.py` exercised
15 explicitly synthetic raw CSVs → fixed transform → 300 unified observations →
grouped RF/XGBoost/benign-only IF fitting → API-compatible serialization/inference.
The [source workflow summary](../reports/detection_quality_baseline/source_workflow_summary.json)
is marked `synthetic_contract_fixture`. Genuine schema/label/adapter correctness
and real source performance remain unvalidated. No raw malware binaries were
read or executed, and no source downloads were performed.

## Verification and reproducibility

- **67 automated tests passed locally** on Python 3.14.7, including the original
  52 foundation/integration tests plus 15 protocol regressions.
- The **39 existing isolated integration checks passed**, including safe demos,
  held-out fixture evaluation, calibration, plots, HTTP watcher and Compose-related
  workflows. No existing historical report was changed.
- Raw synthetic ingestion/training/API contract check passed. Dependency consistency,
  compilation, shell syntax and whitespace checks passed. Compose correctly refused
  empty Grafana credentials and accepted nonsecret configuration placeholders;
  containers and dashboards were not started.
- CI runs the tests, existing integration verifier, new source contract and quality
  protocol smoke check on Python 3.13 and 3.14. Remote outcomes are recorded in the PR.
- A second full run in a separate directory reproduced all three input JSONL
  files, the split manifest, frozen model bundle/decisions, validation diagnostics,
  final predictions and JSON report **byte-for-byte** in this dependency environment.
  This replay verifies reproducibility and did not select or retune any decision.
- Reproduce with the README's three commands in a new output directory. Checksums
  identify generated files, frozen models, code and dependencies; unit tests prove
  calibration parses only development rows. Model selection/calibration use no final
  test observations. The final partition is retired for future confirmatory tuning;
  changing a seed of the same generator does not create genuine new-world evidence.

Further progress requires reviewed captured benign workloads, independent endpoints
and collection sessions, verified attack timing and family identity, defensible
measurement units, temporal coverage/missingness and deployment-specific alert costs.
This baseline is the evaluation infrastructure and an honest failure analysis.
