# Task 02 evaluation protocol, version 1

This protocol establishes a reproducible **synthetic** baseline before endpoint
telemetry or fileless detection work. It is not a production accuracy study. The
current hybrid behavior is measured without changing deployed rules/thresholds.

## Development and final separation

- Exploratory implementation checks use seed `20261026` with 12 sessions/profile,
  and unit tests use separate fixture seeds. They are development data.
- The recorded final run uses seed **20261027**, **30 sessions per profile**,
  **8 consecutive 60-second windows per session**. The fixed seed/configuration,
  selection criterion and anomaly target are specified here before reading this
  run's final test results.
- Profiles: backup, compression, software updates, cloud sync, development tools,
  routine administration; three scripted positive profiles (known progressive,
  novel impact tokens and mixed legitimate/impact activity). All profiles share
  the same generator across partitions; this is not held-out family/source/site
  generalization. Synthetic labels express scenario intent, not malware evidence.
- Every session has an independently seeded activity rate, path pool, event mix
  and synthetic normalized scores, and its own endpoint. Independence is only
  conditional on the shared templates. No real workload operation is performed.
- A deterministic approximately 60/20/20 group split keeps whole endpoints,
  sessions, and connected exact-feature duplicates together. Splits require every
  profile in each partition; unsuccessful grouping fails rather than weakening
  the relationships. Profile counts can differ between partitions.
- Source grouping is deliberately excluded for this explicit within-generator
  study. The foundation trainer retains its stricter cross-source grouping.

## Fit, diagnose, freeze, evaluate

1. `prepare` writes separate train/validation/test JSONL files and a manifest with
   row/session/endpoint/profile counts, seeds, group IDs, implementation digest,
   external-source inventory and all file checksums.
2. `calibrate` parses train and validation only. It checks the final file checksum
   for integrity but does not parse its observations for fitting or selection.
   RF/XGBoost specifications come from the foundation trainer. Both classifiers
   fit training data; validation macro-F1 selects one (RF wins an exact tie).
   Classifier decisions use the existing class prediction, not a tuned cutoff.
   IF fits **benign training observations only**, with the foundation calibrator
   using benign validation scores at a fixed nominal **5% FPR**. Threshold equality
   is an inlier. No hybrid rule optimization is performed.
3. Validation-only diagnostics record signal combinations, false-alarm rules,
   alert types, active symptoms, training/validation ranges and single-feature
   median-replacement sensitivities. Reference medians come only from benign
   training observations. These counterfactual probes retain raw impact flags;
   correlated/off-manifold features make them sensitivity evidence, not causality.
4. The model bundle, configuration, selection/calibration results, validation
   diagnostics, dependency versions, implementation and data checksums are frozen
   before test evaluation. No model serving or active artifact promotion occurs.
5. `evaluate` checks the frozen inputs/code/dependencies and then opens test rows.
   It emits all four methods, per-profile counts, latency/censoring and cluster
   bootstrap intervals. It refuses existing predictions/reports and recalibration
   of frozen runs. These local-file checks prevent accidental reuse/tampering;
   they are not access controls against a trusted user rewriting local files.

## Detection and metric definitions

Primary methods are **rule-only**, **classifier-only**, **anomaly-only** and
**hybrid**, each with its own TP/FP/TN/FN, positive-class precision/recall/F1/FPR.
Zero denominators are `null`. The rule is the existing impact-event-count rule;
classifier output never enters anomaly-only results. The hybrid uses the existing
watcher/actionable-alert definition and response recommendation policy. A plain
impact review is not an actionable detection, even if the classifier/anomaly
signal is positive; that suppressive behavior is explicitly diagnosed. Learned
signatures are disabled so an external review database cannot change this run.

Each alerting window is one alert. There is no hidden incident deduplication or
cooldown. FPR uses benign windows, including the pre-onset windows of scripted
positive sessions. Session-level bootstrap intervals (500 deterministic replicates)
resample whole test sessions and are conditional on these templates. They do not
assume that windows are independent or quantify enterprise uncertainty.

Window starts/ends describe **simulated exposure**, not generator runtime. Real
`false_alerts_per_endpoint_hour` and `detection_latency_seconds` remain `null`.
Explicitly named simulated fields use FP / benign simulated endpoint-hours and
window-end alert time minus scripted attack onset. Overlapping endpoint exposure
is rejected. Missing/nonfinite times, gaps or unknown onset cannot establish
latency. Pre-onset alerts count as false positives; missed sessions are censored,
not assigned zero latency. Median/p95 latency is conditional on detection, with
eligible/detected/missed session counts always shown. Positive impact lead means
an alert before scripted impact, not evidence of stopping real damage.

## Source-data workflow

Registry presence is not review or approval. This environment currently has no
installed external datasets. No datasets are downloaded or executed by this task.
An explicit source-review manifest can admit checksummed, normalized symptom CSVs
with reviewed schema/labels, source provenance, reviewer identity/date and notes.
Missing/unknown/conflicting labels or insufficient independent groups fail clearly.
Source-specific raw mappings still need domain/schema review before admission.

A separate synthetic contract fixture verifies raw numeric observations → fixed
normalization → unified CSV → unchanged grouped RF/XGBoost/benign-IF training →
compatible API artifacts. It is tagged `synthetic_contract_fixture`, can never be
relabeled as captured data through its fixture adapter, and cannot establish real
accuracy. Captured feature datasets lack endpoint-hour/latency evidence unless
actual exposure/onset telemetry is supplied and reviewed separately.

## Holdout retirement

After publication, the recorded final test is **retired as an untouched holdout**.
It remains a reproducible regression baseline. Future feature/rule/threshold
experiments may diagnose it but must label such results exploratory and use newly
collected, reviewed, independent observations for confirmatory evaluation. Merely
changing a random seed of these same templates is not new real-world evidence.
All real process termination, host isolation and destructive actions stay disabled.
