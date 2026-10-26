# Task 03 synthetic development/validation results

These are policy-development observations from scripted data, not real-world improvement claims. Task 02’s published final holdout was not accessed. No external endpoint dataset or real containment was used.

## Reproduction and provenance

`PYTHONPATH=src python -m symptom_ai.benchmark.policy_development --output /tmp/new-policy-run --seed 310026 --sessions-per-profile 30`

Generator v1 creates 330 independent RNG sessions/endpoints and 2,640 simulated 60-second windows across eight benign and three scripted positive profiles. Grouping preserves endpoint/session/source-file relationships and exact duplicates. Development: 231 sessions, 1,848 windows (1,452 benign/396 positive). Validation: 99 sessions, 792 windows (648 benign/144 positive), including 48 benign warmup windows from positive-profile sessions. No final-test split exists. Shared generator templates remain a major limitation.

Fixed RF/XGB model specifications were fit on development, selected by validation macro-F1 (RF won a tie), and IF was fit on development benign windows and calibrated on 648 validation benign windows at nominal 5% FPR (observed 33/648 = 5.0926%; threshold -0.5896006078129729, strict score < threshold). Model selection/calibration makes these developmental metrics. The evidence-balanced default was chosen semantically, not as a score-selected winner. After an exploratory development/validation comparison, an invariant review added the shared requirement for specific context when models drive escalation. This reduced the sensitive alternative’s benign alerts from 33 to 16 on these development partitions. Policy logic has therefore been iterated on development/validation; no numeric threshold optimization or final-test access was performed.

The committed [protocol](protocol.json) and [JSON report](report.json) include session membership, source/code/input/model/prediction hashes and dependency versions. Full logs, model and predictions are reproducible under the ignored `artifacts/reviewed` directory. Two independent runs in the same environment produced byte-identical development/validation logs, protocol, model, predictions and JSON report.

## Separate detector evidence on validation

| Method | TP | FP | TN | FN | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anomaly_only | 35 | 33 | 615 | 109 | 0.5147 | 0.2431 | 0.3302 | 0.0509 |
| classifier_only | 144 | 0 | 648 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| rule_only | 96 | 0 | 648 | 48 | 1.0000 | 0.6667 | 0.8000 | 0.0000 |

The perfect classifier score reflects an easy shared-template problem (including explicit attack-like extension/entropy/impact assertions). It is not evidence of real malware detection. Anomaly-only evidence still produces 33 benign outliers and misses 109 positive windows.

## Policy alternatives on validation

A policy positive means **analyst alert or containment recommendation**. Evidence collection is reported separately; it is not silently counted as a successful detector or analyst alert.

| Policy | TP | FP | TN | FN | Precision | Recall | F1 | FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| analyst_sensitive | 144 | 16 | 632 | 0 | 0.9000 | 1.0000 | 0.9474 | 0.0247 |
| corroboration_only | 96 | 0 | 648 | 48 | 1.0000 | 0.6667 | 0.8000 | 0.0000 |
| evidence_balanced | 98 | 0 | 648 | 46 | 1.0000 | 0.6806 | 0.8099 | 0.0000 |

All alternatives share the same strict containment gate: 56 true-positive recommendation windows, 0 benign recommendations and 88 positive windows without a recommendation (window recall 38.89%). This gate is intentionally conservative. Only 14/24 positive sessions achieve a containment recommendation; all 10 ambiguous-impact sessions remain uncontained by recommendation. These are recommendations only.

The default emits 447 monitor, 247 evidence-collection, 42 analyst-alert and 56 containment-recommendation windows. **201/648 benign windows (31.02%) still enter evidence collection**, chiefly backup/compression entropy. Zero benign analyst alerts in this script does not mean zero benign operational overhead. The sensitive alternative escalates 16/33 benign IF outliers that also have specific high-entropy context: compression 15 and backup 1. Other generic-activity outliers stay at evidence collection, even with positive model signals. The remaining false analyst alerts demonstrate that high entropy is legitimate in benign workloads.

## Default stages by workload

| Profile | Windows | Monitor | Evidence collection | Analyst alert | Containment recommendation |
| --- | ---: | ---: | ---: | ---: | ---: |
| administration | 80 | 80 | 0 | 0 | 0 |
| ambiguous_impact | 80 | 19 | 21 | 40 | 0 |
| backup | 56 | 0 | 56 | 0 | 0 |
| build_tools | 88 | 85 | 3 | 0 | 0 |
| bulk_file_operations | 32 | 32 | 0 | 0 | 0 |
| cloud_sync | 48 | 41 | 7 | 0 | 0 |
| compression | 128 | 0 | 128 | 0 | 0 |
| destructive_with_note | 80 | 19 | 20 | 1 | 40 |
| destructive_with_recovery_tamper | 32 | 8 | 7 | 1 | 16 |
| network_polling | 64 | 64 | 0 | 0 | 0 |
| software_update | 104 | 99 | 5 | 0 | 0 |

Profile rows include warmup windows; positive-profile totals are not attack-only denominators. Confusion counts above use explicit per-window labels.

## Alert-rate and latency tradeoffs

Validation contains 10.8 simulated benign endpoint-hours. The sensitive policy produces 1.4815 and anomaly evidence 3.0556 false alert windows per **simulated** benign endpoint-hour; default and conservative analyst-alert rates are zero in this script. One alerting window is one count, with no cooldown or incident deduplication. Real endpoint-hour fields remain null.

Default/conservative policies detect 24/24 scripted positive sessions, but conditional median latency is 180 seconds from scripted onset, 60 seconds after scripted impact. Sensitive policy detects 24/24 at median 60 seconds, 60 seconds before impact, with the stated false alerts. Default still misses 46/144 positive windows, mostly pre-impact. Containment recommendations detect 14/24 sessions at median 180 seconds; 10 misses are censored. All latency is simulated, measured at window ends; real latency fields remain null.

## Actionable findings and limits

- Keep high-volume reads, touches, writes, builds and polling as activity, with contextual evidence collection rather than automatic analyst escalation. Distinct impact directories are measured separately from observed paths/directories.
- High archive entropy legitimately persists in backup/compression. Future work needs trusted file-operation context and before/after content evidence; suppressing this warning by tuning on the published holdout would not establish quality.
- IF’s calibrated validation outliers are not malware labels. Letting a lone outlier create an analyst alert buys earlier scripted recall at a visible benign workload cost.
- Model consensus is not independent impact corroboration. Preserve original model hits during review; require separately referenced, attributed impact evidence for recommendations.
- Corroboration, confidence and labels here are scripted. Sensor authentication, confidence calibration, family/source generalization, symlink/volume aliases, delayed/out-of-order events, cross-window persistence and telemetry loss remain unresolved.
- There is no real Sysmon collector, raw-event adapter, file-content measurement or reviewed external v3 endpoint corpus. Existing v2 model outputs are shadow-only for active v3 policy; no candidate was promoted.
- These new features and templates differ from Task 02. The 82.1% historical hybrid FPR is not a directly comparable control, and no numeric real-world improvement is asserted. A new genuinely independent dataset and preregistered holdout are needed for confirmation.
- All real termination, host isolation and destructive response remain disabled. The evaluation invokes no response APIs or sandbox stops.

## Verification

All 94 unit/integration regression tests passed locally (67 prior tests retained with obsolete semantic assertions/fixtures updated, plus 27 new regressions). The 39 isolated safe integration checks and reviewed-source ingestion/training contract passed. The historical quality-protocol smoke check, dependency consistency, compilation, shell syntax, whitespace and Compose credential/configuration checks passed. CI results are recorded in the draft PR after execution.

See [semantics and policy details](../../docs/DETECTION_SEMANTICS_POLICY.md) and [telemetry schema](../../docs/TELEMETRY_EVENT_SCHEMA.md).
