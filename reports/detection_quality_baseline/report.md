# Detection-quality baseline

Synthetic detection-quality baseline; no real-world performance claim

Protocol: prepare session holdouts → fit/train and diagnose on development data → freeze → evaluate test once.

| Method | TP | FP | TN | FN | Precision | Recall | F1 | FPR | False alerts / simulated benign endpoint-hour |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rule_only | 54 | 46 | 272 | 60 | 0.5400 | 0.4737 | 0.5047 | 0.1447 | 8.6792 |
| classifier_only | 114 | 0 | 318 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| anomaly_only | 34 | 18 | 300 | 80 | 0.6538 | 0.2982 | 0.4096 | 0.0566 | 3.3962 |
| hybrid | 79 | 261 | 57 | 35 | 0.2324 | 0.6930 | 0.3480 | 0.8208 | 49.2453 |

Real endpoint-hour and real latency fields are null. Rates above use simulated exposure and one alert per alerting window; no deduplication.

## Provenance

Generator configuration: `{"seed": 20261027, "sessions_per_profile": 30, "window_seconds": 60, "windows_per_session": 8}`.
Implementation SHA-256: `bd7275720f58e3463e366f9666563f97c9755be89af220e8214e8472907d98d0`.
Split manifest SHA-256: `9f51233a11f8661218fbeb865d67e56dcffbfc39553fc4b8432fb2b824be5e0e`.
Frozen decisions SHA-256: `e2cac5e2225672864e16416b30b4ac32ae1ae20fe462148e866d035451ad19d0`.
Captured external datasets were inventoried; none is automatically considered reviewed.

## Limits

- All observations, endpoints, timestamps and labels are generated; no captured endpoint exposure
- Independent random sessions share templates and a generator; no new family/source/site generalization
- Positive labels reflect scripted phase onset, not verified malware or fileless telemetry
- Window counts are correlated within sessions; intervals resample sessions
- Real endpoint-hour and latency metrics unavailable; simulated equivalents are explicitly named
- No live artifacts loaded, no reviewed signatures, no response actions or threshold changes deployed
- Known grouping cannot exclude near duplicates or unspecified real-world dependencies
- Once a final result is viewed it is exploratory for future changes; a future protocol needs new independent data
