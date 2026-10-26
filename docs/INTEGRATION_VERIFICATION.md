# PR #1 / PR #2 integration verification

Integration date: October 26, 2026. Branch: `fix/foundation-hardening`.
Original foundation head: `d12277ed9b485a77d615f3762ec8074b92e1e9bc`.
Fetched main/merged PR #1: `9508b60c4b0c7a3a653f29d4bd75fa926ceffbda`.
Common ancestor: `9b158b6f72a54ce76f3ab9de8177c5c2f3ec85eb`.

This is an ordinary merge of `origin/main` into the foundation branch. Neither
main nor published branch history was rewritten. PR #2 remains draft and must
not be merged before final review. The earlier [foundation audit](FOUNDATION_AUDIT.md)
continues to describe PR #2's security fixes; this document records integration.

## Conflicts and reconciliation

Git reported content conflicts in exactly two files:

| File | Resolution |
| --- | --- |
| `.gitignore` | Union of foundation outputs and PR #1's source snapshots, large split CSVs and preliminary report exclusions. |
| `live_log_ai_watcher.py` | Keep foundation pending cases, evidence-bound approvals, observation-based signature matching, authenticated explanation calls, robust event reading, and recommendation/execution separation. Integrate PR #1's timing metrics, calibrated-shadow telemetry and sandbox stop metadata. |

Additional semantic conflicts required changes despite a clean textual merge:

- **One review authority.** PR #1's `analyst_review` CLI delegates to
  `auto_learning.review`. Explicit positive, benign and rejected decisions require
  identity and evidence notes. Signatures and candidate training read the same
  digest-bound case. Revocation or evidence changes invalidate both consumers.
  Separate legacy signature files and old approval flags cannot bypass review.
  `--signature-file` remains an accepted compatibility argument without creating a
  second database. Existing approved records require explicit re-review.
- **One active detection path.** CLI prediction delegates to API inference with
  the same schema and artifact checks. The active IF is already benign-only and
  validation-calibrated. PR #1's supplementary IF remains shadow evidence;
  unavailable or anomalous shadow output cannot change risk or policy.
- **Shared calibration.** Offline and active IF training call the same lower-tail
  calibration function. Equality is an inlier, including tied scores. Calibration
  reports record sample counts, observed FPR and threshold convention. The shadow
  bundle and report bind schema, normalization, ordered feature names and checksum;
  equal feature counts alone are insufficient. The serialized shadow model also
  retains the default nominal 5% calibration offset.
- **Evaluation compatibility.** Both family and IF evaluators use the foundation
  observation allowlist and value checks. Preparation manifests bind CSV checksums;
  legacy/changed splits fail clearly. Within-source family evaluation explicitly
  excludes source grouping, while preserving session, file, sample, family and
  exact-observation separation. Default foundation training retains its stricter
  cross-source grouping. Positive family identities, duplicate IDs and missing
  provenance are checked. No test-set model selection was added.
- **Truthful responses and timing.** Current alerts use schema 2 and retain
  `early_warning_known_ransomware`. Impact review alone and API failures do not
  count as successful hybrid detection. Early warnings do not stop generators.
  Qualifying simulated impact may request a cooperative stop; actual containment
  and generator acknowledgement remain false. Legacy `containment_triggered` is
  false, while `sandbox_safety_stop` describes a requested sandbox stop. Existing
  stop files are preserved. Affected-path timing counts impact paths only.
- **Classifier ablations.** Schema-2 methods are `classifier_only` and
  `classifier_plus_isolation_forest`, with the active validation-selected model
  recorded per window. They no longer incorrectly imply that every active model
  is XGBoost. Summary readers support legacy keys and refuse mixed schema versions.
- **Monitoring and clients.** Exporter/notifier recognize the retained known-family
  early-warning state. Watcher/ablation and progressive demo clients accept
  `SYMPTOM_AI_API_BASE_URL`; sensitive API calls retain the foundation key.

## Preserved work

PR #2's ingestion/label safeguards, fixed transforms, grouped training, validation
selection, benign calibration, API authentication/input checks, isolated candidate
retraining, safe simulation behavior and monitoring configuration remain in place.
All original 37 foundation tests remain.

PR #1's held-out family preparation/evaluation, benign workload generators,
scenario ablations, analyst-approved learned-signature capability, calibration
tradeoff experiment, shadow telemetry, figures, summaries and timing metrics remain
available. All tracked files under `reports/evaluation/` and
`reports/evaluation_v2/` are **byte-for-byte identical to fetched main**. Historical
summary readers and figure generators remain usable. Historical scores are not
new schema-2 verification results. See [report provenance](../reports/README.md).

## Actual local verification

Python 3.14.7; installed requirements retained. No external malware execution,
real host isolation, real encryption, production model promotion, or SMTP delivery
was performed. Demonstrations ran in temporary lab directories. The verifier
starts/stops only its own localhost API; active repository data/models and tracked
reports are untouched.

| Check | Actual outcome |
| --- | --- |
| `PYTHONPATH=src python -m unittest discover -s tests -v` | **52 passed**: original 37 plus 15 integration regressions. Actual RF/XGBoost/IF fitting, serialization, family evaluation and FastAPI inference use synthetic fixtures. |
| `PYTHONPATH=src python scripts/verify_integration.py --summary /tmp/integration-summary.json` | **39 checks passed**. 37 commands exited 0; two expected validation refusals exited 1. |
| Safe CSV generation → features → demonstration training → MITRE | Passed in a disposable lab; single-session training without the explicit override correctly refused. |
| Existing `safe_demo`, `unknown_demo`, and all four log scenario profiles | Passed. |
| Progressive known/unknown demos over authenticated HTTP | Passed; all 72 original dummy documents remained in each disposable lab. |
| Held-out Akira/LockBit tools and family summary | Passed with 900 synthetic rows, two observation features, 100 selected training rows per class and 30 test rows per class. These are interface checks, not source-dataset metrics. |
| Benign-only IF experiment | Passed on synthetic prepared splits: 60 benign fit, 40 calibration, 30 benign test, 30 Akira and 30 LockBit observations; all four nominal FPR thresholds emitted. |
| Both figure scripts | Passed with regenerated synthetic reports; existing tracked PNGs unchanged. |
| Symptom trainer → references → CLI prediction | Passed using 300 synthetic observations across 15 source groups and schema-2 artifacts. |
| Three retained scenario-input ablations and summary | Passed over the disposable HTTP API, using the actual API outputs. |
| Four benign workload generators and benchmarks | Passed; each produced one complete 20-event evaluation window. Actual false-alarm counts are below. |
| Watcher → HTTP prediction + authenticated explanation | Passed with available calibrated shadow telemetry and a qualifying simulated stop request; no executed containment claimed. |
| Existing timeline shell helper | Passed against the disposable watcher output. |
| Legacy ablation/family summary readers | Passed on temporary copies of original tracked reports. No historical report overwritten. |
| Dependency consistency, compilation, shell syntax | `pip check`, `compileall` for source/tests/scripts/root modules and `bash -n` passed. |
| Git conflict/whitespace checks | No unresolved entries or markers. Diff against main passed whitespace checks. PR #1 CSVs retain CRLF bytes; combined-parent checking passes with `core.whitespace=cr-at-eol`. |
| Compose configuration | Missing Grafana credentials correctly refused; nonsecret placeholders passed `docker compose ... config --quiet`. Services were not launched. |
| External dataset inventory | All 14 configured external datasets absent. Default family preparation correctly refused the missing real dataset. |

Regression coverage includes shared positive/benign/rejected review, evidence
changes, legacy approval bypasses, reviewed-match stop requests, early warning and
impact timing, API failure handling, actionable ablation semantics, missing split
columns, correlated/duplicate partitions, invalid labels/ranges/IDs, legacy and
tampered split manifests, threshold ties, schema/name/checksum incompatibilities,
and shadow alarms remaining independent of active risk/response.

Actual synthetic benchmark outcomes from the local run:

| Benign profile | Windows | Rule false alarms | Classifier false alarms | Classifier + IF false alarms | Hybrid false alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Backup-like | 1 | 0 | 1 | 1 | 1 |
| Cloud-sync-like | 1 | 0 | 1 | 1 | 1 |
| Compression-like | 1 | 0 | 0 | 1 | 1 |
| Software-update-like | 1 | 0 | 1 | 1 | 1 |

On the three retained ablation logs, the same synthetic model produced 3/6
ransomware-scenario hybrid detections and 3/3 benign hybrid false alarms. Scenario
windows inherit the scenario label; those operational counts are not sample-level
malware ground truth. The high false-alarm counts are recorded without tuning the
model or policy to the fixtures. They demonstrate why interface success cannot
establish efficacy. Workload generation is stochastic, and later runs can differ.

CI runs dependency, compilation, all tests, the 39-check disposable verifier and
Compose validation on Python 3.13 and 3.14. Actual remote run results are linked
in PR #2 after push; they are separate from these local results.

## Remaining compatibility and dataset limits

1. Original raw source datasets, the old prepared split CSVs and trained artifacts
   are unavailable. Historical source accuracy/calibration claims cannot be
   reproduced here. Re-ingest reviewed source measurements and regenerate splits,
   models, references, calibration and reports together; never reuse legacy scales
   or infer artifact compatibility from feature count alone. A user-supplied CSV's
   claimed normalization/provenance still requires external review.
2. The family experiment is explicitly within-source and uses selected class
   counts. It cannot establish cross-source generalization or enterprise FPR.
   Missing sessions/files, near duplicates and proxy observations remain limits;
   conservative grouping may refuse a dataset entirely. IF additionally requires
   identical benign observations across both family experiments. Family names are
   source-specific; the retained hyphen-prefix canonicalization needs review for
   other source naming conventions.
3. Relationship-aware preparation loads the source CSV in memory. Large datasets
   require sufficient RAM; the earlier row-streaming sampler was replaced to check
   connected relationships before selection. No full-size memory benchmark was
   possible without those datasets.
4. Old analyst flags/signature files need re-review; the local identity/digest
   workflow assumes trusted reviewers and storage. Candidate promotion remains
   manual. New alert/method names require downstream consumer migration; legacy
   summaries remain readable, but mixed-schema summaries are rejected.
5. Calibration FPR targets are nominal; small samples, ties, prevalence and domain
   shift matter. Synthetic workload false alarms are substantial. No production
   accuracy, zero-day efficacy or pre-impact detection claim is made.
6. Monitoring configuration was validated without running containers or rendering
   Grafana dashboards. Existing upstream Starlette/httpx deprecation warnings
   remain nonfatal. Operational deployment controls and artifact release work
   described in the foundation audit remain outstanding.
