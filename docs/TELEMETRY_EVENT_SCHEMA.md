# Telemetry event contract v1

Task 03 defines an ingestion boundary, not a Sysmon collector. The executable contract is `src/symptom_ai/telemetry/schema.py`; the generated JSON Schema is [telemetry-event-v1.schema.json](schemas/telemetry-event-v1.schema.json). Unknown fields/versions, nonfinite measurements, incompatible indicator/operation pairs, naive timestamps and invalid entropy types are rejected. Observation-only `POST /events/evaluate` accepts 1–1,000 typed events and returns observations, individual detector evidence and the policy recommendation. It performs no endpoint response.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`, independently versioned from feature semantics 3 and alert schema 3 |
| `source` | Kind, provider, provider event ID, unique record ID; provider identity should include collection channel/session to avoid record-number reuse |
| `host` | Stable endpoint identity, not a mutable display name |
| `process.entity_id` | Stable process GUID/start identity; PID alone is insufficient |
| `process.pid`, `image`, `parent_entity_id` | Optional PID, image and parent identity; context, never implicit approval |
| `timestamp`, `received_at` | Timezone-aware event time and optional ingestion time; retain UTC/offset information |
| `operation` | Read, touch, create, modify, rename, delete, extension change, lock, scramble, note, network connect/poll, process create/access, memory observation, recovery change or other |
| `path`, `destination_path` | File operation path and required rename destination; literal data, never opened or executed |
| `observation_confidence` | Numeric [0,1] assertion about observation quality, not probability of malware |
| `entropy` | Optional file-content or memory-region byte entropy; Shannon method, bits/byte [0,8], >=256 sampled bytes, optional before and required after |
| `indicators` | Explicit suspected/confirmed network, destructive-content, note, recovery-tamper or injection assertions, each with reviewable reference and confidence |
| `scenario`, `scenario_event_index` | Optional safe-demo timing context; excluded from detector features |

Example ordinary polling record:

```json
{
  "schema_version": 1,
  "source": {"kind": "synthetic", "provider": "example-v1", "event_id": "poll", "record_id": "session-1:42"},
  "host": "endpoint-1",
  "process": {"entity_id": "process-guid-1", "pid": 500, "image": "sync-placeholder"},
  "timestamp": "2026-01-01T12:00:00Z",
  "operation": "network_poll",
  "observation_confidence": 0.95,
  "indicators": []
}
```

This contributes observed network activity, not C2/exfiltration. A supplied `confirmed_c2` indicator is still an upstream assertion whose reference and sensor quality require review. The current implementation does not independently authenticate sensors or calibrate confidence. Low-confidence assertions cannot satisfy the high-confidence corroboration gate. The 0.8 confidence gate is an engineering choice, not an empirically validated confidence threshold.

File entropy is accepted only on file create/modify/scramble observations; memory entropy only on memory observations. High entropy in an archive is not evidence of destructive encryption. Arbitrary `extra.entropy_score`, read events and polling cannot populate file entropy. Historical `high_entropy_write` simulation event names remain explicitly marked unmeasured assertions, not Shannon measurements.

Within a window, `(host, provider, record_id)` deduplicates repeated records; conflicting duplicates fail closed. Typed event span must be <=300 seconds. Corroboration is scoped to the same host/process identity within that window. No cross-window state machine or clock-skew correction is claimed. Windows paths are normalized lexically (case and separators); POSIX paths remain case-sensitive. There is no filesystem lookup, symlink resolution, volume alias resolution or document-role inference.

For a future Sysmon adapter, retain provider event/record IDs, UTC event time and ProcessGuid. Microsoft documents process creation (1), file creation-time changes (2), process-linked network connections (3), process access (10), file create/overwrite (11) and file deletion (23/26). These observations need distinct mappings; an ordinary connection or process access does not establish malicious intent. See [Microsoft's Sysmon event documentation](https://learn.microsoft.com/en-us/sysinternals/downloads/sysmon). No Sysmon installation, capture, XML adapter, comprehensive file-modification coverage or entropy enrichment is implemented here. Missing observations must stay unknown rather than being invented.
