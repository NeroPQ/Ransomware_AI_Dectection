"""Behavior semantics v3, with explicit low-confidence legacy simulation adapter."""
from collections import Counter, defaultdict
from datetime import datetime
import posixpath
import ntpath
from pathlib import PureWindowsPath

from symptom_ai.telemetry.schema import TelemetryEvent
from symptom_ai.dataset_ingestion.ingest_all_available_datasets import SYMPTOMS

FEATURE_SEMANTICS_VERSION = 3
MODIFICATIONS = {"file_modify", "file_scramble"}
POSSIBLE_IMPACT = {"file_scramble", "file_lock", "ransom_note"}
LEGACY_OPERATIONS = {
    "file_write": "file_modify", "high_entropy_write": "file_modify", "archive_chunk_write": "file_create",
    "strange_temp_write": "file_create", "file_read": "file_read", "file_touch": "file_touch",
    "file_rename": "file_rename", "extension_change": "file_extension_change",
    "ransom_note_created": "ransom_note", "stealth_note_marker": "ransom_note",
    "stealth_file_lock": "file_lock", "stealth_content_scramble": "file_scramble",
    "stealth_extension_mutation": "file_extension_change", "file_delete": "file_delete",
    "network_beacon": "network_poll", "process_spawn": "process_create", "rare_process": "process_create",
    "memory_anomaly": "memory_observation", "process_injection_suspected": "process_access",
    "backup_disable_attempt": "recovery_change", "shadow_copy_delete_attempt": "recovery_change",
}


def path_key(path):
    """Lexical normalization only; Windows case rules work on Linux too."""
    if not path:
        return None
    if "\\" in path or path.startswith("//") or (len(path) > 1 and path[1] == ":"):
        return ntpath.normcase(ntpath.normpath(path))
    return posixpath.normpath(path)


def directory_key(path):
    key = path_key(path)
    if key is None:
        return None
    return str(PureWindowsPath(key).parent) if "\\" in key else posixpath.dirname(key) or "."


def normalize_event(event):
    if "schema_version" in event:
        return TelemetryEvent.model_validate(event).model_dump(mode="json")
    # Historical generators lack trustworthy timestamps/process IDs/units. Do not
    # silently promote their assertions to confirmed observations.
    name = event.get("event_type")
    if not isinstance(name, str):
        raise ValueError("Legacy simulation requires event_type")
    return {"schema_version": 0, "source": {"kind": "legacy_simulation"},
            "host": str(event.get("host") or "unknown"),
            "process": {"entity_id": None}, "operation": LEGACY_OPERATIONS.get(name, "other"),
            "path": event.get("path"), "destination_path": None, "observation_confidence": .5,
            "entropy": None, "indicators": [], "legacy_event_type": name,
            "scenario": event.get("scenario"), "scenario_event_index": event.get("scenario_event_index")}


def analyze_events(events):
    normalized = [normalize_event(event) for event in events]
    # Deduplicate actual provider records before extracting rates/evidence.
    seen, records = {}, []
    for event in normalized:
        source = event["source"]
        key = (event["host"], source.get("provider"), source.get("record_id"))
        if event["schema_version"] and key in seen:
            if event != seen[key]:
                raise ValueError("Conflicting duplicate source record identity")
            continue
        if event["schema_version"]:
            seen[key] = event
        records.append(event)
    times = [datetime.fromisoformat(e["timestamp"]) for e in records if e["schema_version"]]
    span = (max(times) - min(times)).total_seconds() if len(times) > 1 else None
    if span is not None and span > 300:
        raise ValueError("Window exceeds five-minute correlation horizon")
    # Density over observed event span is not endpoint exposure. Use only with
    # >=2 typed timestamps; legacy logs retain count proxies, explicitly marked.
    density_scale = 60 / max(1, span) if span is not None else 1.0
    activity_divisor = 100 if span is not None else 10
    counts = Counter(e["operation"] for e in records)
    old_counts = Counter(e.get("legacy_event_type", e["operation"]) for e in records)
    # Only retain the fixed feature vocabulary; none of the old values survive.
    symptoms = dict.fromkeys(SYMPTOMS, 0.0)
    paths, directories, impact_paths, impact_dirs = set(), set(), set(), set()
    groups = defaultdict(lambda: {"indicators": set(), "references": set(), "modifications": 0,
                                   "impact_paths": set(), "impact_directories": set()})
    possible, suspicious, observed, trusted = set(), set(), set(), set()
    file_entropy, memory_entropy = [], []
    impact_count = possible_impact_count = 0
    for event in records:
        host, op, path = event["host"], event["operation"], event.get("path")
        confidence = event["observation_confidence"]
        if path:
            paths.add((host, path_key(path)))
            directories.add((host, directory_key(path)))
        observed.add(op)
        indicators = {item["kind"] for item in event["indicators"]
                      if min(item["confidence"], confidence) >= .8}
        typed_entropy = event.get("entropy")
        entropy_high = bool(typed_entropy and typed_entropy["after"] >= 7.2 and confidence >= .8)
        # High entropy is content distribution, not proof of encryption or damage.
        if entropy_high:
            (file_entropy if typed_entropy["object_type"] == "file_content" else memory_entropy).append(typed_entropy["after"] / 8)
            suspicious.add("high_file_entropy" if typed_entropy["object_type"] == "file_content" else "high_memory_entropy")
        if event.get("legacy_event_type") == "high_entropy_write":
            suspicious.add("legacy_entropy_assertion")
        if op == "file_extension_change":
            suspicious.add("extension_change")
        if indicators & {"suspected_beacon", "possible_exfiltration", "injection"}:
            suspicious.update(indicators & {"suspected_beacon", "possible_exfiltration", "injection"})
        impact = op in POSSIBLE_IMPACT or bool(indicators & {"destructive_content", "recovery_tamper", "confirmed_exfiltration"})
        # A legacy explicit entropy-write assertion is possible impact, with no
        # measured entropy or confidence upgrade. A rename/read alone is not.
        impact = impact or event.get("legacy_event_type") == "high_entropy_write"
        file_impact = impact and (op.startswith("file_") or op == "ransom_note")
        if impact:
            possible_impact_count += 1
            impact_count += file_impact
            possible.add(op)
            possible.update(indicators & {"destructive_content", "recovery_tamper", "confirmed_exfiltration"})
            if file_impact and path:
                impact_paths.add((host, path_key(path)))
                impact_dirs.add((host, directory_key(path)))
        trusted.update(indicators)
        # Containment requires same endpoint/process attribution and >=.8
        # observation confidence. Legacy events cannot satisfy this gate.
        entity = event["process"]["entity_id"]
        if entity and confidence >= .8:
            group = groups[(host, entity)]
            group["indicators"].update(indicators)
            group["references"].update(item["reference"] for item in event["indicators"] if min(item["confidence"], confidence) >= .8)
            group["modifications"] += op in MODIFICATIONS
            if file_impact and path:
                group["impact_paths"].add(path_key(path))
                group["impact_directories"].add(directory_key(path))
    clamp = lambda value: min(1.0, max(0.0, value))
    symptoms.update({
        "file_write_burst": clamp((counts["file_modify"] + counts["file_create"] + counts["file_scramble"]) * density_scale / activity_divisor),
        "file_read_burst": clamp(counts["file_read"] * density_scale / activity_divisor),
        "file_rename_burst": clamp(counts["file_rename"] * density_scale / activity_divisor),
        "mass_file_modification": clamp((counts["file_modify"] + counts["file_scramble"]) * density_scale / activity_divisor),
        "suspicious_extension_change": clamp(counts["file_extension_change"] / 4),
        "high_entropy_write": max(file_entropy + [clamp(old_counts["high_entropy_write"] / 4)]),
        "ransom_note_created": clamp(counts["ransom_note"] / 2),
        "multi_directory_impact": clamp(len(impact_dirs) / 15),
        "user_document_impact_high": 0.0,  # document role cannot be inferred from a path/count
        "suspicious_process_spawn": 0.0,  # process creation alone is observed activity
        "rare_process_name": 0.0,  # requires a measured population baseline
        "process_tree_anomaly": 0.0,
        "process_injection_suspected": float("injection" in trusted),
        "memory_access_spike": 0.0,
        "memory_entropy_region_high": max(memory_entropy + [0.0]),
        "network_api_usage": clamp((counts["network_poll"] + counts["network_connect"]) * density_scale / activity_divisor),
        "c2_beaconing": float("confirmed_c2" in trusted),
        "data_exfiltration_pattern": float("confirmed_exfiltration" in trusted),
        "storage_write_spike": clamp(old_counts["storage_write_spike"] / 4),
    })
    corroboration = []
    for (host, entity), group in sorted(groups.items()):
        indicators = group["indicators"]
        if (group["modifications"] >= 3 and len(group["impact_paths"]) >= 3
            and "destructive_content" in indicators
            and indicators & {"ransom_note", "recovery_tamper"}
            and len(group["references"]) >= 2):
            corroboration.append({"host": host, "process_entity_id": entity,
                                  "indicators": sorted(indicators), "references": sorted(group["references"]),
                                  "impact_path_count": len(group["impact_paths"]),
                                  "impact_directory_count": len(group["impact_directories"])})
    damage = {"feature_semantics_version": FEATURE_SEMANTICS_VERSION,
              "event_count": len(records), "input_event_count": len(events),
              "observed_span_seconds": span,
              "activity_basis": "observed_span_density_per_100_events_per_minute" if span is not None else "legacy_count_proxy",
              "duplicate_records_ignored": len(normalized) - len(records),
              "host_count": len({e["host"] for e in records}), "hosts": sorted({e["host"] for e in records}),
              "unique_path_count": len(paths), "observed_directory_count": len(directories),
              "impacted_directory_count": len(impact_dirs), "impacted_directories": [f"{h}:{p}" for h, p in sorted(impact_dirs)],
              "modified_file_event_count": counts["file_modify"] + counts["file_scramble"],
              "read_event_count": counts["file_read"], "touch_event_count": counts["file_touch"],
              "rename_event_count": counts["file_rename"], "delete_event_count": counts["file_delete"],
              "high_impact_event_count": impact_count, "possible_impact_event_count": possible_impact_count,
              "event_type_counts": dict(old_counts),
              "operation_counts": dict(counts), "has_file_impact": bool(impact_count),
              "legacy_observations": sum(not e["schema_version"] for e in records),
              "evidence_categories": {"observed_activity": sorted(observed), "suspicious_behavior": sorted(suspicious),
                                      "possible_impact": sorted(possible), "corroborated_high_confidence_detection": corroboration}}
    return symptoms, damage


def event_has_possible_impact(event):
    return bool(analyze_events([event])[1]["has_file_impact"])
