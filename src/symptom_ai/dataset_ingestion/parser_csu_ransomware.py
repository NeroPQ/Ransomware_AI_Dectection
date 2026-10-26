from pathlib import Path
import pandas as pd
import numpy as np
from symptom_ai.dataset_ingestion.validation import norm01, text_label, binary_labels

OUT = Path("data/symptom_labels/csu_symptom_dataset.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

CSU_FILES = [
    Path("data/raw/csu_ransomware/dataset/Ransomware_Data.csv"),
    Path("data/raw/csu_ransomware/dataset/Drift-Anomaly-Versions/processed-data drift Phase 1.csv"),
    Path("data/raw/csu_ransomware/dataset/Drift-Anomaly-Versions/processed-data drift-phase 2.csv"),
    Path("data/raw/csu_ransomware/dataset/Drift-Anomaly-Versions/processed-data drift Phase 3a.csv"),
    Path("data/raw/csu_ransomware/dataset/Drift-Anomaly-Versions/processed-data drift Phase 3b.csv"),
    Path("data/raw/csu_ransomware/dataset/Drift-Anomaly-Versions/processed data drift Phase 3 c.csv"),
]



def yes01(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    return (s > 0).astype(float)


def detect_label(row):
    fields = [text_label(row.get(name)) for name in ("target-class-name", "Ware Type")]
    target = binary_labels(pd.Series([row.get("target-class")])).iloc[0]
    labels = set(fields + [target]) - {"unknown"}
    return labels.pop() if len(labels) == 1 else "unknown"


def response_policy(label, symptoms):
    if label == "benign":
        return "monitor_only"
    if label != "known_ransomware_like":
        return "analyst_review"

    if (
        symptoms.get("shadow_copy_delete_attempt", 0) >= 0.7
        or symptoms.get("backup_disable_attempt", 0) >= 0.7
    ):
        return "protective_lockdown"

    return "isolate_and_backup"


def parse_one_file(path: Path):
    print(f"[+] Parsing {path}")
    df = pd.read_csv(path)
    has_extension_similarity = "extension_similarity" in df
    # Missing raw features are absent observations, with aligned Series defaults.
    for col in ["File_Delete_archived", "File_created", "file-related", "extension_similarity", "file_name_entropy", "Process_Create", "process_vs_parent_freq_ratio", "suspicious_path", "process_name_length", "Registry_value_set", "network-related", "directory_depth", "system_executable"]:
        if col not in df:
            df[col] = 0.0

    out = pd.DataFrame()
    out["sample_id"] = [f"csu_{path.stem}_{i}" for i in range(len(df))]
    out["dataset_source"] = "CSU_Ransomware_Data"
    out["family"] = df.get("target-class-name", df.get("Ware Type", pd.Series("unknown", index=df.index))).astype(str)
    out["behavior_type"] = "sysmon_engineered_features"
    out["collection_type"] = "sysmon_features"
    out["platform"] = "windows"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    # File symptoms
    out["file_delete_burst"] = yes01(df.get("File_Delete_archived", 0))
    out["file_write_burst"] = yes01(df.get("File_created", 0))
    out["file_open_burst"] = norm01(df.get("file-related", 0))
    out["mass_file_modification"] = norm01(df.get("file-related", 0))
    out["suspicious_extension_change"] = (1 - pd.to_numeric(df["extension_similarity"], errors="coerce").clip(0, 1)).fillna(0) if has_extension_similarity else 0.0
    out["high_entropy_write"] = norm01(df.get("file_name_entropy", 0))
    out["entropy_increase_after_write"] = norm01(df.get("file_name_entropy", 0))

    # Process symptoms
    out["suspicious_process_spawn"] = yes01(df.get("Process_Create", 0))
    out["process_tree_anomaly"] = norm01(df.get("process_vs_parent_freq_ratio", 0))
    out["process_from_temp_directory"] = yes01(df.get("suspicious_path", 0))
    out["process_from_appdata"] = yes01(df.get("suspicious_path", 0))
    out["rare_process_name"] = norm01(df.get("process_name_length", 0))

    # Registry / persistence
    out["registry_run_key_modified"] = yes01(df.get("Registry_value_set", 0))
    out["registry_startup_modified"] = yes01(df.get("Registry_value_set", 0))
    out["persistence_attempt"] = yes01(df.get("Registry_value_set", 0))

    # Network
    out["high_outbound_connection_count"] = norm01(df.get("network-related", 0))
    out["c2_beaconing"] = norm01(df.get("network-related", 0))
    out["data_exfiltration_pattern"] = norm01(df.get("network-related", 0))

    # Impact / context
    out["user_document_impact_high"] = norm01(df.get("file-related", 0))
    out["multi_directory_impact"] = norm01(df.get("directory_depth", 0))
    out["critical_file_touch_attempt"] = yes01(df.get("system_executable", 0))

    labels = df.apply(detect_label, axis=1)
    out["label"] = labels

    # Conservative defaults for unavailable symptoms
    default_symptoms = [
        "file_rename_burst",
        "file_read_burst",
        "rare_extension_created",
        "many_unique_extensions_created",
        "original_extension_removed",
        "file_overwrite_pattern",
        "small_file_to_high_entropy_pattern",
        "large_number_of_user_files_touched",
        "document_file_targeting",
        "image_file_targeting",
        "database_file_targeting",
        "source_code_file_targeting",
        "backup_file_targeting",
        "ransom_note_created",
        "ransom_note_multiple_directories",
        "command_shell_execution",
        "powershell_execution",
        "wscript_cscript_execution",
        "cmd_batch_execution",
        "living_off_the_land_binary_usage",
        "unexpected_admin_tool_usage",
        "process_injection_suspected",
        "process_hollowing_suspected",
        "unsigned_process_execution",
        "process_from_user_downloads",
        "scheduled_task_created",
        "service_created",
        "service_modified",
        "startup_folder_modified",
        "autorun_entry_created",
        "wmi_persistence_suspected",
        "backup_disable_attempt",
        "shadow_copy_delete_attempt",
        "vssadmin_usage",
        "wmic_shadowcopy_usage",
        "bcdedit_recovery_disabled",
        "recovery_catalog_deleted",
        "restore_point_deleted",
        "backup_service_stopped",
        "backup_file_deleted",
        "system_restore_disabled",
        "security_tool_tamper",
        "antivirus_disabled",
        "firewall_disabled",
        "defender_preference_modified",
        "edr_process_kill_attempt",
        "security_service_stopped",
        "security_log_cleared",
        "event_log_clear_attempt",
        "amsi_bypass_suspected",
        "uac_bypass_suspected",
        "suspicious_dns",
        "rare_domain_contact",
        "newly_seen_domain",
        "tor_or_proxy_usage",
        "suspicious_ip_reputation",
        "network_share_scan",
        "smb_connection_burst",
        "rdp_connection_attempt",
        "remote_admin_tool_connection",
        "dns_tunneling_suspected",
        "http_post_burst",
        "encrypted_traffic_to_rare_host",
        "packed_binary",
        "high_section_entropy",
        "suspicious_section_name",
        "crypto_api_usage",
        "file_api_usage",
        "network_api_usage",
        "process_api_usage",
        "registry_api_usage",
        "service_api_usage",
        "anti_analysis",
        "anti_debugging",
        "anti_vm",
        "anti_sandbox",
        "suspicious_string",
        "ransomware_keyword_string",
        "hardcoded_extension_list",
        "hardcoded_ransom_note_template",
        "embedded_public_key",
        "suspicious_import_table",
        "low_import_count_packed",
        "storage_write_spike",
        "storage_read_write_ratio_anomaly",
        "random_write_pattern",
        "sequential_file_rewrite_pattern",
        "memory_access_spike",
        "memory_entropy_region_high",
        "rapid_buffer_write_pattern",
        "disk_io_queue_spike",
        "abnormal_storage_latency",
        "high_iops_short_window",
        "known_family_similarity_low",
        "known_family_similarity_medium",
        "known_family_similarity_high",
        "anomaly_score_high",
        "novel_symptom_combination",
        "partial_ransomware_match",
        "unknown_high_risk",
        "unknown_low_risk",
        "analyst_review_required",
        "retraining_candidate",
    ]

    defaults = pd.DataFrame({col: 0.0 for col in default_symptoms if col not in out}, index=out.index)
    out = pd.concat([out, defaults], axis=1)

    policies = []
    for _, row in out.iterrows():
        symptoms = row.to_dict()
        policies.append(response_policy(row["label"], symptoms))

    out["response_policy"] = policies

    return out


def main():
    frames = []

    for path in CSU_FILES:
        if path.exists():
            frames.append(parse_one_file(path))
        else:
            print(f"[!] Missing {path}")

    if not frames:
        raise SystemExit("[!] No CSU files found.")

    final = pd.concat(frames, ignore_index=True)
    final.to_csv(OUT, index=False)

    print(f"\n[+] CSU symptom dataset saved to {OUT}")
    print(f"[+] Rows: {len(final)}")
    print(f"[+] Columns: {len(final.columns)}")
    print("\nLabel counts:")
    print(final["label"].value_counts().to_string())
    print("\nPolicy counts:")
    print(final["response_policy"].value_counts().to_string())


if __name__ == "__main__":
    main()
