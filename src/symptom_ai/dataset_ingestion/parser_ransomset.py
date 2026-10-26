from pathlib import Path
import pandas as pd
import numpy as np
from symptom_ai.dataset_ingestion.validation import norm01, text_label, binary_labels

OUT = Path("data/symptom_labels/ransomset_symptom_dataset.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

RANSOMSET_FILE = Path("data/raw/ransomset/Dataset/ransomset-multiclass-dataset.csv")



def any_api_score(df, api_names):
    cols = [c for c in api_names if c in df.columns]
    if not cols:
        return pd.Series([0.0] * len(df))
    temp = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return (temp.sum(axis=1) > 0).astype(float)


def sum_api_score(df, api_names):
    cols = [c for c in api_names if c in df.columns]
    if not cols:
        return pd.Series([0.0] * len(df))
    temp = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return norm01(temp.sum(axis=1))


def get_label(row):
    if "score_binary" in row:
        binary = binary_labels(pd.Series([row["score_binary"]])).iloc[0]
        textual = text_label(row.get("class"))
        if textual != "unknown" and binary != textual:
            return "unknown"
        return binary
    return text_label(row.get("class"))


def main():
    if not RANSOMSET_FILE.exists():
        raise SystemExit(f"[!] Missing {RANSOMSET_FILE}")

    print(f"[+] Reading {RANSOMSET_FILE}")
    df = pd.read_csv(RANSOMSET_FILE)

    out = pd.DataFrame()
    out["sample_id"] = [f"ransomset_{i}" for i in range(len(df))]
    out["dataset_source"] = "RansomSet"
    out["family"] = "unknown_ransomset_family"
    out["behavior_type"] = "api_system_call_features"
    out["collection_type"] = "dynamic_api_features"
    out["platform"] = "windows"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    # API groups
    file_api = [
        "NtCreateFile", "NtReadFile", "NtWriteFile", "NtOpenFile",
        "FindFirstFileExW", "GetFileAttributesW", "GetFileSize",
        "SetFilePointer", "NtQueryDirectoryFile", "CreateDirectoryW",
        "CopyFileA", "GetTempPathW"
    ]

    registry_api = [
        "RegOpenKeyExA", "RegOpenKeyExW", "RegQueryValueExW",
        "RegQueryValueExA", "RegCloseKey", "RegCreateKeyExA",
        "RegSetValueExA", "NtOpenKey", "NtQueryValueKey",
        "NtEnumerateKey", "NtEnumerateValueKey"
    ]

    process_api = [
        "CreateProcessInternalW", "ShellExecuteExW", "NtTerminateProcess",
        "NtCreateMutant", "NtDuplicateObject", "NtQuerySystemInformation"
    ]

    memory_api = [
        "NtAllocateVirtualMemory", "NtFreeVirtualMemory",
        "NtCreateSection", "NtMapViewOfSection", "NtUnmapViewOfSection",
        "GlobalMemoryStatusEx"
    ]

    dll_api = [
        "LdrLoadDll", "LdrGetProcedureAddress", "LdrGetDllHandle",
        "LdrUnloadDll", "LoadStringW", "LoadStringA"
    ]

    system_api = [
        "GetSystemWindowsDirectoryW", "GetSystemDirectoryW",
        "GetNativeSystemInfo", "GetSystemTimeAsFileTime"
    ]

    # File symptoms
    out["file_write_burst"] = sum_api_score(df, ["NtWriteFile", "NtCreateFile", "CreateDirectoryW", "CopyFileA"])
    out["file_read_burst"] = sum_api_score(df, ["NtReadFile", "NtOpenFile", "FindFirstFileExW", "NtQueryDirectoryFile"])
    out["file_open_burst"] = sum_api_score(df, file_api)
    out["mass_file_modification"] = sum_api_score(df, ["NtCreateFile", "NtWriteFile", "CreateDirectoryW"])
    out["document_file_targeting"] = any_api_score(df, ["FindFirstFileExW", "GetFileAttributesW"])
    out["file_api_usage"] = sum_api_score(df, file_api)

    # Process symptoms
    out["suspicious_process_spawn"] = any_api_score(df, ["CreateProcessInternalW", "ShellExecuteExW"])
    out["process_tree_anomaly"] = sum_api_score(df, process_api)
    out["process_api_usage"] = sum_api_score(df, process_api)
    out["rare_process_name"] = any_api_score(df, ["NtCreateMutant"])

    # Registry / persistence symptoms
    out["registry_run_key_modified"] = any_api_score(df, ["RegCreateKeyExA", "RegSetValueExA"])
    out["registry_startup_modified"] = any_api_score(df, ["RegCreateKeyExA", "RegSetValueExA"])
    out["registry_api_usage"] = sum_api_score(df, registry_api)
    out["persistence_attempt"] = any_api_score(df, ["RegCreateKeyExA", "RegSetValueExA", "NtCreateMutant"])

    # Static/reverse-like behavior symptoms from API use
    out["crypto_api_usage"] = any_api_score(df, ["CryptAcquireContextW", "CryptGenKey", "CryptEncrypt", "CryptDecrypt"])
    out["anti_analysis"] = any_api_score(df, ["NtQuerySystemInformation", "GetNativeSystemInfo"])
    out["anti_debugging"] = any_api_score(df, ["NtQueryInformationProcess", "IsDebuggerPresent"])
    out["suspicious_string"] = any_api_score(df, ["LoadStringW", "LoadStringA"])
    out["network_api_usage"] = any_api_score(df, ["InternetOpenA", "InternetConnectA", "HttpSendRequestA", "connect", "send", "recv"])
    out["service_api_usage"] = any_api_score(df, ["OpenSCManagerA", "CreateServiceA", "StartServiceA"])

    # Memory/storage symptoms
    out["memory_access_spike"] = sum_api_score(df, memory_api)
    out["memory_entropy_region_high"] = sum_api_score(df, ["NtAllocateVirtualMemory", "NtCreateSection", "NtMapViewOfSection"])
    out["rapid_buffer_write_pattern"] = sum_api_score(df, ["NtWriteFile", "NtAllocateVirtualMemory"])

    # Network symptoms
    out["c2_beaconing"] = out["network_api_usage"]
    out["data_exfiltration_pattern"] = out["network_api_usage"]
    out["high_outbound_connection_count"] = out["network_api_usage"]

    # unavailable symptoms default to zero
    default_cols = [
        "file_delete_burst", "file_rename_burst", "high_entropy_write",
        "entropy_increase_after_write", "suspicious_extension_change",
        "rare_extension_created", "many_unique_extensions_created",
        "original_extension_removed", "file_overwrite_pattern",
        "small_file_to_high_entropy_pattern", "large_number_of_user_files_touched",
        "image_file_targeting", "database_file_targeting", "source_code_file_targeting",
        "backup_file_targeting", "ransom_note_created", "ransom_note_multiple_directories",
        "process_from_temp_directory", "process_from_appdata", "process_from_user_downloads",
        "command_shell_execution", "powershell_execution", "wscript_cscript_execution",
        "cmd_batch_execution", "living_off_the_land_binary_usage",
        "unexpected_admin_tool_usage", "process_injection_suspected",
        "process_hollowing_suspected", "unsigned_process_execution",
        "scheduled_task_created", "service_created", "service_modified",
        "startup_folder_modified", "autorun_entry_created", "wmi_persistence_suspected",
        "backup_disable_attempt", "shadow_copy_delete_attempt", "vssadmin_usage",
        "wmic_shadowcopy_usage", "bcdedit_recovery_disabled",
        "recovery_catalog_deleted", "restore_point_deleted", "backup_service_stopped",
        "backup_file_deleted", "system_restore_disabled", "security_tool_tamper",
        "antivirus_disabled", "firewall_disabled", "defender_preference_modified",
        "edr_process_kill_attempt", "security_service_stopped",
        "security_log_cleared", "event_log_clear_attempt", "amsi_bypass_suspected",
        "uac_bypass_suspected", "suspicious_dns", "rare_domain_contact",
        "newly_seen_domain", "tor_or_proxy_usage", "suspicious_ip_reputation",
        "network_share_scan", "smb_connection_burst", "rdp_connection_attempt",
        "remote_admin_tool_connection", "dns_tunneling_suspected",
        "http_post_burst", "encrypted_traffic_to_rare_host",
        "packed_binary", "high_section_entropy", "suspicious_section_name",
        "anti_vm", "anti_sandbox", "ransomware_keyword_string",
        "hardcoded_extension_list", "hardcoded_ransom_note_template",
        "embedded_public_key", "suspicious_import_table", "low_import_count_packed",
        "storage_write_spike", "storage_read_write_ratio_anomaly",
        "random_write_pattern", "sequential_file_rewrite_pattern",
        "disk_io_queue_spike", "abnormal_storage_latency", "high_iops_short_window",
        "user_document_impact_high", "shared_folder_impact",
        "system_folder_touch_attempt", "critical_file_touch_attempt",
        "business_file_touch_pattern", "multi_directory_impact",
        "multi_drive_impact", "network_share_file_impact",
        "known_family_similarity_low", "known_family_similarity_medium",
        "known_family_similarity_high", "anomaly_score_high",
        "novel_symptom_combination", "partial_ransomware_match",
        "unknown_high_risk", "unknown_low_risk",
        "analyst_review_required", "retraining_candidate"
    ]

    zero_df = pd.DataFrame({col: 0.0 for col in default_cols if col not in out.columns}, index=out.index)
    out = pd.concat([out, zero_df], axis=1)

    # Missing and malformed labels are review cases, never positive defaults.
    out["label"] = df.apply(get_label, axis=1)

    out["response_policy"] = np.where(
        out["label"] == "benign",
        "monitor_only",
        np.where(out["label"] == "known_ransomware_like", "isolate_and_backup", "analyst_review")
    )

    out.to_csv(OUT, index=False)

    print(f"[+] RansomSet symptom dataset saved to {OUT}")
    print(f"[+] Rows: {len(out)}")
    print(f"[+] Columns: {len(out.columns)}")
    print("\nLabel counts:")
    print(out["label"].value_counts().to_string())
    print("\nPolicy counts:")
    print(out["response_policy"].value_counts().to_string())


if __name__ == "__main__":
    main()
