from pathlib import Path
import zipfile
import pandas as pd
import numpy as np
from symptom_ai.dataset_ingestion.validation import norm01, text_label, binary_labels, combine_frames


OUT_DIR = Path("data/symptom_labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_OUT = OUT_DIR / "unified_symptom_dataset.csv"


META_COLS = [
    "sample_id",
    "dataset_source",
    "family",
    "behavior_type",
    "collection_type",
    "platform",
    "is_simulated",
    "is_real_malware_executed",
]

SYMPTOMS = [
    "file_write_burst", "file_rename_burst", "file_delete_burst", "file_read_burst",
    "file_open_burst", "mass_file_modification", "high_entropy_write",
    "entropy_increase_after_write", "suspicious_extension_change",
    "rare_extension_created", "many_unique_extensions_created",
    "original_extension_removed", "file_overwrite_pattern",
    "small_file_to_high_entropy_pattern", "large_number_of_user_files_touched",
    "document_file_targeting", "image_file_targeting", "database_file_targeting",
    "source_code_file_targeting", "backup_file_targeting", "ransom_note_created",
    "ransom_note_multiple_directories",

    "suspicious_process_spawn", "process_tree_anomaly", "high_child_process_count",
    "command_shell_execution", "powershell_execution", "wscript_cscript_execution",
    "cmd_batch_execution", "living_off_the_land_binary_usage",
    "unexpected_admin_tool_usage", "process_injection_suspected",
    "process_hollowing_suspected", "unsigned_process_execution", "rare_process_name",
    "process_from_temp_directory", "process_from_user_downloads", "process_from_appdata",

    "registry_run_key_modified", "registry_startup_modified", "scheduled_task_created",
    "service_created", "service_modified", "startup_folder_modified",
    "persistence_attempt", "autorun_entry_created", "wmi_persistence_suspected",

    "backup_disable_attempt", "shadow_copy_delete_attempt", "vssadmin_usage",
    "wmic_shadowcopy_usage", "bcdedit_recovery_disabled", "recovery_catalog_deleted",
    "restore_point_deleted", "backup_service_stopped", "backup_file_deleted",
    "system_restore_disabled",

    "security_tool_tamper", "antivirus_disabled", "firewall_disabled",
    "defender_preference_modified", "edr_process_kill_attempt",
    "security_service_stopped", "security_log_cleared", "event_log_clear_attempt",
    "amsi_bypass_suspected", "uac_bypass_suspected",

    "c2_beaconing", "suspicious_dns", "rare_domain_contact", "newly_seen_domain",
    "tor_or_proxy_usage", "suspicious_ip_reputation",
    "high_outbound_connection_count", "high_outbound_bytes",
    "data_exfiltration_pattern", "network_share_scan", "smb_connection_burst",
    "rdp_connection_attempt", "remote_admin_tool_connection",
    "dns_tunneling_suspected", "http_post_burst",
    "encrypted_traffic_to_rare_host",

    "packed_binary", "high_section_entropy", "suspicious_section_name",
    "crypto_api_usage", "file_api_usage", "network_api_usage",
    "process_api_usage", "registry_api_usage", "service_api_usage",
    "anti_analysis", "anti_debugging", "anti_vm", "anti_sandbox",
    "suspicious_string", "ransomware_keyword_string", "hardcoded_extension_list",
    "hardcoded_ransom_note_template", "embedded_public_key",
    "suspicious_import_table", "low_import_count_packed",

    "storage_write_spike", "storage_read_write_ratio_anomaly", "random_write_pattern",
    "sequential_file_rewrite_pattern", "memory_access_spike",
    "memory_entropy_region_high", "rapid_buffer_write_pattern",
    "disk_io_queue_spike", "abnormal_storage_latency", "high_iops_short_window",

    "user_document_impact_high", "shared_folder_impact", "system_folder_touch_attempt",
    "critical_file_touch_attempt", "business_file_touch_pattern",
    "multi_directory_impact", "multi_drive_impact", "network_share_file_impact",

    "known_family_similarity_low", "known_family_similarity_medium",
    "known_family_similarity_high", "anomaly_score_high",
    "novel_symptom_combination", "partial_ransomware_match",
    "unknown_high_risk", "unknown_low_risk", "analyst_review_required",
    "retraining_candidate",
]

FINAL_COLS = META_COLS + SYMPTOMS + ["label", "response_policy"]



def empty_frame(n):
    return pd.DataFrame(0.0, index=range(n), columns=SYMPTOMS)


def finalize(df):
    for c in FINAL_COLS:
        if c not in df.columns:
            df[c] = 0.0 if c in SYMPTOMS else "unknown"
    provenance = [c for c in ("session_id", "source_file") if c in df]
    return df[FINAL_COLS + provenance]


def save_part(df, name):
    out = OUT_DIR / f"{name}_symptom_dataset.csv"
    df = finalize(df)
    df.to_csv(out, index=False)
    print(f"[+] Saved {name}: {len(df)} rows -> {out}")
    return out


def label_to_policy(label):
    if label == "benign":
        return "monitor_only"
    if label == "unknown_high_risk":
        return "protective_lockdown"
    return "isolate_and_backup" if label == "known_ransomware_like" else "analyst_review"


detect_label_from_text = text_label


def ingest_mlran():
    paths = [
        Path("data/raw/mlran/6_experiments/FS_MLRan_Datasets/MLRan_X_train_RFE.csv"),
        Path("data/raw/mlran/6_experiments/FS_MLRan_Datasets/MLRan_X_test_RFE.csv"),
    ]
    frames = []

    for p in paths:
        if not p.exists():
            continue
        raw = pd.read_csv(p)
        n = len(raw)
        out = empty_frame(n)

        out["sample_id"] = raw.get("sample_id", pd.Series([f"mlran_{p.stem}_{i}" for i in range(n)])).astype(str)
        out["dataset_source"] = "MLRan"
        out["family"] = raw.get("family_label", "unknown").astype(str) if "family_label" in raw else "unknown"
        out["behavior_type"] = "dynamic_cuckoo_features"
        out["collection_type"] = "dynamic_features"
        out["platform"] = "windows"
        out["is_simulated"] = 0
        out["is_real_malware_executed"] = 0

        feature_cols = [c for c in raw.columns if str(c).isdigit()]
        if feature_cols:
            numeric_sum = raw[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
            score = norm01(numeric_sum)
            out["process_api_usage"] = score
            out["file_api_usage"] = score
            out["registry_api_usage"] = score * 0.6
            out["network_api_usage"] = score * 0.4
            out["suspicious_process_spawn"] = score
            out["anti_analysis"] = score * 0.3
            out["suspicious_string"] = score * 0.4

        if "sample_type" in raw.columns:
            st = raw["sample_type"].astype(str).str.lower()
            out["label"] = st.apply(text_label)
        else:
            out["label"] = "unknown"

        out["response_policy"] = out["label"].apply(label_to_policy)
        frames.append(out)

    if not frames:
        print("[!] MLRan missing")
        return None

    return save_part(pd.concat(frames, ignore_index=True), "mlran")


def ingest_windows_pe_api_calls():
    root = Path("data/raw/windows_pe_api_calls")
    calls = root / "sample_analysis_data.csv"
    labels = root / "labels.csv"

    if not calls.exists():
        print("[!] Windows_PE_API_Calls missing")
        return None

    X = pd.read_csv(calls, header=None)
    n = len(X)
    out = empty_frame(n)

    out["sample_id"] = [f"windows_pe_api_{i}" for i in range(n)]
    out["dataset_source"] = "Windows_PE_API_Calls"
    out["family"] = "malware_api_family"
    out["behavior_type"] = "api_call_sequence"
    out["collection_type"] = "cuckoo_api_sequence"
    out["platform"] = "windows"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    seq = X.iloc[:, 0].astype(str).str.lower()

    def has(words):
        pattern = "|".join(words)
        return seq.str.contains(pattern, regex=True).astype(float)

    out["file_api_usage"] = has(["ntcreatefile", "ntreadfile", "ntwritefile", "copyfile", "findfirstfile"])
    out["file_write_burst"] = has(["ntwritefile", "ntcreatefile", "copyfile"])
    out["file_read_burst"] = has(["ntreadfile", "findfirstfile"])
    out["registry_api_usage"] = has(["regopen", "regquery", "regset", "ntopenkey"])
    out["registry_run_key_modified"] = has(["regset", "regcreate"])
    out["persistence_attempt"] = has(["regset", "regcreate", "ntcreatemutant"])
    out["process_api_usage"] = has(["createprocess", "ntterminateprocess", "shellexecute"])
    out["suspicious_process_spawn"] = has(["createprocess", "shellexecute"])
    out["memory_access_spike"] = has(["ntallocatevirtualmemory", "ntmapviewofsection", "ntcreatesection"])
    out["anti_analysis"] = has(["ntquerysysteminformation", "getnative"])
    out["suspicious_string"] = has(["loadstring"])
    out["network_api_usage"] = has(["internet", "connect", "send", "recv", "http"])
    out["c2_beaconing"] = out["network_api_usage"]
    out["data_exfiltration_pattern"] = out["network_api_usage"] * 0.7

    if labels.exists():
        y = pd.read_csv(labels, header=None).iloc[:, 0].astype(str)
        out["family"] = y
        out["label"] = y.apply(detect_label_from_text)
    else:
        out["label"] = "unknown"

    out["response_policy"] = out["label"].apply(label_to_policy)
    return save_part(out, "windows_pe_api_calls")


def ingest_riss():
    print("[!] RISS adapter disabled: original features were derived from target labels; implement a verified observation mapping before use")
    return None


def generic_csv_ingest(dataset_name, root_path, source_label="unknown"):
    root = Path(root_path)
    csvs = sorted(root.rglob("*.csv"))
    frames = []

    for p in csvs[:20]:
        try:
            raw = pd.read_csv(p)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"[!] Cannot read {p}: {type(exc).__name__}")
            continue

        if len(raw) == 0:
            continue

        n = len(raw)
        out = empty_frame(n)
        out["sample_id"] = [f"{dataset_name}_{p.stem}_{i}" for i in range(n)]
        out["dataset_source"] = dataset_name
        out["family"] = "unknown"
        out["behavior_type"] = "generic_feature_csv"
        out["collection_type"] = "csv_features"
        out["platform"] = "windows"
        out["is_simulated"] = 0
        out["is_real_malware_executed"] = 0

        lower_cols = {c.lower(): c for c in raw.columns if c.lower() not in {"label", "class", "target", "target-class", "target-class-name", "category", "family", "prediction", "threats"}}

        def col_contains(keys):
            cols = [orig for low, orig in lower_cols.items() if any(k in low for k in keys)]
            if not cols:
                return pd.Series([0.0] * n)
            return norm01(raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1))

        out["file_write_burst"] = col_contains(["write", "file_created", "file_create"])
        out["file_delete_burst"] = col_contains(["delete"])
        out["file_read_burst"] = col_contains(["read"])
        out["high_entropy_write"] = col_contains(["entropy"])
        out["suspicious_extension_change"] = col_contains(["extension"])
        out["suspicious_process_spawn"] = col_contains(["process", "proc"])
        out["registry_run_key_modified"] = col_contains(["registry", "reg"])
        out["persistence_attempt"] = out["registry_run_key_modified"]
        out["c2_beaconing"] = col_contains(["network", "dns", "http", "connection"])
        out["data_exfiltration_pattern"] = col_contains(["exfil", "bytes", "upload", "network"])
        out["packed_binary"] = col_contains(["pack"])
        out["crypto_api_usage"] = col_contains(["crypto", "encrypt"])
        out["file_api_usage"] = col_contains(["file"])
        out["process_api_usage"] = col_contains(["process", "api"])
        out["network_api_usage"] = col_contains(["network", "dns", "http"])
        out["storage_write_spike"] = col_contains(["write", "iops", "storage"])

        label_col = None
        for c in raw.columns:
            if c.lower() in ["label", "class", "target", "target-class", "target-class-name", "category"]:
                label_col = c
                break

        if label_col:
            out["label"] = raw[label_col].apply(detect_label_from_text)
        else:
            out["label"] = "unknown"  # Source name is not ground truth

        out["response_policy"] = out["label"].apply(label_to_policy)
        frames.append(out)

    if not frames:
        print(f"[!] {dataset_name} no usable CSV")
        return None

    return save_part(pd.concat(frames, ignore_index=True), dataset_name.lower())


def read_trace_csv_or_zip(path: Path):
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if n.endswith(".csv")]
                if not names:
                    return pd.DataFrame()
                with z.open(names[0]) as f:
                    return pd.read_csv(f)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def ingest_storage_trace_dataset(name, root_path):
    root = Path(root_path)
    if not root.exists():
        print(f"[!] {name} missing")
        return None

    run_dirs = sorted([p for p in root.rglob("*") if p.is_dir() and any((p / f).exists() for f in ["ata_read.csv", "ata_write.csv", "ata_read.zip", "ata_write.zip"])])
    rows = []

    benign_words = ["firefox", "excel", "zip", "normal", "benign"]

    for i, run in enumerate(run_dirs[:500]):
        rel = str(run.relative_to(root))
        app = run.parts[-2] if len(run.parts) >= 2 else rel
        text = rel.lower()
        label = text_label(text)  # Unrecognized run names require review

        read_file = next((run / f for f in ["ata_read.csv", "ata_read.zip"] if (run / f).exists()), None)
        write_file = next((run / f for f in ["ata_write.csv", "ata_write.zip"] if (run / f).exists()), None)

        rdf = read_trace_csv_or_zip(read_file) if read_file else pd.DataFrame()
        wdf = read_trace_csv_or_zip(write_file) if write_file else pd.DataFrame()

        read_n = len(rdf)
        write_n = len(wdf)
        total = max(read_n + write_n, 1)
        write_ratio = write_n / total

        row = {c: 0.0 for c in SYMPTOMS}
        row.update({
            "sample_id": f"{name}_{i}",
            "dataset_source": name,
            "family": app,
            "behavior_type": "storage_access_pattern",
            "collection_type": "ata_trace",
            "platform": "windows",
            "is_simulated": 0,
            "is_real_malware_executed": 0,

            "file_read_burst": min(read_n / 50000, 1.0),
            "file_write_burst": min(write_n / 50000, 1.0),
            "storage_write_spike": min(write_n / 50000, 1.0),
            "storage_read_write_ratio_anomaly": min(write_ratio * 2, 1.0),
            "disk_io_queue_spike": min((read_n + write_n) / 100000, 1.0),
            "high_iops_short_window": min((read_n + write_n) / 100000, 1.0),
            "random_write_pattern": min(write_ratio * 1.5, 1.0),
            "label": label,
            "response_policy": label_to_policy(label)
        })
        rows.append(row)

    if not rows:
        print(f"[!] {name} no trace runs found")
        return None

    return save_part(pd.DataFrame(rows), name.lower())


def build_unified(parts):
    existing = [
        OUT_DIR / "csu_symptom_dataset.csv",
        OUT_DIR / "ransomset_symptom_dataset.csv",
    ]

    all_files = []
    for p in existing + [x for x in parts if x is not None]:
        if p and Path(p).exists():
            all_files.append(Path(p))

    frames = []
    for p in all_files:
        print(f"[+] Loading into unified: {p}")
        frames.append(pd.read_csv(p))

    if not frames:
        raise SystemExit("[!] No datasets to unify")

    final = combine_frames(frames)
    final.to_csv(UNIFIED_OUT, index=False)

    print("\n[+] Unified saved:", UNIFIED_OUT)
    print("[+] Rows:", len(final))
    print("[+] Columns:", len(final.columns))
    print("\nDataset counts:")
    print(final["dataset_source"].value_counts().to_string())
    print("\nLabel counts:")
    print(final["label"].value_counts().to_string())


def main():
    parts = []

    parts.append(ingest_mlran())
    parts.append(ingest_windows_pe_api_calls())
    parts.append(ingest_riss())

    parts.append(generic_csv_ingest("SILRAD", "data/raw/silrad"))
    parts.append(generic_csv_ingest("Ransomware_Dataset_2024", "data/raw/ransomware_dataset_2024"))

    # If these folders are available, ingest selected trace subsets.
    parts.append(ingest_storage_trace_dataset("RanSAP", "data/raw/ransap"))
    parts.append(ingest_storage_trace_dataset("RanSMAP", "data/raw/ransmap"))

    try:
        build_unified(parts)
    except ValueError as exc:
        raise SystemExit(f"Dataset validation failed: {exc}") from exc




# ===== Additional parsers added after dataset inspection =====











# ===== Additional parsers added after dataset inspection =====

def ingest_silrad():
    root = Path("data/raw/silrad/extracted")
    csvs = [
        root / "fasttext-trainmodel.csv",
        root / "fasttext-testmodel.csv",
        root / "fasttext-all-nofamily.csv",
    ]

    frames = []
    for p in csvs:
        if not p.exists():
            continue

        raw = pd.read_csv(p)
        n = len(raw)
        out = empty_frame(n)

        out["sample_id"] = [f"silrad_{p.stem}_{i}" for i in range(n)]
        out["dataset_source"] = "SILRAD"
        out["family"] = raw.get("class", "unknown").astype(str) if "class" in raw.columns else "unknown"
        out["behavior_type"] = "sysmon_ransomware_logs"
        out["collection_type"] = "sysmon_csv"
        out["platform"] = "windows"
        out["is_simulated"] = 0
        out["is_real_malware_executed"] = 0

        text_cols = [
            "CommandLine", "ParentCommandLine", "Image", "ParentImage",
            "TargetObject", "RuleName", "Details", "Description", "task"
        ]

        text = pd.Series([""] * n)
        for col in text_cols:
            if col in raw.columns:
                text = text + " " + raw[col].astype(str).str.lower()

        out["command_shell_execution"] = text.str.contains("cmd.exe|powershell|pwsh", regex=True).astype(float)
        out["powershell_execution"] = text.str.contains("powershell|pwsh", regex=True).astype(float)
        out["wscript_cscript_execution"] = text.str.contains("wscript|cscript", regex=True).astype(float)
        out["living_off_the_land_binary_usage"] = text.str.contains("vssadmin|wmic|bcdedit|schtasks|reg.exe|rundll32|mshta|certutil", regex=True).astype(float)

        out["vssadmin_usage"] = text.str.contains("vssadmin", regex=False).astype(float)
        out["wmic_shadowcopy_usage"] = text.str.contains("shadowcopy|wmic", regex=True).astype(float)
        out["shadow_copy_delete_attempt"] = text.str.contains("shadowcopy delete|delete shadows|vssadmin delete", regex=True).astype(float)
        out["bcdedit_recovery_disabled"] = text.str.contains("bcdedit|recoveryenabled no", regex=True).astype(float)
        out["backup_disable_attempt"] = text.str.contains("backup|wbadmin|recovery", regex=True).astype(float)

        out["registry_run_key_modified"] = text.str.contains("runonce|currentversion\\\\run|registry|regset|reg.exe", regex=True).astype(float)
        out["registry_startup_modified"] = out["registry_run_key_modified"]
        out["persistence_attempt"] = text.str.contains("runonce|startup|schtasks|service|autorun", regex=True).astype(float)
        out["scheduled_task_created"] = text.str.contains("schtasks|scheduled task", regex=True).astype(float)
        out["service_created"] = text.str.contains("createservice|service created|sc.exe", regex=True).astype(float)

        out["security_tool_tamper"] = text.str.contains("defender|windows defender|security|antivirus|edr|disableantispyware", regex=True).astype(float)
        out["event_log_clear_attempt"] = text.str.contains("wevtutil|clear-log|eventlog", regex=True).astype(float)

        out["file_write_burst"] = text.str.contains("filecreate|createfile|writefile", regex=True).astype(float)
        out["file_delete_burst"] = text.str.contains("filedelete|deletefile", regex=True).astype(float)
        out["file_read_burst"] = text.str.contains("fileread|readfile", regex=True).astype(float)
        out["suspicious_extension_change"] = text.str.contains(".locked|.encrypted|.crypt|.ryk|.wannacry|.lockbit", regex=True).astype(float)
        out["ransom_note_created"] = text.str.contains("readme|decrypt|ransom|restore files", regex=True).astype(float)

        out["suspicious_process_spawn"] = text.str.contains("processcreate|process created|cmd.exe|powershell|rundll32", regex=True).astype(float)
        out["process_from_temp_directory"] = text.str.contains("\\\\temp\\\\|/temp/|appdata", regex=True).astype(float)
        out["process_from_appdata"] = text.str.contains("appdata", regex=False).astype(float)

        out["c2_beaconing"] = text.str.contains("network|dns|http|https|tcp|connection", regex=True).astype(float)
        out["suspicious_dns"] = text.str.contains("dns", regex=False).astype(float)
        out["data_exfiltration_pattern"] = text.str.contains("upload|post|http|network", regex=True).astype(float)

        if "class" in raw.columns:
            out["label"] = raw["class"].apply(detect_label_from_text)
        else:
            out["label"] = "unknown"

        out["response_policy"] = out["label"].apply(label_to_policy)
        frames.append(out)

    if not frames:
        print("[!] SILRAD no usable CSV")
        return None

    return save_part(pd.concat(frames, ignore_index=True), "silrad")


def ingest_kaggle_ransomware_pe():
    p = Path("data/raw/kaggle_ransomware_pe/data_file.csv")
    if not p.exists():
        print("[!] Kaggle_Ransomware_PE missing")
        return None

    raw = pd.read_csv(p)
    n = len(raw)
    out = empty_frame(n)

    out["sample_id"] = raw.get("md5Hash", pd.Series([f"kaggle_pe_{i}" for i in range(n)])).astype(str)
    out["dataset_source"] = "Kaggle_Ransomware_PE"
    out["family"] = "static_pe"
    out["behavior_type"] = "static_pe_features"
    out["collection_type"] = "pe_metadata"
    out["platform"] = "windows"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    out["packed_binary"] = norm01(raw.get("NumberOfSections", 0))
    out["high_section_entropy"] = norm01(raw.get("ResourceSize", 0))
    out["suspicious_import_table"] = norm01(raw.get("IatVRA", 0))
    out["suspicious_section_name"] = norm01(raw.get("ExportSize", 0))
    out["low_import_count_packed"] = 1 - norm01(raw["IatVRA"]) if "IatVRA" in raw else 0.0
    out["suspicious_string"] = norm01(raw.get("BitcoinAddresses", 0))
    out["ransomware_keyword_string"] = norm01(raw.get("BitcoinAddresses", 0))

    if "Benign" in raw.columns:
        benign = pd.to_numeric(raw["Benign"], errors="coerce")
        out["label"] = binary_labels(benign, one_is_benign=True)
    else:
        out["label"] = "unknown"

    out["response_policy"] = out["label"].apply(label_to_policy)
    return save_part(out, "kaggle_ransomware_pe")


def ingest_api_call_sequences():
    p = Path("data/raw/api_call_sequences/dynamic_api_call_sequence_per_malware_100_0_306.csv")
    if not p.exists():
        print("[!] API_Call_Sequences missing")
        return None

    raw = pd.read_csv(p)
    n = len(raw)
    out = empty_frame(n)

    out["sample_id"] = raw.get("hash", pd.Series([f"api_seq_{i}" for i in range(n)])).astype(str)
    out["dataset_source"] = "API_Call_Sequences"
    out["family"] = "malware_api_sequence"
    out["behavior_type"] = "dynamic_api_call_sequence"
    out["collection_type"] = "api_sequence"
    out["platform"] = "windows"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    token_cols = [c for c in raw.columns if c.startswith("t_")]
    seq = raw[token_cols].astype(str).agg(" ".join, axis=1).str.lower()

    out["file_api_usage"] = seq.str.contains("file|write|read|create", regex=True).astype(float)
    out["file_write_burst"] = seq.str.contains("write|createfile|ntwrite", regex=True).astype(float)
    out["file_read_burst"] = seq.str.contains("read|openfile|ntread", regex=True).astype(float)
    out["registry_api_usage"] = seq.str.contains("reg|registry|ntopenkey", regex=True).astype(float)
    out["process_api_usage"] = seq.str.contains("process|createprocess|shell", regex=True).astype(float)
    out["memory_access_spike"] = seq.str.contains("virtualmemory|mapview|section|alloc", regex=True).astype(float)
    out["network_api_usage"] = seq.str.contains("internet|connect|send|recv|http", regex=True).astype(float)
    out["c2_beaconing"] = out["network_api_usage"]
    out["anti_analysis"] = seq.str.contains("debug|querysystem|sandbox|vm", regex=True).astype(float)

    # This dataset is malware-only in current project usage.
    out["label"] = "unknown"
    out["response_policy"] = "isolate_and_backup"

    return save_part(out, "api_call_sequences")


def ingest_ugransome2024():
    p = Path("data/raw/ugransome2024/final(2).csv")
    if not p.exists():
        print("[!] UGRansome2024 missing")
        return None

    raw = pd.read_csv(p)
    n = len(raw)
    out = empty_frame(n)

    out["sample_id"] = [f"ugransome_{i}" for i in range(n)]
    out["dataset_source"] = "UGRansome2024"
    out["family"] = raw.get("Family", "unknown").astype(str) if "Family" in raw.columns else "unknown"
    out["behavior_type"] = "network_ransomware_features"
    out["collection_type"] = "network_features"
    out["platform"] = "network"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    out["c2_beaconing"] = norm01(raw.get("Netflow_Bytes", 0))
    out["high_outbound_bytes"] = norm01(raw.get("Netflow_Bytes", 0))
    out["data_exfiltration_pattern"] = norm01(raw.get("Netflow_Bytes", 0))
    out["suspicious_ip_reputation"] = 0.0  # Threats is potentially a post-detection label
    out["suspicious_dns"] = raw.get("Protcol", "").astype(str).str.lower().str.contains("dns").astype(float) if "Protcol" in raw.columns else 0.0
    out["tor_or_proxy_usage"] = raw.get("Port", "").astype(str).str.contains("9050|9150|1080").astype(float) if "Port" in raw.columns else 0.0

    # Prediction may be an upstream model output, not verified ground truth.
    if "Family" in raw.columns:
        out["label"] = raw["Family"].apply(detect_label_from_text)
    else:
        out["label"] = "unknown"

    out["response_policy"] = out["label"].apply(label_to_policy)
    return save_part(out, "ugransome2024")


def ingest_android_ransomware_detection(max_rows=80000):
    p = Path("data/raw/android_ransomware/Android_Ransomeware.csv")
    if not p.exists():
        print("[!] Android_Ransomware_Detection missing")
        return None

    raw = pd.read_csv(p)
    if len(raw) > max_rows:
        raw = raw.sample(n=max_rows, random_state=42).reset_index(drop=True)

    n = len(raw)
    out = empty_frame(n)

    out["sample_id"] = [f"android_ransomware_{i}" for i in range(n)]
    out["dataset_source"] = "Android_Ransomware_Detection"
    out["family"] = "android_ransomware"
    out["behavior_type"] = "android_network_flow_features"
    out["collection_type"] = "network_flow"
    out["platform"] = "android"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    def getcol(name):
        for c in raw.columns:
            if c.strip().lower() == name.strip().lower():
                return raw[c]
        return pd.Series([0] * n)

    out["c2_beaconing"] = norm01(getcol("Flow Packets/s"))
    out["high_outbound_bytes"] = norm01(getcol("Total Length of Fwd Packets"))
    out["data_exfiltration_pattern"] = norm01(getcol("Total Length of Fwd Packets"))
    out["high_outbound_connection_count"] = norm01(getcol("Total Fwd Packets"))
    out["suspicious_ip_reputation"] = norm01(getcol("Destination Port"))

    label_col = None
    for c in raw.columns:
        if c.strip().lower() in ["label", "class", "category", "type"]:
            label_col = c
            break

    if label_col:
        out["label"] = raw[label_col].apply(detect_label_from_text)
    else:
        # This is optional cross-platform ransomware data; mark ransomware-like conservatively.
        out["label"] = "unknown"

    out["response_policy"] = out["label"].apply(label_to_policy)
    return save_part(out, "android_ransomware_detection")


def ingest_cic_malmem2022(max_rows=120000):
    root = Path("data/raw/cic_malmem2022")
    csvs = sorted(root.rglob("*.csv"))

    if not csvs:
        print("[!] CIC_MalMem2022 missing")
        return None

    frames = []

    for p in csvs[:5]:
        try:
            raw = pd.read_csv(p)
        except Exception as e:
            print(f"[!] Cannot read {p}: {e}")
            continue

        if len(raw) == 0:
            continue

        if len(raw) > max_rows:
            raw = raw.sample(n=max_rows, random_state=42).reset_index(drop=True)

        n = len(raw)
        out = empty_frame(n)

        out["sample_id"] = [f"cic_malmem2022_{p.stem}_{i}" for i in range(n)]
        out["dataset_source"] = "CIC_MalMem2022"
        out["family"] = "memory_obfuscated_malware"
        out["behavior_type"] = "memory_forensics_features"
        out["collection_type"] = "memory_dump_features"
        out["platform"] = "windows"
        out["is_simulated"] = 0
        out["is_real_malware_executed"] = 0

        lower_cols = {c.lower().strip(): c for c in raw.columns}

        def find_cols(keys):
            result = []
            for low, orig in lower_cols.items():
                if any(k in low for k in keys):
                    result.append(orig)
            return result

        def score(keys):
            cols = find_cols(keys)
            if not cols:
                return pd.Series([0.0] * n)
            return norm01(
                raw[cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0)
                .sum(axis=1)
            )

        # Memory/process forensic symptoms
        out["memory_access_spike"] = score(["pslist", "psscan", "vadinfo", "malfind", "dlllist"])
        out["memory_entropy_region_high"] = score(["entropy", "malfind", "vad"])
        out["rapid_buffer_write_pattern"] = score(["handles", "mutants", "callbacks"])
        out["process_api_usage"] = score(["pslist", "psscan", "pstree"])
        out["process_tree_anomaly"] = score(["pstree", "psxview", "hidden", "unlinked"])
        out["suspicious_process_spawn"] = score(["pslist", "pstree"])
        out["process_injection_suspected"] = score(["malfind", "injection", "vad"])
        out["anti_analysis"] = score(["callbacks", "modules", "ldrmodules", "ssdt"])
        out["anti_vm"] = score(["vm", "virtual", "driver"])
        out["anti_debugging"] = score(["debug"])
        out["packed_binary"] = score(["malfind", "entropy"])
        out["high_section_entropy"] = score(["entropy"])
        out["suspicious_import_table"] = score(["dlllist", "ldrmodules"])
        out["suspicious_string"] = score(["cmdline", "command", "path"])

        # Network/service/persistence hints
        out["network_api_usage"] = score(["netscan", "connections", "sockets"])
        out["c2_beaconing"] = score(["netscan", "connections"])
        out["service_api_usage"] = score(["svcscan", "services"])
        out["service_created"] = score(["svcscan", "services"])
        out["registry_api_usage"] = score(["hivelist", "printkey", "registry"])
        out["persistence_attempt"] = score(["svcscan", "services", "hivelist", "registry"])

        # Label mapping
        label_col = None
        for c in raw.columns:
            lc = c.lower().strip()
            if lc in ["class", "category", "label", "type", "malware"]:
                label_col = c
                break

        if label_col:
            out["label"] = raw[label_col].apply(detect_label_from_text)
            out["family"] = raw[label_col].astype(str)
        else:
            # CIC-MalMem is malware/benign memory dump dataset; if no label col, mark review.
            out["label"] = "unknown"

        out["response_policy"] = out["label"].apply(label_to_policy)
        frames.append(out)

    if not frames:
        print("[!] CIC_MalMem2022 no usable CSV")
        return None

    return save_part(pd.concat(frames, ignore_index=True), "cic_malmem2022")


def ingest_bodmas_npz(max_rows=120000):
    npz_path = Path("data/raw/bodmas/bodmas.npz")
    meta_path = Path("data/raw/bodmas/bodmas_metadata.csv")
    cat_path = Path("data/raw/bodmas/bodmas_malware_category.csv")

    if not npz_path.exists():
        print("[!] BODMAS npz missing")
        return None

    import numpy as np

    data = np.load(npz_path, allow_pickle=False)
    keys = list(data.files)
    print(f"[+] BODMAS npz keys: {keys}")

    X_key = None
    y_key = None

    for candidate in ["X", "x", "features", "data", "arr_0"]:
        if candidate in keys:
            X_key = candidate
            break

    for candidate in ["y", "Y", "labels", "label", "arr_1"]:
        if candidate in keys:
            y_key = candidate
            break

    if X_key is None:
        raise SystemExit(f"[!] Cannot find feature key in BODMAS npz. Keys={keys}")

    X = data[X_key]

    if y_key is not None:
        y = data[y_key]
    else:
        y = None

    n = X.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=max_rows, replace=False)
        X = X[idx]
        if y is not None:
            y = y[idx]
    else:
        idx = np.arange(n)

    n = X.shape[0]
    out = empty_frame(n)

    out["sample_id"] = [f"bodmas_{i}" for i in range(n)]
    out["dataset_source"] = "BODMAS"
    out["family"] = "unknown_bodmas_family"
    out["behavior_type"] = "static_pe_ember_features"
    out["collection_type"] = "static_pe_features_npz"
    out["platform"] = "windows"
    out["is_simulated"] = 0
    out["is_real_malware_executed"] = 0

    # Convert feature matrix to DataFrame for coarse symptom mapping.
    # BODMAS/EMBER feature indices are static PE numeric features, so we map broad statistics to symptoms.
    Xdf = pd.DataFrame(X)
    numeric_sum = norm01(Xdf.apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1))
    numeric_mean = norm01(Xdf.apply(pd.to_numeric, errors="coerce").fillna(0).mean(axis=1))
    numeric_max = norm01(Xdf.apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1))
    numeric_std = norm01(Xdf.apply(pd.to_numeric, errors="coerce").fillna(0).std(axis=1))

    out["packed_binary"] = numeric_std
    out["high_section_entropy"] = numeric_max
    out["suspicious_import_table"] = numeric_sum
    out["low_import_count_packed"] = 1 - numeric_mean
    out["suspicious_section_name"] = numeric_std
    out["suspicious_string"] = numeric_sum * 0.5
    out["crypto_api_usage"] = numeric_mean
    out["file_api_usage"] = numeric_mean
    out["process_api_usage"] = numeric_mean
    out["registry_api_usage"] = numeric_mean * 0.3
    out["network_api_usage"] = numeric_mean * 0.3
    out["anti_analysis"] = numeric_std
    out["anti_debugging"] = numeric_std * 0.5
    out["anti_vm"] = numeric_std * 0.5

    if y is not None:
        ys = pd.Series(y).astype(str).str.lower()
        # Binary malware labels do not establish ransomware ground truth.
        out["label"] = np.where(ys.isin(["0", "benign", "goodware", "clean"]), "benign", "unknown")
    else:
        out["label"] = "unknown"

    # Optional metadata enrichment if row count aligns.
    if meta_path.exists():
        try:
            meta = pd.read_csv(meta_path)
            if len(meta) >= len(out):
                fam_col = None
                for c in meta.columns:
                    if c.lower() in ["family", "category", "malware_family"]:
                        fam_col = c
                        break
                if fam_col:
                    out["family"] = meta.iloc[idx][fam_col].astype(str).values
        except Exception as e:
            print(f"[!] BODMAS metadata enrichment skipped: {e}")

    out["response_policy"] = out["label"].apply(label_to_policy)

    return save_part(out, "bodmas")


if __name__ == "__main__":
    main()
