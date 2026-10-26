from pathlib import Path
import json
import pandas as pd
from datetime import datetime

OUT = Path("reports/demo/demo_summary.md")
OUT.parent.mkdir(parents=True, exist_ok=True)

UNIFIED = Path("data/symptom_labels/unified_symptom_dataset.csv")
TRAINING_REPORT = Path("reports/symptom_ai/training_report.json")
MODEL_META = Path("models/model_metadata.json")


def safe_read_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# AI Ransomware Symptom Detection Demo Summary")
    lines.append("")
    lines.append(f"Generated at: `{now}`")
    lines.append("")

    if UNIFIED.exists():
        df = pd.read_csv(UNIFIED, usecols=["dataset_source", "label", "response_policy"])
        lines.append("## 1. Unified Dataset Status")
        lines.append("")
        lines.append(f"- Total rows: **{len(df):,}**")
        lines.append(f"- Dataset sources: **{df['dataset_source'].nunique()}**")
        lines.append("")
        lines.append("### Dataset Source Counts")
        lines.append("")
        lines.append("| Dataset | Rows |")
        lines.append("|---|---:|")
        for k, v in df["dataset_source"].value_counts().items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")
        lines.append("### Label Counts")
        lines.append("")
        lines.append("| Label | Rows |")
        lines.append("|---|---:|")
        for k, v in df["label"].value_counts().items():
            lines.append(f"| {k} | {v:,} |")
        lines.append("")
    else:
        lines.append("## 1. Unified Dataset Status")
        lines.append("")
        lines.append("- Unified dataset not found yet.")
        lines.append("")

    meta = safe_read_json(MODEL_META)
    report = safe_read_json(TRAINING_REPORT)

    lines.append("## 2. AI Model Status")
    lines.append("")
    if meta:
        lines.append(f"- Classifier: **{meta.get('primary_classifier')}**")
        lines.append(f"- Unknown detector: **{meta.get('models', {}).get('unknown_detector')}**")
        lines.append(f"- Training rows: **{meta.get('training_rows'):,}**")
        lines.append(f"- Feature count: **{meta.get('feature_count')}**")
        lines.append(f"- Labels: `{', '.join(meta.get('labels', []))}`")
    else:
        lines.append("- Model metadata not found yet.")
    lines.append("")

    if report:
        lines.append("## 3. Training Result")
        lines.append("")
        lines.append("Held-out test results (selection uses validation macro F1):")
        lines.append("")
        for name in ("random_forest", "xgboost"):
            result = report.get(name, {})
            lines.append(f"- {name}: macro F1={result.get('f1')}; benign false-positive rate={result.get('false_positive_rate')}")
            lines.append(f"  Confusion matrix ({result.get('confusion_matrix_labels')}): {result.get('confusion_matrix')}")
        lines.append("")
        lines.append("These results depend on source labels and provenance. Anomalies do not prove zero-day detection.")
        lines.append("")
    else:
        lines.append("## 3. Training Result")
        lines.append("")
        lines.append("- Training report not found yet.")
        lines.append("")

    lines.append("## 4. API Demo Endpoints")
    lines.append("")
    lines.append("```bash")
    lines.append("curl http://localhost:8000/health")
    lines.append("curl http://localhost:8000/model/info | python3 -m json.tool")
    lines.append("```")
    lines.append("")
    lines.append("### Known ransomware-like case")
    lines.append("")
    lines.append("```bash")
    lines.append("""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "symptoms": {
      "file_write_burst": 0.95,
      "file_rename_burst": 0.90,
      "high_entropy_write": 0.92,
      "mass_file_modification": 0.87,
      "suspicious_extension_change": 0.83,
      "suspicious_process_spawn": 0.75
    }
  }' | python3 -m json.tool""")
    lines.append("```")
    lines.append("")
    lines.append("### Unknown high-risk case")
    lines.append("")
    lines.append("```bash")
    lines.append("""curl -X POST http://localhost:8000/respond \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: $SYMPTOM_AI_API_KEY" \\
  -d '{
    "symptoms": {
      "file_write_burst": 0.82,
      "high_entropy_write": 0.91,
      "backup_disable_attempt": 0.88,
      "shadow_copy_delete_attempt": 0.82,
      "security_tool_tamper": 0.80,
      "c2_beaconing": 0.76,
      "data_exfiltration_pattern": 0.74
    }
  }' | python3 -m json.tool""")
    lines.append("```")
    lines.append("")

    lines.append("## 5. Explanation for Report")
    lines.append("")
    lines.append(
        "The system normalizes multiple ransomware datasets into a unified symptom schema. "
        "Each row represents one behavior sample with standardized symptoms such as file write burst, "
        "high entropy write, registry modification, API usage, C2 beaconing, backup tampering, and storage activity. "
        "The Random Forest classifier predicts whether the behavior is benign or ransomware-like, while Isolation Forest "
        "supports anomaly scoring. The response engine recommends actions and records a safe simulation; "
        "it does not execute host isolation, process termination, or file permission changes."
    )
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Demo summary saved to {OUT}")


if __name__ == "__main__":
    main()
