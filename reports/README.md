# Evaluation provenance

The tracked reports, JSONL inputs and figures in `evaluation/` and `evaluation_v2/`
were introduced by PR #1 and are preserved unchanged during integration with
PR #2. They describe the earlier implementation and feature representation.
They are historical evidence, not new schema-2 results or deployment accuracy.
The original external datasets and model artifacts are not present in this lab.

Regenerate source features, splits, models, calibration and reports together before
comparing current models. Old shadow artifacts/reports are intentionally rejected
by schema-2 inference. Do not combine legacy and current family/ablation results;
summary scripts reject mixed versions. Current synthetic verification outcomes and
remaining limitations are documented in `docs/INTEGRATION_VERIFICATION.md` at the
repository root. New verification runs use temporary directories and do not
replace these historical artifacts.
