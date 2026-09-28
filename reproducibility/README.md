# Mitochondrial morphometry — revision 14

## Current manuscript and evidence

- `paper/manuscript_revised_v14.pdf`: current compiled paper.
- `paper/main.tex`, `paper/references.bib`, `paper/figures/`: editable Overleaf project.
- `data/original_frozen_run/`: newly supplied original run 20260924T193719_753199Z, preserved unchanged.
- `audit/original_frozen_verification.json`: full model hash, five protocol-input hash checks and direct prediction replay.
- `RESULT_FILE_MAP.md`: table/figure and experiment links to supporting files.
- `PRIORITY_STATUS.md`: completed items and closure of the three requested priorities.

## What is resolved

The original model SHA-256 is `93b6daed1431d5cd11d83d8f90e1f4c9cec2e466866fee58f6d1e63f6be8be7f`, matching the protocol and earlier Section 24 environment record. All five protocol-listed input files match. Applying the model to its archived audit table without refitting reproduces all 2,728 occupancy-only residuals to within 6.67e-14 percentage points. Original and reconstructed artifacts are retained separately.

Table III now includes E8 (55-object offset intervention), E9 (ten-object hybrid check), and E10 (exploratory model sensitivity). Table II identifies reused subsets. The manuscript scopes the original saved-model evaluation separately from subsequent exploratory use of the second batch.

## Section 25 evidence gap resolved

The original Section 25 output folder is now archived unchanged in `data/section25_original/`. All 40 condition records are complete and unique (10 objects × 2 offsets × 2 conventions); all 20 centred-convention hybrid records contain the required metrics. Independent reaggregation exactly matches the saved JSON and notebook summaries. The interpolation errors match the earlier Section 24 rows exactly. The model-comparison table matches the independent local MSE-based rerun to numerical precision.

Verification files: `audit/section25_verification.json`, `audit/section25_verified_summary.json`, and `audit/section25_input_sha256.json`. The new hash manifest is an audit-time integrity record, not a claim that those hashes were recorded before the experiment.

All three requested priorities are now addressed within the supplied evidence package. Full mesh-containment equivalence, independent-source validation and a complete raw-data rerun remain separate scientific limitations. No public repository publication or DOI is claimed.

## Verify locally

From this package root, with NumPy and pandas:

```sh
python code/verify_original_frozen.py --run data/original_frozen_run --out audit
python code/verify_section25_outputs.py
python code/section24_recalculate.py --input data/section24_original --out data/section24_corrected
python code/section25_model_mse_recheck.py --out-root data --dev-audit data/development_geometry.csv --frozen-dir data/original_frozen_run --out data/section25_model_mse_recheck
```

The first command verifies and applies the original model, without fitting. The last command is a separate exploratory refit. Exact raw-data reconstruction instructions and earlier run histories remain under `audit/README_revision10_history.md` and `audit/README_revision12_history.md`; earlier claims about unavailable model bytes are historical and superseded here. A public repository/deposition has not been created. The ZIP is ready to supply as supplementary material after venue requirements are checked.
