# Mitochondrial morphometry — revision 13

## Current manuscript and evidence

- `paper/manuscript_revised_v13.pdf`: current compiled paper.
- `paper/main.tex`, `paper/references.bib`, `paper/figures/`: editable Overleaf project.
- `data/original_frozen_run/`: newly supplied original run 20260924T193719_753199Z, preserved unchanged.
- `audit/original_frozen_verification.json`: full model hash, five protocol-input hash checks and direct prediction replay.
- `RESULT_FILE_MAP.md`: table/figure and experiment links to supporting files.
- `PRIORITY_STATUS.md`: completed items and one outstanding evidence gap.

## What is resolved

The original model SHA-256 is `93b6daed1431d5cd11d83d8f90e1f4c9cec2e466866fee58f6d1e63f6be8be7f`, matching the protocol and earlier Section 24 environment record. All five protocol-listed input files match. Applying the model to its archived audit table without refitting reproduces all 2,728 occupancy-only residuals to within 6.67e-14 percentage points. Original and reconstructed artifacts are retained separately.

Table III now includes E8 (55-object offset intervention), E9 (ten-object hybrid check), and E10 (exploratory model sensitivity). Table II identifies reused subsets. The manuscript scopes the original saved-model evaluation separately from subsequent exploratory use of the second batch.

## Remaining evidence gap

The Section 25 hybrid-containment values are supported by the saved executed notebook, but the new per-object containment CSVs were not included in the supplied folder. This is disclosed in the manuscript and file map. The original frozen-model gap is resolved; this separate gap is not. Do not substitute the Section 24 recreation CSVs, which lack the new hybrid columns.

Needed from `HIT_outputs_verified/validation/section25_20260926T223354Z/`: `mesh_check/recreation_per_condition.csv`, `mesh_check/mesh_check_summary.json`, and preferably its protocol and full model-sensitivity output folder. A ZIP of that Section 25 run folder is sufficient. Raw meshes need not be uploaded again.

## Verify locally

From this package root, with NumPy and pandas:

```sh
python code/verify_original_frozen.py --run data/original_frozen_run --out audit
python code/section24_recalculate.py --input data/section24_original --out data/section24_corrected
python code/section25_model_mse_recheck.py --out-root data --dev-audit data/development_geometry.csv --frozen-dir data/original_frozen_run --out data/section25_model_mse_recheck
```

The first command verifies and applies the original model, without fitting. The last command is a separate exploratory refit. Exact raw-data reconstruction instructions and earlier run histories remain under `audit/README_revision10_history.md` and `audit/README_revision12_history.md`; earlier claims about unavailable model bytes are historical and superseded here. A public repository/deposition has not been created. The ZIP is ready to supply as supplementary material after venue requirements are checked.
