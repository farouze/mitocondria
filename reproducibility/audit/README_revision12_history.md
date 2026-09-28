# Mitochondrial morphometry — revision 12

Revision 12 corrects the interpretation and provenance wording in the supplied revision 11 manuscript. No new experiment or model-selection decision was made for this revision.

## Current files

- `paper/manuscript_revised_v12.pdf`: compiled current manuscript.
- `paper/main.tex`, `paper/references.bib`, `paper/figures/`: Overleaf sources.
- `notebooks/HIT_section25_audit_followups_executed.ipynb`: unchanged supplied notebook containing the Section 25 run outputs. Execution counters are null, but outputs are present.
- `data/section25_recorded/`: extracted outputs, numerical summary, verification record and report.
- `data/section25_model_mse_recheck/`: independent local MSE-based ridge comparison performed during the revision 11 audit.
- `code/section25_original/`: unchanged scripts extracted from the supplied Section 25 notebook. These are provenance records, not newly corrected implementations.
- `code/section25_model_mse_recheck.py`: local NumPy replication of the MSE-based ridge comparison.
- `audit/revision_v12_checks.json`: final source/PDF checks.
- `CHANGELOG.md`: current corrections and revision history.

## What changed

The ten-object check is described as hybrid labeling: winding-number containment replaces labels only where the absolute interpolated fused-field value is below 2/256; labels elsewhere remain interpolated. This is not a verified Euclidean-distance band or full mesh-containment test. Subset selection is described as exploratory, without a prospective-selection claim.

The ridge follow-up changed both preprocessing and the selection objective. Corrected fold-specific scaling with mean fold R-squared selects alpha 0.3981; with mean squared error it selects 0.2512. Revision 12 retains the verified MSE-selected results reported in revision 11.

Residual boundary displacement is not assigned to specific untested fusion steps. The original frozen model's reported hash is distinguished from possession of its bytes; local coefficients remain identified as reconstructions.

## Evidence limits

The ten-object values are supported by saved notebook outputs. Its interpolation mean also exactly matches the prior Section 24 per-object data. New containment-specific per-object CSVs were not supplied, so those summaries have not been independently reaggregated. The recorded original-model hash is available; the matching original model JSON was not supplied. No exact historical-pipeline reproduction, independent-source validation, or clean full-workflow rerun is claimed.

## Reproduce the inexpensive checks

From this package root, with NumPy and pandas installed:

```sh
python code/section24_recalculate.py --input data/section24_original --out data/section24_corrected
python code/section25_model_mse_recheck.py --out-root data --dev-audit data/development_geometry.csv --frozen-dir data/section24_local_inputs --out data/section25_model_mse_recheck
```

Original Section 24 outputs and earlier model comparisons remain preserved under their existing directories. The revision 10 notebook and R-squared comparison are historical artifacts, not the current MSE-selected comparison. Earlier archive-based reproduction instructions are retained in `audit/README_revision10_history.md`. Raw datasets are not redistributed.
