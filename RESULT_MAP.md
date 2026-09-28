# Revision 18 result-to-file map

Paths are relative to the repository root. All code and data below are included in the prepared package. Historical maps inside `reproducibility/` retain revision 14 numbering.

| Current manuscript item | Evidence and code |
|---|---|
| Table I: related work | `manuscript/references.bib`; `reproducibility/audit/reference_audit.csv` |
| Table II: E1–E8 design | Methods; `reproducibility/data/input_manifest.json`; `reproducibility/data/resolution/selection_flow.json` |
| Table III: correction performance | `reproducibility/data/original_frozen_run/frozen_predictions.csv`; `reproducibility/data/reconstructed_development/validation_v3_predictions.csv`; `reproducibility/code/verify_original_frozen.py` |
| Table IV: 55-object offset test | `reproducibility/data/section24_corrected/nonpilot55_per_condition.csv`; `reproducibility/data/section24_corrected/corrected_summary.json`; `reproducibility/code/section24_recalculate.py` |
| Table V: subgroup coverage | `reproducibility/data/coverage_groups.csv`; `reproducibility/data/coverage_recheck/coverage_check.json`; `reproducibility/code/verify_original_frozen.py` |
| Table VI: regridding | `reproducibility/data/resolution/summary.csv`; `reproducibility/code/resolution_robustness.py` |
| Table VII / S2: model comparison | `reproducibility/data/section25_original/model_sensitivity_v2/model_sensitivity_table.csv`; `reproducibility/code/section25_model_mse_recheck.py` |
| Figure 1: boundary | `reproducibility/data/boundary/summary.json`; `reproducibility/data/boundary/replayed_fits.csv`; `reproducibility/code/recheck_boundary.py`; `reproducibility/code/generate_revision_figures.py` |
| Figure 2: frozen-error distributions | `reproducibility/data/original_frozen_run/frozen_predictions.csv`; `reproducibility/code/generate_revision_figures.py` |
| Figure 3: offset dose response | `reproducibility/data/section24_original/recreation_full/recreation_per_condition.csv`; `reproducibility/code/section24_original_label_pipeline_recreation.py`. Figure includes all 60 objects; caption discloses this. Primary table excludes five pilot objects. |
| Figure 4 / S3: grid phase and tie rules | `reproducibility/data/resolution/phase_tie_topology.csv`; `reproducibility/data/resolution/summary.csv`; `reproducibility/code/resolution_robustness.py` |
| Figure 5: agreement diagnostics | `reproducibility/data/development_geometry.csv`; `reproducibility/data/verified_summary.json`; `reproducibility/code/generate_revision_figures.py` |
| E8 hybrid check | `reproducibility/data/section25_original/mesh_check/recreation_per_condition.csv`; `reproducibility/code/section25_original/label_pipeline_recreation.py`; `reproducibility/code/verify_section25_outputs.py` |
| S1 imaging | `reproducibility/code/original_extracted/imaging_v2.py`; recorded notebooks in `reproducibility/notebooks/`. Historical outputs, not a new raw-data rerun. |
| Appendix analysis history: script provenance | `reproducibility/data/section24_original/environment_record.json`; `reproducibility/code/section24_original_label_pipeline_recreation.py` |
| Boundary replay and influence | `reproducibility/data/boundary/replayed_fits.csv`; `reproducibility/data/boundary/influence_check.json`; `reproducibility/code/recheck_boundary.py` |

The original frozen model is `reproducibility/data/original_frozen_run/frozen_inputs/validation_v3_model.json`. The separately reconstructed model is retained for comparison and is not substituted for it. The verification records establish numerical and file consistency; they do not independently establish the chronology of every decision.
