# Result-to-file map (revision 14)

Paths below are relative to this reproducibility package. Original data and current verification are distinct. Run scripts only with their documented inputs; many original scripts retain Colab paths.

| Paper item | Evidence / inputs | Code / provenance |
|---|---|---|
| Table I: related work | `paper/references.bib`; `audit/reference_audit.csv` | Literature synthesis; not a rerun of cited models |
| Table II: datasets | `data/input_manifest.json`; `data/resolution/selection_flow.json`; `data/original_frozen_run/inclusion_audit.csv` | `code/recompute_geometry.py`; `code/resolution_robustness.py` |
| Table III: E1–E10 | Methods and the rows below | Organizational map, not prospectively registered experiments |
| Table IV: original correction | `data/original_frozen_run/frozen_inputs/validation_v3_model.json`; `data/original_frozen_run/frozen_predictions.csv`; `data/reconstructed_development/validation_v3_predictions.csv` | `code/verify_original_frozen.py`; original saved model and separate internal reconstruction |
| Table V: offset intervention | `data/section24_corrected/nonpilot55_per_condition.csv`; `data/section24_corrected/corrected_summary.json` | `code/section24_recalculate.py`; executed `notebooks/HIT_section24_validation_executed.ipynb` |
| Table VI: subgroup coverage | `data/original_frozen_run/frozen_predictions.csv`; `data/coverage_groups.csv` | `code/verify_original_frozen.py`; `data/coverage_recheck/coverage_check.json` |
| Table VII: regridding | `data/resolution/summary.csv`; `data/resolution/phase_tie_topology.csv` | `code/resolution_robustness.py` |
| Table VIII: model sensitivity | `data/section25_original/model_sensitivity_v2/`; independent local rerun in `data/section25_model_mse_recheck/` | `code/section25_model_mse_recheck.py`; supplied Section 25 notebook |
| Figure 1: agreement | `data/development_geometry.csv`; `data/verified_summary.json` | `code/generate_revision_figures.py` |
| Figure 2: boundary | `data/boundary/summary.json`; `data/original_boundary_ids.json`; boundary per-object CSVs | `code/recheck_boundary.py`; `code/generate_revision_figures.py` |
| Figure 3: frozen errors | Original run audit/model/predictions; `paper/figures/frozen_ecdf.pdf` | `code/original_extracted/frozen_figure_and_coverage.py`; `code/generate_revision_figures.py` |
| Figure 4: phase sensitivity | `data/resolution/summary.csv`; `data/resolution/phase_tie_topology.csv` | `code/resolution_robustness.py` |
| E8: 55-object intervention | Same evidence as Table V; original all-60 rows in `data/section24_original/recreation_full/` | Pilot excluded retrospectively; not an independent sample |
| E9: ten-object hybrid check | `data/section25_original/mesh_check/recreation_per_condition.csv`; `data/section25_original/mesh_check/mesh_check_summary.json`; executed Section 25 notebook | `code/section25_original/label_pipeline_recreation.py`; `code/verify_section25_outputs.py` independently reaggregates the supplied CSV; verified summaries in `audit/section25_verified_summary.json` |
| E10: model sensitivity | Same evidence as Table VIII | MSE-selected ridge is exploratory; older R-squared results preserved separately |
| S1: optical imaging | Recorded original notebook analyses, not a new raw-data rerun | `code/original_extracted/imaging_v2.py`; simulator provenance remains unresolved |

## Original versus reconstructed model

Authoritative supplied run artifact: `data/original_frozen_run/frozen_inputs/validation_v3_model.json`.

Separately reconstructed artifact: `data/reconstructed_development/validation_v3_model.json`. Its numerical behavior matches, but it is not a byte-identical original file. `audit/original_frozen_verification.json` documents verification of the supplied original and preserves the full hash. A hash match establishes consistency with the recorded artifact, not the chronology of every analysis decision.
