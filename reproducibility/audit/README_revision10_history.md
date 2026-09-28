# Audited manuscript and code, revision 10

This package contains the revised paper, two revised notebooks, extracted analysis scripts, and the local reproduction results. Original attachments and source archives were not modified. 

## Revision 10 validation update

The paper incorporates the pilot-excluded 55-object intervention, both coordinate conventions, fit acceptance counts, all-60 sensitivity, and exploratory correction-model comparisons. The original executed Section 24 notebook and its outputs are archived unchanged. The revised original notebooks mentioned below are separate, older artifacts; their execution status does not describe the archived Section 24 notebook.

New paths:
- `data/section24_original/`: supplied results, preserved unchanged.
- `data/section24_corrected/`: corrected per-object summaries and audit.
- `data/section24_model_corrected/`: locally refitted model sensitivity with fold-specific ridge scaling.
- `code/section24_recalculate.py`: portable summary correction.
- `code/section24_model_sensitivity_corrected.py`: corrected model comparison (NumPy ridge CV).
- `notebooks/HIT_section24_corrections_v10.ipynb`: unexecuted portable rerun notebook.

Run from the package root:

```sh
python code/section24_recalculate.py --input data/section24_original --out data/section24_corrected
python code/section24_model_sensitivity_corrected.py --out-root data --dev-audit data/development_geometry.csv --frozen-dir data/section24_local_inputs --out data/section24_model_corrected
```

The local ridge check reproduced the retained model across all 2,728 second-batch rows. Changing CV scaling did not change the chosen alpha or displayed results. The original expensive rendering/fusion was not rerun. Pilot removal is retrospective; neither the intervention nor model sensitivity constitutes new independent-source validation. Section 24 environment records are separate from the older reproduction environment described below.

## Start here

- `paper/manuscript_revised_v10.pdf`: compiled manuscript.
- `paper/main.tex` and `paper/references.bib`: editable manuscript source.
- `notebooks/HIT_3DMSL_MitoEM_R_FULL_section20_1_audited_v5.ipynb`: revised full analysis notebook.
- `notebooks/HIT_section22_paper_fixes_audited_v5.ipynb`: revised follow-up notebook.
- `CHANGELOG.md`: scientific and code changes, verification, and unresolved limitations.
- `data/verified_summary.json`: reproduced primary correction results and additional diagnostics.
- `data/input_manifest.json`, `data/environment.json`, and `OUTPUT_SHA256.json`: input provenance, tested environment, and output checksums.

The notebooks retain the original Colab workflow and input configuration cells. Their saved outputs have been cleared; they have **not** been executed end to end as revised notebooks. The numerical reruns described below were performed with the supplied analysis code and the standalone reproduction scripts. Do not interpret cleared notebooks as execution records. Keep the original supplied notebooks for the earlier recorded outputs.

## What was reproduced

Mesh and occupancy measurements were recomputed directly from both original 3DMSL archives: 2,720 development and 2,728 second-batch objects. A deterministic reconstruction using the notebook's actual seed, **2026**, reproduced the internal split, correction results, calibration bounds, and second-batch coverage counts to the reported precision. The reconstruction is explicitly labeled: the original time-stamped frozen model file was not available for byte comparison. Reconstruction does not independently prove the chronology of the original analysis.

The original 60 boundary-analysis IDs were replayed. Primary results retain the 57 originally accepted IDs; current optimizer status is stored separately. The added MitoEM-R analysis remeshed all 550 eligible objects under 17 phase/tie configurations, yielding 9,350 object/configuration results. These added analyses are exploratory.

The entire original workflow, including simulated imaging and all point-cloud sampling experiments, was not rerun in this revision. Those retained results originate in the supplied notebook records. Raw archives are not redistributed in this package.

## Environment

Python 3.12 was used. Exact observed package versions are in `requirements.txt` and `data/environment.json`. Use an isolated environment:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the commands below from this package's root directory. All paths to raw archives are explicit placeholders and should be replaced with your local paths. The independent geometry and resolution jobs use multiple processes and may take substantial time. The boundary replay is also computationally expensive. Use fewer `--workers` when memory is limited.

## Reproduce the primary volume correction

```sh
python code/recompute_geometry.py --zip /path/to/10_24553_27272.zip --out data/development_geometry.csv --workers 3
python code/recompute_geometry.py --zip /path/to/1_1_2729.zip --out data/second_batch_geometry.csv --workers 3
python code/original_extracted/validation_v3.py --in data/development_geometry.csv --outdir data/reconstructed_development --seed 2026 --n-boot 2000
python code/summarize_revision.py
```

`original_extracted/` holds the historical code extracted from the supplied notebooks. It is used deliberately for faithful reconstruction of the original numerical pipeline. Use `code/revised/` or the revised notebooks for future analyses. The revised calibration function corrects the small-calibration-sample edge case; that edge case does not occur with the 540 calibration objects here. Additional metadata written by the revised function can change a model file's byte hash without changing this study's coefficients or bounds.

The second-batch bootstrap reproduces the recorded generator ordering: one generator seeded 790, processing methods in their saved order. Internal validation uses seed 2026 and method-specific offsets. Object-level intervals do not quantify between-animal variation.

## Replay boundary fits

Extract only `raw_mesh.off` and `occupancies.npz` for the IDs listed in `data/original_boundary_ids.json` from the development archive. Arrange them as `/path/to/geometry/INSTANCE_ID/raw_mesh.off` and `/path/to/geometry/INSTANCE_ID/occupancies.npz`, retaining the recorded ID order in the JSON file.

```sh
python code/recheck_boundary.py --geometry-root /path/to/geometry --workers 2
```

`original_accepted` preserves the historical exclusions 25111, 26173, and 26662. `exact_converged` records the local replay decision. All 60 locally converged; the original 57-object summary remains primary. Numerical optimizer decisions can change across environments.

## Reproduce phase, tie, and topology checks

```sh
python code/resolution_robustness.py --label-zip /path/to/EM30-R-mito-train-val-v2.zip --workers 2
python code/generate_revision_figures.py
```

The packaged `data/resolution/instance_selection.csv` records all 1,203 intersecting IDs and eligibility fields. The script verifies the ordered exclusion flow against that table, reconstructs masks from the original labels, and remeshes the eligible objects. To independently rebuild the selection table, use `code/revised/mitoem_external_validation_v2.py` with its documented `--help` options or the full notebook. This additional check does not estimate lower-resolution acquisition effects.

## Tests and manuscript compilation

```sh
python code/test_revision.py
```

Seven focused tests cover saved-field extraction, rejection of inconsistent global offsets, corrected-residual algebra, conformal order statistics and small-sample behavior, interval inversion, original-phase equivalence, and odd-block tie handling.

Upload the contents of `paper/` to Overleaf, set `main.tex` as the main document, and select pdfLaTeX. The supplied PDF was built with Tectonic, which ran the bibliography and reference passes. All 11 rendered pages were visually inspected. The source needs IEEEtran, standard mathematics/table packages, and the included figure PDFs. Python is not required to compile the paper.

## Provenance and publication limits

No public repository or DOI deposit was created. A controlled label-generation offset-on/offset-off experiment was not performed. The manuscript now presents the watertighting explanation as a plausible mechanism, identifies the half-shift estimate as a heuristic, and distinguishes these unresolved questions from the observed boundary displacement. Exact cross-batch mesh-file hashes do not establish biological independence or rule out equivalent geometry with different encodings.

## Revision 7 history

Revision 7 applied the v6 audit while retaining the v6 title, shorter abstract, and compact performance table. The corrected v5 notebooks and numerical reproduction tables are retained unchanged; their filenames still identify them as v5. No new end-to-end notebook execution is claimed.

The boundary figure is generated from the packaged replay table. It shows fitted transitions for the 57 historically accepted objects and adds open markers for the three historically rejected fits using their local replay estimates. It does not reproduce the v6 pooled-query plot, whose generator and pooled bin table were absent from the supplied ZIP. The caption explicitly describes the plotted quantities.

`data/boundary/influence_check.json` records the new exploratory leave-largest-offset-out correlation check. Object 25660 remains included in primary results. The grid-phase figure is now included in Appendix B. The subsequent reference-by-reference audit and corrections are documented in audit/AUDIT_v8.md.

## Reference audit (revision 8)

See `audit/AUDIT_v8.md` and `audit/reference_audit.csv` for verified identities, access limits, claim corrections, and final reference numbering. All 33 final entries have URLs. Some publisher endpoints restrict automated access; unrestricted full-text access is not claimed. Scientific data, figures, notebooks, and analysis code are unchanged from revision 8.

## Revision 9: manuscript and submission planning

The title now foregrounds representation bias, correction transfer and resolution sensitivity. The abstract separates within-resource correction transfer from the MitoEM resolution experiment. The methods explicitly distinguish the all-query bounding box from the inside-point bounding box. The raw/global comparators and mesh-informed diagnostic have clearer roles. Homeostatic Invariance remains a proposed application in the discussion.

The duplicate resolution plot was removed from the article; Table VI retains its numerical values and quantiles. The former Figure 5 is now Figure 4. The article has six tables, four figures, and 33 bibliography entries. All labels and citations resolve. Historical plot assets and numerical tables are retained in the package. The master remains in its IEEE working layout, not a claim of compliance with a selected journal's template.

- `submission/Stronger_Validation_Plan.md`: proposed reproduction, mechanism intervention, new-source testing and model sensitivity. These experiments have not been performed in this revision.
- `submission/Journal_Shortlist.md`: subject-fit-first ranking, verified costs, access limitations and target-specific adaptations.
- `submission/Submission_Notes.md`: lay description, practitioner points, shorter abstract and author-provided information needed for submission.
- `audit/REVISION_9_CHANGES.md`: changes, verification scope and remaining limits.
- `audit/reference_audit_v9.csv`: current printed bibliography numbering; reference-access findings are carried forward from revision 8, not re-fetched comprehensively.

Analysis source, notebooks, numerical data and model artifacts are unchanged. Incidental Python caches were excluded from this release. No end-to-end clean-environment notebook run, external validation, controlled label-generation intervention, submission or public deposit was performed in revision 10.
