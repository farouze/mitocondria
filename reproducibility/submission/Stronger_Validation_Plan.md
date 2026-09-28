# Revision 14 update

Section 25 per-object containment outputs are now archived and independently reaggregated, resolving the remaining artifact gap. All three requested priorities are complete within the package. Full containment equivalence, independent-source validation and a clean end-to-end rerun remain separate scientific extensions. Earlier entries below are historical.

# Revision 13 update

Original frozen-model bytes, predictions and protocol are now archived and verified. This resolves that earlier artifact gap. Section 25 containment per-object files, full containment equivalence, independent-source validation and a clean complete workflow rerun remain outstanding. Earlier progress entries below are historical.

# Revision 12 progress

The saved Section 25 outputs now support the reported ten-object hybrid check and MSE-selected ridge results. Revision 12 limits interpretation to the band-restricted check and discloses the selection-objective change. Full containment equivalence, independent reaggregation of the new containment CSVs, original frozen-model bytes, independent-source validation, and a clean full-workflow rerun remain outstanding. Prior plans below are historical.

# Revision 10 progress

The controlled reimplementation and exploratory model comparisons are now reported, with retrospective pilot exclusion and both conventions. Ridge fold-specific scaling was corrected and rerun locally. Boundary summaries use accepted endpoint pairs. Independent-source validation, released-label fit diagnostics, exact historical pipeline equivalence, and a clean full-workflow rerun remain outstanding. The original plan below is retained as history.

# Stronger validation plan

Prepared September 25, 2026 for revision 9. **Proposed work only: none of the experiments below has been executed for this revision.** The published numerical results remain unchanged. This plan is not a preregistration until the final protocol, input IDs, code versions and analysis choices are locked before inspecting the new outcomes.

## Recommended minimum sequence

| Priority | Work | Question answered | Deliverable |
|---|---|---|---|
| 1 | Clean-environment reproduction | Can another researcher reproduce the claimed results? | Execution record, environment, input hashes and result comparisons |
| 2 | Paired label-generation intervention | Does the depth offset cause a measurable boundary/volume change in the recreated pipeline? | Paired effect estimates and explicit limits on historical attribution |
| 3 | New-source frozen evaluation | Does the unchanged correction work on an acquisition not used for development? | Per-source errors, coverage, failure rates and applicability limits |
| 4 | Development-only model sensitivity | Are the five features and unregularized model necessary? | Small, controlled comparator table; clearly exploratory where old outcomes are reused |

For a narrowly scoped measurement paper, clean reproduction plus the controlled intervention is a practical first package. For a general correction-transfer claim, the new-source evaluation is essential. Neither package tests biological Homeostatic Invariance; that would require a separate morphology–function study.

## 1. Reproduction and unit checks

Start from the released revision 9 package in a fresh environment. Record operating system, Python and library versions, hardware, random seeds, source/archive hashes and all command exit statuses. First run the existing focused tests and a small smoke case. Then reproduce the declared primary analyses; retain full logs and machine-readable outputs. The supplied notebooks' cleared outputs must not be presented as execution records.

Acceptance means deterministic IDs/counts and saved model/bound fields agree with the documented pipeline, and continuous metrics agree to manuscript rounding precision. Record every discrepancy before deciding whether it is numerical, environmental or substantive. Boundary optimizer acceptance may differ across environments: preserve historical acceptance and new convergence status as separate fields. Do not silently rewrite the original 57-object primary set.

Separate two reproducibility claims: (a) primary correction and selected boundary/resolution analyses reproduced from inputs; (b) the entire notebook workflow reproduced, including imaging and point-cloud sampling. Claim (b) only after it actually passes. The unresolved simulator configuration must remain a limitation if it cannot be resolved.

Before external transport, verify the physical interpretation of raw 3DMSL mesh coordinates against the generating code and archive. The model includes log-volume features and is sensitive to units. MitoEM analysis code currently constructs meshes with micrometre spacing, whereas the correction was fitted on raw 3DMSL mesh units. Do not feed micrometre-valued volumes directly into that model. If it is verified that one raw mesh coordinate unit represents 24 nm, convert new physical coordinates to those units before extracting features; otherwise stop the transport experiment until a defensible mapping exists. Choose units from provenance, never by optimizing test error.

Define the two bounding boxes identically to revision 9: B_q spans all queries; B_in spans only inside-labeled queries; V_o=p B_q. Include translation, axis-order and uniform-scale sanity cases with known expected transformations.

## 2. Controlled test of the proposed label-generation mechanism

### Design

Use the 60 previously analyzed development shapes, whose IDs are already recorded, to avoid outcome-driven selection of a favorable new subset. These are previously examined shapes, not independent confirmatory data. The new intervention and its analysis can nevertheless be fixed before its new outcomes are observed.

First use a separate small engineering pilot, excluded from the primary intervention summaries, to check that the fusion implementation builds and runs. Record the exact repository commit, dependencies, camera views, resolution, normalization, depth morphology, meshing and query conventions. If the historical generating version cannot be recovered, call this a controlled recreation, not an exact reproduction of historical generation.

For each of the 60 shapes, generate three paired conditions: depth offset 0, 0.75 and 1.5 voxels, with 100 views and 256-cubed fusion resolution. Change only the offset. Keep the same raw mesh, viewing geometry, random draws, grid convention and extraction settings. This yields 180 planned shape/condition outputs. Check that a zero-offset setting does not silently change another branch of the implementation.

### Measurements and analysis

Primary endpoint: the paired change in signed percentage volume error, comparing offset 1.5 with offset 0 for each shape, relative to the same raw-mesh volume. Report its mean, median, spread and an object-resampled paired interval. This interval describes the sampled shapes, not independent animals.

Use common query points for the paired occupancy estimates. Begin with the released 100,000-query convention. On the engineering pilot, estimate whether sampling uncertainty is small compared with the paired difference; if not, choose a larger fixed query count before the main run and use it for every condition. Preserve an additional 100,000-query comparison for compatibility with the original study. Do not adapt each object's query budget using its eventual favorable/unfavorable result.

Secondary endpoints: the paired change in fitted transition midpoint t50, transition width, mesh topology, reconstruction success and raw-reference volume error. Plot the intermediate 0.75 condition to assess dose response, without making it a separate success criterion. Apply one fit-acceptance rule across conditions and report all failed/missing outputs and their IDs. Volume comparisons can remain evaluable even when a transition fit fails; keep their denominators distinct.

### Interpretation

A positive paired difference supports a causal effect of the offset in this recreated pipeline. Smaller reference error at zero offset supports offset removal as one contributor to improved agreement. A residual zero-offset bias implies other processing effects remain. If the interval spans zero or effects vary by shape, report that result. Do not tune the offset to match the original 0.00304 midpoint, and do not treat approximate numerical agreement as proof of the historical generating code. No result from this experiment establishes preserved mitochondrial function.

### Compute planning

Benchmark time, peak memory and accelerator requirements on the engineering pilot before allocating the full run. Estimate runtime as roughly 180 times the per-condition runtime, adjusted for safe concurrency, plus boundary-distance fitting. Legacy graphics/CUDA dependencies may dominate setup. No measured cost or completion time is claimed in advance.

## 3. New-source frozen correction evaluation

### Choose the claim and acquisition before running

A practical candidate is the previously unused human-cortex portion of MitoEM. The dataset contains human and rat cortex volumes, as documented by the [original authors](https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf). Verify available labeled regions, data terms, completeness and lack of prior use before committing IDs. One human volume would provide a new acquisition domain, not multi-donor replication.

Preferably add another independently acquired, documented volume from a different source. Do not declare additional slices or spatial tiles to be independent animals. An unused 3DMSL archive is another within-resource test; EMPIAR liver data may overlap the library and requires source-lineage verification before being called external.

Generate the required paired raw mesh and occupancy inputs with the pinned pipeline. Applying the same generation pipeline to new acquisitions tests acquisition/shape transport. Testing a genuinely different label-generation pipeline is a separate, harder question and should be a prespecified secondary stress test. Neither is automatically supplied by the existing MitoEM resolution experiment.

### Freeze what must remain unchanged

Hash and archive the reconstructed development model, its means/scales, coefficient ordering, clipping, global offset and 2.661348551355358% bound before evaluating the new source. Clearly label it as the deterministic reconstruction, since the historical model file is unavailable. Save all eligible test IDs and exclusion rules before calculating prediction errors. Do not adjust features, clipping, low-occupancy threshold or calibration based on external outcomes.

Use all eligible objects if feasible; otherwise specify a deterministic random sample and seed before outcomes. Start with a target of approximately 500 objects per available acquisition as a feasibility target, not a justified biological sample size. Audit source counts and failure reasons. Keep the low-occupancy subgroup in the main analysis rather than excluding it after failures occur.

### Endpoints and uncertainty

Report raw, global-corrected and frozen occupancy-only mean signed error, median absolute percentage error, root mean square error, upper-tail error, saved-bound coverage and processing failures, separately for each acquisition. Compare methods on the same objects, using paired resampling. Report failed processing as a separate denominator so success-only accuracy cannot hide exclusions.

For the prespecified main contrast, report the paired difference in absolute error between the frozen occupancy-only and global correction, with its interval. Report coverage and its interval at the unchanged bound. If coverage is materially below the nominal 95%, that is failed uncertainty transport even when volume accuracy improves. An interval containing 95% is not proof of a guarantee; nominal coverage remains dependent on assumptions. Do not call an arbitrary tolerance clinically or biologically acceptable.

Repeat coverage summaries for occupancy below 0.02 and at least 0.02, with counts and intervals. Approximately 500 independent observations would give a normal-approximation half-width near 1.9 percentage points at 95% coverage, but spatial/object dependence makes that optimistic. Near 50% coverage, about 385 independent subgroup objects would be needed for a half-width near five points. These calculations illustrate precision, not a promise that the required low-occupancy count or independence exists.

With only one or two source volumes, report source-specific results and explicitly limited generalization. With enough independent acquisitions, use acquisition-aware resampling and report between-source variation. A cluster bootstrap with very few clusters must not be used to imply reliable population-level precision. A separately enriched low-occupancy stress sample is useful, but must not be pooled into prevalence-weighted main coverage without the correct sampling weights.

### Decision rule

Improved paired error plus acceptable, transparently reported coverage on new acquisitions strengthens the scope of the claim. Improved error with poor coverage supports correction transport but not uncertainty transport. Poor accuracy supports a narrower dataset-specific conclusion. Recalibration or model revision after this result is new development and requires another untouched test set for confirmation.

## 4. Small model-sensitivity study

Compare the retained model with a global offset, an occupancy-fraction-only model, and one regularized version of the same feature set. Select regularization only within development training folds; keep calibration separate. Use grouped folds if biological/acquisition lineage is available. Keep preprocessing and target definitions identical, report paired performance and coefficients, and avoid a broad model search.

The existing development and second-batch outcomes have already been inspected. Analyses on them are exploratory even if a fresh random split is drawn. Freeze any resulting candidate before the new external evaluation. Preserve the original model and results as the primary historical analysis rather than overwriting them with whichever alternative looks best.

## Protocol outputs to lock before execution

Create protocol_version.md, source_and_object_manifest.csv, environment_lock, model_hashes.json, parameter_grid.json, eligibility_rules.md and analysis_specification.md. Include primary/secondary endpoints, missing-data handling, random seeds, planned sample counts or all-eligible rules, stopping rules for technical failures, and explicit unit conversions. Report every deviation with its reason. Do not alter the manuscript's completed-results tables until new outputs exist and have been audited.
