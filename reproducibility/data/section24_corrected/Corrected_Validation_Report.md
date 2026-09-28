# Corrected Section 24 validation results

The supplied output folder contains all 360 expected condition records (60 objects × 3 offsets × 2 conventions), without duplicate condition keys. The pilot records exactly match the corresponding full-run condition records. The 60-object paired volume results were independently recalculated from the CSVs and reproduce the saved summary. This verifies the summaries, not a rerun of rendering, fusion, or model fitting.

## Pilot-excluded intervention

Five pilot IDs were removed: 24558, 24611, 24853, 24898, 24925. Exclusion is a retrospective correction after the full-run results were inspected. It must not be presented as a newly untouched validation set. Both conventions are reported because convention selection in the original workflow used full-run agreement.

| Result | Centred convention | Literal convention |
|---|---:|---:|
| Non-pilot objects | 55 | 55 |
| Mean paired volume-error increase, offset 1.5 minus 0 (percentage points) | 1.571 | 1.593 |
| 95% object-bootstrap interval | 1.374–1.773 | 1.393–1.806 |
| Objects increasing at both offset steps | 55/55 | 55/55 |
| Accepted boundary-fit pairs, offsets 0 and 1.5 | 53/55 | 52/55 |

For the centred convention, median signed volume errors were +1.578%, +2.649%, and +3.435% at offsets 0, 0.75, and 1.5. These are medians within conditions; their subtraction is not the mean paired effect. The paired fused-mesh volume-error increase was 1.477 percentage points (95% interval 1.297–1.658).

Uncertainty was recalculated with the notebook's 5,000 paired object bootstrap replicates and seed 0. It describes variation among these objects and does not establish between-cell or between-specimen uncertainty.

## Boundary fits: confirmed failures

Among the 55 non-pilot objects, centred fits passed the saved acceptance flag in 54/55 objects at each offset. Failures occur in different objects, leaving 53 accepted endpoint pairs. The accepted-pair mean midpoint change is 0.001264 in normalized coordinates (95% interval 0.001219–0.001311); all 53 changes are positive.

Literal fits passed in 54/55, 55/55, and 53/55 objects at offsets 0, 0.75, and 1.5, respectively. There are 52 accepted endpoint pairs. The accepted-pair mean midpoint change is 0.001289 (95% interval 0.001243–0.001336); all 52 changes are positive.

The original summaries included unsuccessful fits. The corrected summaries use the recorded acceptance flags. The CSVs do not contain optimizer diagnostics or failure reasons, so the causes cannot be reconstructed from these outputs. Released-label fits also lack acceptance flags; their boundary comparisons remain unverified. These limitations do not require excluding valid occupancy-volume measurements for the same objects.

## Model results and remaining work

The supplied model JSON confirms that the numerical reproduction comparison covered 2,728 second-batch rows. The retained-model maximum residual difference is 4.46e-14 percentage points. Its reported second-batch median absolute percentage error remains 0.664%, versus 3.481% without correction. These additional files verify the recorded comparison count; they do not provide per-object model predictions for a new independent check of the model table.

Revision 10 follow-up: ridge was refitted locally with scaling learned separately within each cross-validation training fold. The selected penalty remains 0.398107 and displayed metrics are unchanged. See ../section24_model_corrected for current results. The output folder includes a package lock record, improving environment documentation; missing raw-input/development-audit hashes remain a provenance limitation.

The mechanistic experiment remains a controlled reimplementation with interpolation-based labels, not proof of the exact historical label-generation process. Independent-source validation and the clean full-pipeline rerun remain outstanding.

## Suggested manuscript wording

After retrospectively excluding the five engineering-pilot objects, the controlled reimplementation was summarized for 55 objects. Under the centred coordinate convention, increasing the depth offset from 0 to 1.5 voxels increased signed occupancy-volume error by a mean of 1.571 percentage points (95% object-bootstrap interval, 1.374–1.773); every object increased at both offset steps. The literal convention gave a comparable mean increase of 1.593 percentage points (1.393–1.806). These findings support a depth-offset contribution to volume inflation within the recreated pipeline, while leaving its equivalence to the historical implementation unresolved.

## Deliverables

- corrected_summary.json: full-precision results for non-pilot 55 and all-60 sensitivity analyses, both conventions.
- nonpilot55_per_condition.csv: all 330 non-pilot condition records.
- failed_boundary_fits.csv: failed fits from the full run, including pilots.
- input_sha256.json: hashes of all supplied output files (computed at this audit, not before the experiment).
- script_hash_check.json: notebook-embedded script comparison against the recorded run hashes.
- recalculate.py: reproducible calculation script.

Original input files and the manuscript were left unchanged.
