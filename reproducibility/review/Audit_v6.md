# Audit of Mitochondrial Morphometry v6

**Assessment: major revision before submission.** The central correction results remain supported by the earlier local reproduction, but several previously identified reporting errors and overinterpretations remain in this ZIP. Its shorter abstract, compact Table IV, and boundary figure with rejected fits are useful editorial improvements. They do not resolve the scientific issues below.

## Scope and evidence

Reviewed all nine pages of the supplied `manuscript_preview.pdf`, `main.tex`, bibliography, README, figure assets, and figure-generation script. Compared them against the previously supplied notebooks and the verified local v5 reproduction tables. Original files were not modified. This audit did not rerun the entire analysis or independently recheck every cited paper. The bibliography is unchanged from the audited v5 source; all citation keys resolve, and the preview has no unresolved-reference markers.

The v6 file name should not be interpreted as evidence that this package contains the audited v5 changes: many are absent. References to pages below refer to the included nine-page preview; source line numbers refer to this ZIP's `main.tex`.

## Major findings

### 1. Causal language exceeds the boundary evidence

**Locations:** abstract and contributions, p. 1; Methods IV-F, p. 4; Results V-B, p. 5; conclusion, p. 7. Source lines 21, 38, 180, 208–210, 335.

The abstract calls 0.00326 an expected watertighting offset, and the contributions/conclusion attribute the observed discrepancy to that step. The Results sentence acknowledging that the mechanism is supported rather than proved is more defensible, but contradicts those stronger statements.

The upstream fusion code implements a 1.5-voxel depth shift and also a morphological erosion of the depth map. Its output is not determined solely by a simple average projection of the shift. The extra factor of one half is a heuristic unless derived for the actual visibility, weighting, morphology, and fusion operations. No verified offset-on/offset-off generation experiment is available.

**Action:** use “consistent with inflation during label generation” throughout. Label 0.00326 as a half-shift heuristic, and explicitly state that the exact generating version and causal ablation remain unresolved. The observed outward boundary transition itself is a supported result.

Primary implementation: https://github.com/davidstutz/mesh-fusion/blob/master/2_fusion.py

### 2. Coverage loss is not localized exclusively to the low-occupancy group

**Locations:** contributions, p. 1; Table V and Results V-D, p. 6; Discussion VI-B, p. 7. Source lines 38, 281, 321.

The flag identifies much higher individual failure risk, but the reproduced counts are:

| Second-batch group | Covered / total | Failures | Coverage |
|---|---:|---:|---:|
| Occupancy < 0.02 | 89 / 181 | 92 | 49.17% |
| Occupancy ≥ 0.02 | 2,423 / 2,547 | 124 | 95.13% |
| All | 2,512 / 2,728 | 216 | 92.08% |

Thus 124/216 = **57.4% of all failures occur outside the flag**. Unflagged coverage also falls from 522/534 = 97.75% internally to 95.13% in the second batch. Reweighting internal subgroup coverages to second-batch proportions gives 94.17%, leaving a further 2.09 percentage-point decline attributable descriptively to within-group changes.

A flag defined earlier does not by itself establish that its use as a model applicability restriction was prespecified. Observed 95.1% coverage is not a demonstrated conditional guarantee. The object-level Wilson interval is approximately 94.2–95.9%; it does not address biological dependence.

**Action:** report “observed coverage near 95% in the unflagged subgroup”; remove claims that the loss is localized or that a validated applicability domain was fixed in advance unless supported by dated records. Restore counts and uncertainty intervals.

### 3. Bland–Altman percentage-error formula does not match the reported result

**Locations:** Section III, p. 2; Results V-A, p. 5. Source lines 89 and 202.

The displayed definition uses SD of differences divided by mean mesh volume, with no explicit factor of 100. The reported 16.5% instead reproduces using the mean of both methods:

- `100 × 1.96 × SD(difference) / mean(pair mean)` = **16.4683%**.
- The same expression using mean mesh volume = **16.8532%**.

**Action:** define the denominator as the mean pair volume and include the percentage conversion. Retain 16.5% after rounding. Also soften “regardless of object-level error” in the ICC explanation: high between-object variation can mask important disagreement, but does not guarantee ICC near one regardless of error magnitude.

### 4. Conformal score is underspecified and the generic cap is wrong

**Location:** Methods IV-E, p. 4, source line 168.

“Absolute calibration residual” needs to identify the implemented corrected-volume score:

`a_i = abs(100 × (V_corrected,i / V_mesh,i − 1))`.

The signed score before taking its absolute value equals `(e_i − predicted_e_i)/(1 + predicted_e_i/100)`, not simply the regression residual. With 540 calibration objects, the 95% bound uses ordered score 514. Capping the requested index at n is not the general split-conformal rule: when the index exceeds n, the conservative convention is an infinite bound. This edge case does **not** change the current 540-object results.

**Action:** provide the exact score, index, interval inversion, and correct small-sample rule. Keep the limitation that feature exploration preceded internal splitting; exchangeability alone does not cure data-adaptive model selection using calibration information.

Primary reference, Appendix D: https://arxiv.org/html/2107.07511v6

### 5. Resolution conclusions omit available sensitivity evidence and overextend across datasets

**Locations:** Methods IV-G, pp. 4–5; Results V-E, pp. 6–7. Source lines 190, 302.

The statement about comparing 3DMSL to finer-resolution EM “after accounting for shifts of this size” suggests portable correction magnitudes. These shifts came from one annotation-processing procedure on rat cortex, with z fixed at 30 nm; they are not validated cross-instrument or cross-tissue corrections.

The earlier all-550-object sensitivity analysis found median 16 nm volume shifts ranging **+0.58% to +6.94%** across phases with foreground ties, and **−5.47% to +0.89%** with background ties. At the original phase, 26-neighbor component counts changed in one object at each spacing and Euler characteristics changed in nine. These findings qualify interpretations based only on watertight meshes and the single original phase.

**Action:** restore the phase/tie/topology results, at least in a supplement. Distinguish matching in-plane spacing from matching full voxel geometry. Limit the general conclusion to processing sensitivity.

### 6. Boundary sample size and correlation need clearer reporting

**Locations:** abstract, p. 1; Results V-B, p. 5; Figure 1, p. 6. Source lines 21, 208, 215.

The abstract's “In 60 objects” attaches summary values calculated on 57 accepted fits. Say “Among 57 accepted fits from 60 sampled objects.” The new figure usefully shows the three rejected fits explicitly.

The new narrative correctly notes an influential high-offset object. Using the archived 57-object acceptance set, I additionally checked removal of object **25660**, the largest volume-implied offset:

- Pearson r: **0.5765 → 0.3120**.
- Spearman rho: **0.3506 → 0.3152**.

This is an influence diagnostic, not a reason to exclude the object from the primary analysis. Small dispersion alone does not establish why Pearson and Spearman differ.

**Action:** report this sensitivity alongside the original correlation, or soften the explanation. Keep all valid accepted observations in the main estimate. Supply the exact generator and binned data for the new pooled Figure 1a; this ZIP's `generate_figures.py` produces only the resolution figure, so the pooled curve cannot be independently reconstructed from the package alone.

## Additional corrections

### 7. Repeatability coefficient is not a single-estimate error margin

**Locations:** IV-D, p. 4; IV-H and V-A, p. 5. Source lines 156, 194, 204.

The README's open question about dividing fold SD by sqrt(10) is resolved: that operation appears in the supplied `repeatability.py`. But 2.77 × SD describes pairwise differences between independent repeats. With median SD 0.784%, the illustrative single-estimate normal 95% half-width is 1.96 × 0.784% = **1.54%**, not 2.17%. Both rely on the query-sampling assumptions. Label 2.17% as a descriptive pairwise threshold, not attainable single-estimate accuracy.

### 8. Geometric explanation omits normalization scale

**Locations:** Section III, p. 2; V-D, p. 6. Source lines 96, 260.

For constant normalized displacement, the first-order prediction is `e ≈ 100 × t_norm × A*s/V`. Shape complexity `A/V^(2/3)` omits the factor `s/V^(1/3)`. Its empirical association can remain, but it does not by itself establish a mechanistic decomposition of the batch shift. The earlier development-fitted A*s/V diagnostic already exists and gave second-batch R² = 0.828. Restore it or narrow the explanation to a descriptive association.

### 9. Supplementary pixel-size conclusion is not identified by the ratios

**Location:** Appendix A, p. 8, source line 353.

The stated 57 nm follows from `109 × sqrt(0.77/2.82)`, matching the confocal ratio. Unit image-to-geometry agreement instead gives `109/sqrt(2.82) = 64.9 nm`. Neither is an independent calibration. Calling 57 nm implausible does not exclude calibration/configuration errors or establish blur and segmentation as the unique cause. State the normalization explicitly and retain the unresolved simulator-configuration provenance.

### 10. Restore reproducibility details already available

- Internal split seed: **2026**. The notebook explicitly requests **2,000** internal bootstrap resamples; v6 says 1,000 without distinguishing internal from frozen evaluation. The frozen bootstrap used 1,000, seed 790.
- The occupancy estimator uses observed coordinate spans, not automatically the nominal 1.1*s box. The earlier sensitivity check found a median **0.1066%** volume difference.
- Sequential MitoEM exclusions: 1,203 → remove 586 slab-boundary objects → remove 61 further image-boundary objects → none further for size/z span → remove six for ROI size → **550**. State the ordering, since criteria overlap.
- The original frozen-model file was not available for byte comparison with the deterministic reconstruction. Distinguish those artifacts and do not imply that reconstruction independently establishes original analysis chronology.
- The claim that the calibrated bound was the “only margin fixed” before second-batch opening needs evidence; the code also contains fixed candidate margins. Remove the exclusivity claim if chronology is not documented.
- The README still calls itself revision 2, contains revision-5 notes and unresolved author checks, and requests a missing frozen-error PDF. Update it to describe the actual delivered package.

### 11. Presentation and wording

All nine preview pages were inspected. No major clipping or broken references were evident. Figure 2's labels are very small at column width, and suppression of outliers obscures the errors relevant to undercoverage; use the verified full-distribution ECDF or another plot showing tails. In Figure 3a the −0.08% annotation nearly coincides with the zero line; offset it for readability.

Replace the ethics sentence “No human participants or animals were involved” with “No new animal experiments or human-participant research were conducted,” since the source datasets originate from animal tissue.

## Supported results to retain

The frozen second-batch occupancy-only **MdAPE 0.664%, RMSE 1.743%, and coverage 2,512/2,728 = 92.1%** match the earlier reproduction. Identical raw/global covered-object sets were verified and the arithmetic explanation is retained correctly in v6. The outward label transition is supported. The study appropriately distinguishes computational reference geometry from biological ground truth and acknowledges that these datasets do not test functional Homeostatic Invariance thresholds.

The most efficient revision is to retain v6's useful editorial improvements while restoring the verified statistical definitions, subgroup counts, sensitivity findings, and provenance qualifications from the audited package. A fresh large-scale rerun is not needed to resolve most items above.
