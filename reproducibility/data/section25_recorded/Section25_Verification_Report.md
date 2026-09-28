# Section 25 saved-output verification

The latest HIT_section25_audit_followups.ipynb now contains saved run outputs (82,996 bytes; SHA-256 recorded in verification.json). Execution counts remain null, but outputs are present; null counts alone do not make this an output-free notebook. Its embedded analysis scripts match the previously reviewed Section 25 notebook; two command cells differ only in trailing whitespace.

## Resolved evidence gap

The saved outputs support revision 11's reported ten-object numerical summaries:

| Metric | Offset 0 | Offset 1.5 |
|---|---:|---:|
| Median near-surface label agreement, hybrid containment versus interpolation | 99.4418% | 99.7260% |
| Median volume-error change, hybrid minus interpolation (percentage points) | +0.129696 | +0.019329 |
| Maximum absolute volume-error change (percentage points) | 0.264152 | 0.099881 |

The recorded mean paired offset effect is 1.566482 percentage points with the hybrid labels versus 1.682910 with interpolation. The latter exactly matches a separate calculation from the previously supplied Section 24 per-object CSVs for the first ten non-pilot IDs. This is useful consistency evidence. The new containment-specific values have been checked against the saved summary, not independently recomputed from containment per-object outputs, which are not embedded in full.

The ridge outputs report alpha 0.2512, MdAPE 0.665%, RMSE 1.747%, and coverage 91.972%. These match the independent local MSE-based rerun performed during the revision 11 audit. The claim that the new numbers lacked any saved run evidence is now resolved.

## Remaining methodological and wording corrections

1. **Call the containment calculation a TSDF-band-restricted or hybrid check.** The code evaluates generalized winding numbers only for abs(interpolated TSDF) < 2/256, and retains interpolated labels elsewhere. The threshold is a fused-field value, not a verified Euclidean-distance band. The new execution outputs do not establish the asserted geometric bound outside this band. Avoid describing it as full mesh containment of every query point.
2. **Disclose the ridge scoring change.** MSE rather than mean fold R-squared selects alpha 0.251 rather than 0.398. Both results used corrected fold-specific scaling in our local comparisons; scaling alone should not be credited for the change.
3. **Do not partition the remaining displacement among fusion operations.** Only depth offset was intervened on. The residual cannot be attributed specifically to other fusion steps from these results.
4. **Original frozen model availability remains separately unverified.** This notebook does not supply the original model JSON or independently verify its hash. Saved prediction agreement and a previously recorded hash do not provide the original artifact bytes.
5. **Selection chronology remains limited.** The code uses the first ten non-pilot IDs, consistent with the logs. It does not independently prove selection before any outcomes were inspected. Describe it as a deterministic exploratory subset unless a prospective record supports stronger wording.

## Suggested replacement result wording

In an exploratory check of the first ten non-pilot objects, winding-number containment replaced interpolated labels only within a threshold band of the fused signed-distance field; labels outside that band were retained. Median agreement between these hybrid and interpolated labels on the sampled near-surface queries was 99.4% at offset 0 and 99.7% at offset 1.5. The mean paired volume-error increase was 1.566 percentage points with hybrid labels and 1.683 with interpolated labels. Thus, the tested within-band replacement preserved the positive offset effect in this subset; it does not establish full containment equivalence or reproduce the historical label-generation process.

No manuscript or original notebook was changed. A ZIP of validation/section25_20260926T223354Z would permit independent reaggregation of the new per-object containment metrics, but is not needed merely to confirm that the quoted values appear in the saved notebook outputs.
