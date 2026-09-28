# Revision 14

- Archived the supplied original Section 25 output folder unchanged.
- Verified all 40 unique condition records and all 20 centred hybrid records against the expected first ten non-pilot IDs.
- Reaggregated every containment summary; values exactly match saved JSON and notebook outputs. Section 24 interpolation rows also match exactly.
- Checked the supplied model-comparison table against the independently rerun MSE comparison; differences are numerical rounding only.
- Replaced the obsolete missing-per-object statement in the manuscript and updated the result-to-file map, README and priority status.
- Numerical results and scientific interpretation remain unchanged; no expensive experiment was rerun.

# Revision 13

- Archived the newly supplied original frozen run without modifying its files.
- Verified all five recorded input hashes and replayed the original model on all 2,728 rows without refitting.
- Replaced obsolete missing-model statements with verified original-artifact provenance; kept the separate Section 25 per-object gap explicit.
- Added E8/E9/E10 to the experiment map and updated dataset participation to avoid counting reused objects as new cohorts.
- Scoped test-set chronology to the original fixed-model evaluation and labeled later analyses exploratory.
- Added RESULT_FILE_MAP.md and PRIORITY_STATUS.md.

# Revision 12

- Precisely described the ten-object TSDF-band-restricted hybrid containment check, separating its field threshold from the sampled near-surface metric.
- Removed unsupported prospective-selection and full-containment implications.
- Retained output-supported numerical results, reporting mean effects 1.566 versus 1.683 percentage points and offset-specific maximum differences.
- Explicitly disclosed MSE versus R-squared ridge model selection; retained alpha 0.2512 and verified metrics.
- Removed causal attribution of residual displacement to untested fusion steps.
- Distinguished recorded original-model hashes, reconstructed coefficients and available model bytes.
- Archived the supplied executed Section 25 notebook and its extracted scripts/output summaries.
- Retained bibliography and figure assets unchanged; external reference links were not rechecked.

# Revision 10

- Added corrected 55-object depth-offset results and both coordinate conventions, with an in-text referenced table.
- Reported retrospective pilot exclusion, data-dependent convention selection, interpolation-versus-mesh labeling limits, and historical-pipeline uncertainty.
- Applied paired fit acceptance: 53 centred and 52 literal accepted boundary pairs; valid volume results retain all 55 objects.
- Added exploratory model comparison table and corrected ridge CV using fold-specific scaling; reran locally with unchanged displayed results.
- Updated abstract, contributions, methods, results, limitations, conclusion and data/code availability.
- Preserved original executed notebook and supplied outputs, added portable recalculation scripts and an unexecuted rerun notebook.
- No new bibliography entries or changes to existing references; prior link audit was not repeated.

# Revision 9: submission-audit manuscript fixes

See audit/REVISION_9_CHANGES.md. Numerical evidence is unchanged; proposed validation is separate from completed results.

# Revision 8: reference and claim audit

Verified reference identities, corrected Faitg study type, reduced citation grouping, removed inaccessible unpublished self-citation while retaining the proposed framework, added the mesh-fusion source citation, supplied missing/public links, corrected chapter metadata, and checked all figure/table callouts. See audit/AUDIT_v8.md for access exceptions and evidence. Numerical outputs and analysis code are unchanged.

# Revision 7: changes following the v6 audit

- Preserved v6's original title, shorter abstract, and single-column performance table.
- Corrected the abstract to 57 accepted fits from 60 sampled objects; qualified the causal claim throughout and identified the half-shift estimate as a heuristic.
- Restored the correct Bland–Altman percentage-error formula and denominator, exact calibration score and order statistic, interval inversion, small-sample caveat, and pairwise interpretation of repeatability.
- Restored subgroup counts, Wilson intervals, the 124/216 failures outside the flag, and the composition/within-group decomposition. Removed claims of a proven applicability domain.
- Restored grid-phase, tie-rule, topology, scale-aware geometry, query-box sensitivity, exclusion flow, seed/bootstrap settings, and reconstruction provenance.
- Added the exploratory influence check: Pearson correlation changes from 0.5765 to 0.3120 when the largest-offset object is omitted; that object remains in primary estimates.
- Regenerated the boundary figure from packaged replay results, with historically rejected fits marked separately, and added the phase-sensitivity figure to Appendix B.
- Replaced the small outlier-suppressed boxplot with the verified full-distribution ECDF and retained the complete agreement plot.
- Corrected imaging-calibration interpretation and ethics wording, and replaced the stale package README.

## Verification and limits

The primary correction results and existing sensitivity tables are retained from the documented v5 reproduction. This turn regenerated figures and the influence diagnostic, compiled the manuscript, and visually checked its pages. It did not repeat all raw-data analyses or run the notebooks end to end. The included corrected notebooks remain the audited v5 versions.

No original frozen-model byte comparison, controlled generation ablation, public repository deposit, or full reference-by-reference audit is claimed. These limitations remain explicit in the manuscript and package.
