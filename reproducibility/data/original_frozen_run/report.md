# Frozen-model validation on a second 3DMSL shard

Audited: 2728; evaluated: 2728; exclusions: 0.
No coefficients, feature scaling, or calibration bounds were fitted on this shard.

| Method | Mean bias % | Median absolute error % | RMSE % | Original bound % | Coverage |
|---|---:|---:|---:|---:|---:|
| uncorrected | 4.497 | 3.481 | 5.743 | 8.451 | 90.0% |
| global_train_mean | 0.817 | 1.503 | 3.542 | 4.632 | 90.0% |
| occupancy_only_ols | 0.006 | 0.664 | 1.743 | 2.661 | 92.1% |

Confidence intervals and exploratory candidate-margin results are in summary.csv.
Bootstrap intervals resample objects and assume object independence; they do not quantify between-animal uncertainty.
Same-dataset replication does not establish generalization to independent acquisitions. Source-object provenance and geometry duplication remain unverified.
Nominal calibration coverage is not guaranteed after distribution shift. Candidate-margin pass rates are descriptive, not formal equivalence tests.
Nonwatertight meshes remain flagged and retained, following the original policy; a watertight-only sensitivity summary is provided.