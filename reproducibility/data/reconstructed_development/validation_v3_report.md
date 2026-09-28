# Internal Validation Following Exploratory Development

The earlier correction percentages were computed on the same instances used to fit the correction. This analysis uses a stratified 60/20/20 train/calibration/test split. Model parameters are fitted on training rows; earlier full-shard exploration may have informed model design.

| Split | n |
|---|---:|
| train | 1630 |
| calibration | 540 |
| test | 550 |

## Held-out test performance

| Method | mean bias (95% CI) | median absolute error (95% CI) | RMSE | 95% limits | calibration q95 | test coverage |
|---|---:|---:|---:|---:|---:|---:|
| uncorrected | +3.608 [+3.410, +3.799] | 3.035 [2.838, 3.241] | 4.286 | -0.927 to +8.144 | 8.451 | 96.4% |
| global_train_mean | -0.040 [-0.217, +0.156] | 1.370 [1.240, 1.478] | 2.231 | -4.416 to +4.336 | 4.632 | 96.4% |
| occupancy_only_ols | -0.032 [-0.124, +0.062] | 0.580 [0.502, 0.635] | 1.149 | -2.286 to +2.222 | 2.661 | 96.2% |

## Sensitivity to candidate error margins

| Method | <=2.17% | <=5% | <=6% | <=10% |
|---|---:|---:|---:|---:|
| uncorrected | 29.1% | 79.5% | 87.8% | 97.8% |
| global_train_mean | 78.0% | 97.1% | 97.8% | 99.5% |
| occupancy_only_ols | 94.4% | 99.5% | 100.0% | 100.0% |

## Interpretation guardrails

- The calibrated q95 is an **in-domain measurement-error bound**, not a biological or functional threshold.
- The 2.17% value is a Monte-Carlo repeatability coefficient estimated from stored query points; it is shown only as a sensitivity threshold.
- The occupancy-only correction may be deployed without mesh measurements, but it still requires validation on another 3DMSL shard.
- The proposed boundary-dilation mechanism remains a hypothesis until verified against the 3DMSL generation code.
- MitoEM can test descriptor portability, but it cannot validate this occupancy correction because it lacks paired 3DMSL occupancy labels.
