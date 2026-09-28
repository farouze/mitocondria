#!/usr/bin/env python3
"""
shard_shift.py - why is the raw occupancy bias larger on the second shard, and where does
the frozen correction lose coverage?

Inputs (all existing outputs; nothing is refitted on the second shard):
  --dev-paired   development paired_feature_table.csv (shard 10_24553_27272)
  --dev-pred     development validation_v3_predictions.csv (held-out test errors)
  --new-audit    second-shard audit.csv from the frozen-validation run
  --new-pred     second-shard frozen_predictions.csv
  --model        validation_v3_model.json (for the frozen calibration bound)

Reports:
  1. covariate shift between shards (standardized mean differences)
  2. whether the bias mechanism replicates (same surface-to-volume slope, same offset
     in the normalized frame)
  3. how much of the larger raw bias the shift in surface-to-volume predicts, using the
     development relationship only
  4. where the frozen correction's errors and coverage loss sit (subgroups, deciles)
  5. the pc_extent_outlier flag under alternative definitions, in both shards

  python3 shard_shift.py --dev-paired ... --dev-pred ... --new-audit ... --new-pred ... --model ... --outdir out
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def derive(df):
    df = df.copy()
    df["instance_id"] = df["instance_id"].astype(str)
    df["err_pct"] = 100 * (df.occ_volume_est / df.mesh_volume - 1)
    df["sv"] = df.mesh_area / df.mesh_volume ** (2 / 3)
    df["t_norm"] = (df.occ_volume_est - df.mesh_volume) / df.mesh_area / df.occ_scale
    df["log_volume"] = np.log(df.mesh_volume)
    df["log_scale"] = np.log(df.occ_scale)
    df["watertight"] = df.mesh_watertight.astype(str).str.lower().eq("true")
    df["low_occ"] = df.occ_fraction_inside < 0.02
    return df


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((b.mean() - a.mean()) / sp) if sp else np.nan


def fit(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ b
    return b, float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def outlier_defs(df):
    out = {}
    if "pc_extent_pct_diff_max" in df:
        out["largest single-axis deviation > 10% (current qc_filter.py)"] = int((df.pc_extent_pct_diff_max.abs() > 10).sum())
    if {"pc_extent_x", "mesh_extent_x"} <= set(df.columns):
        x = 100 * (df.pc_extent_x / df.mesh_extent_x - 1)
        out["x axis only > 10%"] = int((x.abs() > 10).sum())
    cols = [a for a in "xyz" if {"pc_extent_%s" % a, "mesh_extent_%s" % a} <= set(df.columns)]
    if cols:
        m = pd.concat([(100 * (df["pc_extent_%s" % a] / df["mesh_extent_%s" % a] - 1)).abs() for a in cols], axis=1)
        out["mean over axes > 10%"] = int((m.mean(axis=1) > 10).sum())
        out["max over axes > 15%"] = int((m.max(axis=1) > 15).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("dev-paired", "dev-pred", "new-audit", "new-pred", "model", "outdir"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = derive(pd.read_csv(a.dev_paired, dtype={"instance_id": str}))
    new = derive(pd.read_csv(a.new_audit, dtype={"instance_id": str}))
    newp = pd.read_csv(a.new_pred, dtype={"instance_id": str})
    devp = pd.read_csv(a.dev_pred, dtype={"instance_id": str})
    model = json.load(open(a.model))
    bound = float(model["calibration_bounds_pct"]["occupancy_only_ols"])
    new = new.merge(newp[["instance_id", "occupancy_only_ols_error_pct"]], on="instance_id", how="inner")
    new["corr_err"] = new.occupancy_only_ols_error_pct
    dev_test = dev.merge(devp[devp.split == "test"][["instance_id", "corrected_error_occupancy_only_ols"]],
                         on="instance_id", how="inner").rename(columns={"corrected_error_occupancy_only_ols": "corr_err"})

    L = ["# Second-Shard Shift and Coverage Diagnostics", "",
         "Development shard n=%d (held-out test n=%d); second shard n=%d. Nothing is refitted on the"
         % (len(dev), len(dev_test), len(new)),
         "second shard: all comparisons use development fits or the frozen model.", ""]

    # 1. covariate shift
    L += ["## 1. How the shards differ", "",
          "| Variable | development median | second shard median | SMD |", "|---|---|---|---|"]
    rows = []
    for v, lbl in (("log_volume", "log mesh volume"), ("mesh_sphericity", "sphericity"),
                   ("mesh_elongation", "elongation"), ("sv", "surface-to-volume A/V^(2/3)"),
                   ("occ_fraction_inside", "occ_fraction_inside"), ("log_scale", "log scale"),
                   ("err_pct", "raw volume error %")):
        s = smd(dev[v], new[v]); rows.append((lbl, s))
        L.append("| %s | %.4g | %.4g | %+.2f |" % (lbl, dev[v].median(), new[v].median(), s))
    L += ["| non-watertight meshes | %d (%.1f%%) | %d (%.1f%%) | |" % ((~dev.watertight).sum(), 100 * (~dev.watertight).mean(),
                                                                     (~new.watertight).sum(), 100 * (~new.watertight).mean()),
          "| occ_fraction_inside < 2%% | %d (%.1f%%) | %d (%.1f%%) | |" % (dev.low_occ.sum(), 100 * dev.low_occ.mean(),
                                                                         new.low_occ.sum(), 100 * new.low_occ.mean()), "",
          "SMD = standardized mean difference (second − development); |SMD| > 0.2 is a noticeable shift.", ""]

    # 2. mechanism replication
    bd, r2d = fit(dev.err_pct.to_numpy(), dev.sv.to_numpy())
    bn, r2n = fit(new.err_pct.to_numpy(), new.sv.to_numpy())
    cv = lambda x: float(x.std() / abs(x.mean()))
    wt = new[new.watertight]
    L += ["## 2. Does the bias mechanism replicate?", "",
          "| | development | second shard | second shard, watertight only |", "|---|---|---|---|",
          "| mean raw error | %+.3f%% | %+.3f%% | %+.3f%% |" % (dev.err_pct.mean(), new.err_pct.mean(), wt.err_pct.mean()),
          "| share of instances above zero | %.2f%% | %.2f%% | %.2f%% |" % (100 * (dev.err_pct > 0).mean(), 100 * (new.err_pct > 0).mean(), 100 * (wt.err_pct > 0).mean()),
          "| slope of error on A/V^(2/3) | %.3f | %.3f | |" % (bd[1], bn[1]),
          "| R² of error on A/V^(2/3) | %.2f | %.2f | |" % (r2d, r2n),
          "| median offset t / scale | %.5f | %.5f | %.5f |" % (dev.t_norm.median(), new.t_norm.median(), wt.t_norm.median()),
          "| CV of t / scale | %.2f | %.2f | %.2f |" % (cv(dev.t_norm), cv(new.t_norm), cv(wt.t_norm)), ""]
    same_t = abs(new.t_norm.median() / dev.t_norm.median() - 1) < 0.10
    L += ["The normalized-frame offset is %s across shards (%.5f vs %.5f)%s" % (
        "essentially the same" if same_t else "different", dev.t_norm.median(), new.t_norm.median(),
        ". The mechanism is a fixed property of how the labels were generated, not of one shard." if same_t
        else ". Check the non-watertight meshes and generation provenance before pooling shards."), ""]

    # 3. how much of the shift does surface-to-volume predict?
    pred_dev = bd[0] + bd[1] * dev.sv.mean()
    pred_new = bd[0] + bd[1] * new.sv.mean()
    obs_shift = new.err_pct.mean() - dev.err_pct.mean()
    pred_shift = pred_new - pred_dev
    L += ["## 3. Is the larger raw bias explained by shape?", "",
          "Using the development fit of error on surface-to-volume only:", "",
          "- observed change in mean raw error: **%+.3f** percentage points" % obs_shift,
          "- change predicted by the shift in A/V^(2/3): **%+.3f** points (%.0f%% of the observed change)"
          % (pred_shift, 100 * pred_shift / obs_shift if obs_shift else np.nan), ""]
    if abs(obs_shift) < 0.1:
        L += ["The mean raw error barely differs between shards, so there is no shift to explain.", ""]
    else:
      L += [("Most of the larger raw bias comes from the second shard having more surface per unit volume, "
           "which is what the boundary-offset mechanism predicts. That is also why the occupancy-only "
           "model, whose features track shape, still comes out unbiased while the constant correction does not.")
          if obs_shift and pred_shift / obs_shift >= 0.6 else
          ("Shape explains only part of the larger raw bias; the rest comes from something that differs "
             "between shards (check non-watertight meshes and the source of the objects)."), ""]

    # 4. where the frozen model's errors sit
    L += ["## 4. Where the frozen correction loses coverage (bound %.3f%%)" % bound, "",
          "| Group | n | median \\|error\\| | coverage | development test coverage |", "|---|---|---|---|---|"]

    def cov(e):
        return 100 * np.mean(np.abs(e) <= bound)
    groups = [("all", new, dev_test), ("watertight", new[new.watertight], dev_test[dev_test.watertight]),
              ("non-watertight", new[~new.watertight], dev_test[~dev_test.watertight]),
              ("occ_fraction < 2%", new[new.low_occ], dev_test[dev_test.low_occ]),
              ("occ_fraction ≥ 2%", new[~new.low_occ], dev_test[~dev_test.low_occ])]
    for lbl, g, gd in groups:
        if len(g):
            L.append("| %s | %d | %.3f%% | %.1f%% | %s |" % (lbl, len(g), g.corr_err.abs().median(), cov(g.corr_err),
                                                       "%.1f%% (n=%d)" % (cov(gd.corr_err), len(gd)) if len(gd) else "—"))
    new["sv_decile"] = pd.qcut(new.sv, 10, labels=False, duplicates="drop") + 1
    L += ["", "| A/V^(2/3) decile (second shard) | n | mean corrected error | coverage |", "|---|---|---|---|"]
    for q, g in new.groupby("sv_decile"):
        L.append("| %d | %d | %+.3f%% | %.1f%% |" % (q, len(g), g.corr_err.mean(), cov(g.corr_err)))
    worst = new.reindex(new.corr_err.abs().sort_values(ascending=False).index).head(int(max(1, 0.05 * len(new))))
    L += ["", "The worst 5%% of corrected errors (n=%d): median sphericity %.3f vs %.3f overall; median"
          % (len(worst), worst.mesh_sphericity.median(), new.mesh_sphericity.median()),
          "occ_fraction_inside %.3f vs %.3f; non-watertight %.1f%% vs %.1f%%."
          % (worst.occ_fraction_inside.median(), new.occ_fraction_inside.median(),
             100 * (~worst.watertight).mean(), 100 * (~new.watertight).mean()), ""]
    new[["instance_id", "err_pct", "corr_err", "sv", "t_norm", "mesh_sphericity", "occ_fraction_inside",
         "watertight", "sv_decile"]].to_csv(os.path.join(a.outdir, "shard_shift_instances.csv"), index=False)

    # 5. pc_extent_outlier definitions
    od, on = outlier_defs(dev), outlier_defs(new)
    L += ["## 5. `pc_extent_outlier` under alternative definitions", "",
          "| Definition | development | second shard |", "|---|---|---|"]
    for k in od:
        L.append("| %s | %d | %s |" % (k, od[k], on.get(k, "—")))
    L += ["", "Recommendation: keep the current definition (largest single-axis deviation of the raw",
          "point-cloud extent from the mesh extent > 10%). It flags the most instances, which suits an",
          "advisory flag, and it is what the released code does; analyses use the trimmed extent either",
          "way. The earlier count of 1 came from a prior implementation; report the current definition",
          "and its count.", ""]

    with open(os.path.join(a.outdir, "shard_shift_report.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()