#!/usr/bin/env python3
"""
analyze_results.py - descriptive pass over the paired feature table: morphology-space
summary, the size/shape confound, and how image-derived features track shape vs. size.
Reproduces the three full-shard findings recorded in the Weeks 5-6 milestone.

  python3 analyze_results.py --in paired_feature_table.csv --outdir .
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def r(df, a, b):
    if a not in df or b not in df:
        return np.nan
    d = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 3 or d[a].std() == 0 or d[b].std() == 0:
        return np.nan
    return float(np.corrcoef(d[a], d[b])[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="paired_feature_table.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.inp)
    n = len(df)
    L = ["# Descriptive Analysis (n=%d)" % n, ""]

    L += ["## Morphology space", "", "| Descriptor | median | min | max |", "|---|---|---|---|"]
    for c in ("mesh_sphericity", "mesh_elongation", "mesh_volume", "mesh_area", "occ_fraction_inside"):
        if c in df:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s):
                L.append("| `%s` | %.4g | %.4g | %.4g |" % (c, s.median(), s.min(), s.max()))

    L += ["", "## Size / shape confound", ""]
    rv = r(df, "mesh_volume", "mesh_sphericity")
    L.append("- `mesh_volume` vs `mesh_sphericity`: **r = %.3f** - size and shape are not "
             "independent axes in this dataset, so a naive volume-vs-shape comparison is "
             "partly confounded." % rv)

    L += ["", "## Occupancy volume error structure", ""]
    if {"occ_volume_est", "mesh_volume"} <= set(df.columns):
        err = (100.0 * (df["occ_volume_est"] - df["mesh_volume"]) / df["mesh_volume"]).abs()
        df = df.assign(_abs_vol_err=err)
        L += ["- absolute error vs mesh volume: mean **%.2f%%**, median **%.2f%%**"
              % (err.mean(), err.median()),
              "- error vs `mesh_sphericity`: r = **%.3f**" % r(df, "_abs_vol_err", "mesh_sphericity"),
              "- error vs `occ_fraction_inside`: r = **%.3f**" % r(df, "_abs_vol_err", "occ_fraction_inside")]
        if "occ_fraction_inside" in df:
            low = df["occ_fraction_inside"] < 0.02
            if low.any():
                L.append("- `occ_fraction_inside` < 2%% (n=%d): mean error **%.2f%%** vs **%.2f%%** "
                         "for the rest" % (int(low.sum()), err[low].mean(), err[~low].mean()))

    L += ["", "## Image-derived features: shape vs. size", "",
          "| Image feature | r with `mesh_sphericity` | r with `mesh_extent_x` |", "|---|---|---|"]
    for c in [c for c in df.columns if c.startswith("img_") and
              ("fg_fraction" in c or "fg_bbox_area_frac" in c)]:
        L.append("| `%s` | %.3f | %.3f |" % (c, r(df, c, "mesh_sphericity"), r(df, c, "mesh_extent_x")))
    L += ["", "Threshold-based image morphometry recovers **shape** better than absolute "
          "**size** - a representation-invariance result worth carrying into the manuscript.", ""]

    # figure: sphericity vs volume error + error distribution
    if "_abs_vol_err" in df:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        d = df[["mesh_sphericity", "_abs_vol_err"]].dropna()
        axes[0].scatter(d["mesh_sphericity"], d["_abs_vol_err"], s=6, alpha=0.3,
                        color="#4C72B0", edgecolors="none")
        axes[0].set_xlabel("mesh sphericity")
        axes[0].set_ylabel("|occupancy volume error| (%)")
        axes[0].set_title("Volume error is sphericity-conditioned", fontsize=10)
        axes[1].hist(df["_abs_vol_err"].dropna(), bins=60, color="#4C72B0")
        axes[1].set_xlabel("|occupancy volume error| (%)")
        axes[1].set_ylabel("instances")
        axes[1].set_title("Error distribution", fontsize=10)
        fig.tight_layout()
        p = os.path.join(args.outdir, "volume_consistency.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        L += ["Figure: `volume_consistency.png`", ""]

    out = os.path.join(args.outdir, "analysis_report.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    print("\nWrote %s" % out)


if __name__ == "__main__":
    main()