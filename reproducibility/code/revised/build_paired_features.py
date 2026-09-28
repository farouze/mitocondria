#!/usr/bin/env python3
"""
build_paired_features.py - Weeks 5-6 deliverable: merge the QC'd structural table with
the image-derived descriptors into one paired feature table, one row per instance with
every representation's descriptors side by side. This is the table the Weeks 7-8
agreement statistics run on.

  python3 build_paired_features.py --structural audit_descriptors_qc.csv \
                                   --images image_features.csv \
                                   --out paired_feature_table.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--structural", default="audit_descriptors_qc.csv")
    ap.add_argument("--images", default="image_features.csv")
    ap.add_argument("--out", default="paired_feature_table.csv")
    args = ap.parse_args()

    s = pd.read_csv(args.structural)
    i = pd.read_csv(args.images)
    s["instance_id"] = s["instance_id"].astype(str)
    i["instance_id"] = i["instance_id"].astype(str)

    df = s.merge(i, on="instance_id", how="left", validate="one_to_one")

    # derived cross-representation comparison columns
    if {"occ_volume_est", "mesh_volume"} <= set(df.columns):
        df["vol_pct_err_occ_vs_mesh"] = 100.0 * (df["occ_volume_est"] - df["mesh_volume"]) / df["mesh_volume"]
    for ax in "xyz":
        a, b = "pc_trimmed_extent_%s" % ax, "mesh_extent_%s" % ax
        if {a, b} <= set(df.columns):
            df["extent_%s_pct_err_pc_vs_mesh" % ax] = 100.0 * (df[a] - df[b]) / df[b]
        a2 = "occ_extent_%s" % ax
        if {a2, b} <= set(df.columns):
            df["extent_%s_pct_err_occ_vs_mesh" % ax] = 100.0 * (df[a2] - df[b]) / df[b]

    df.to_csv(args.out, index=False)

    nimg = int(df[[c for c in df.columns if c.endswith("_n_views")]].sum(axis=1).gt(0).sum()) \
        if any(c.endswith("_n_views") for c in df.columns) else 0
    print("Paired table: %d rows x %d columns" % df.shape)
    print("  with image features: %d" % nimg)
    const = [c for c in df.columns
             if df[c].dtype.kind in "fi" and df[c].notna().any() and df[c].nunique(dropna=True) <= 1]
    if const:
        print("  zero-variance numeric columns (dropped downstream): %s" % ", ".join(const))
    print("Wrote %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()