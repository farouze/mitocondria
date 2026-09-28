#!/usr/bin/env python3
"""
qc_filter.py - Weeks 3-4 milestone: apply the documented two-tier QC policy to the
audit table and emit a QC'd descriptor table plus a human-readable QC report.

  HARD EXCLUDE  mesh, point cloud or occupancy failed to load -> instance unusable
  FLAG (kept)   not watertight
                occ_fraction_inside < 2%            (Monte-Carlo volume unreliable)
                point-cloud raw extent > 10% off the mesh  (sampling-noise outlier)
                fewer than 6 renders in any image configuration

Edit the THRESHOLDS dict below to change the policy; the report restates whatever
values were actually used, so it never goes stale.

  python3 qc_filter.py --in audit_full_descriptors.csv --out audit_descriptors_qc.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from hit_common import IMG_CONFIGS, EXPECTED_VIEWS

THRESHOLDS = {
    "min_occ_fraction_inside": 0.02,   # below this, occupancy volume error ~11% not ~3.5%
    "max_pc_extent_pct_diff": 10.0,    # raw point-cloud extent vs mesh extent
    "expected_views_per_config": EXPECTED_VIEWS,
}


def apply_qc(df, th=THRESHOLDS):
    df = df.copy()
    for c in ("mesh_ok", "pc_ok", "occ_ok"):
        if c not in df:
            df[c] = False
        df[c] = df[c].fillna(False).astype(bool)

    df["qc_exclude"] = ~(df["mesh_ok"] & df["pc_ok"] & df["occ_ok"])
    df["qc_exclude_reason"] = np.where(
        df["qc_exclude"],
        df.apply(lambda r: ",".join(
            [k for k, v in (("mesh_load_failed", r["mesh_ok"]),
                            ("pointcloud_load_failed", r["pc_ok"]),
                            ("occupancy_load_failed", r["occ_ok"])) if not v]), axis=1),
        "")

    flags = [[] for _ in range(len(df))]

    def add(mask, name):
        for i in np.flatnonzero(np.asarray(mask.fillna(False))):
            flags[i].append(name)

    if "mesh_watertight" in df:
        add(~df["mesh_watertight"].fillna(False).astype(bool) & df["mesh_ok"], "not_watertight")
    if "occ_fraction_inside" in df:
        add(df["occ_fraction_inside"] < th["min_occ_fraction_inside"], "low_occupancy_sampling")
    if "pc_extent_pct_diff_max" in df:
        add(df["pc_extent_pct_diff_max"].abs() > th["max_pc_extent_pct_diff"], "pc_extent_outlier")
    img_cols = [c for c in ("n_images_" + c for c in IMG_CONFIGS) if c in df]
    if img_cols:
        add((df[img_cols] < th["expected_views_per_config"]).any(axis=1), "incomplete_images")

    df["qc_flags"] = [";".join(f) for f in flags]
    df["qc_n_flags"] = [len(f) for f in flags]
    df["qc_pass"] = ~df["qc_exclude"]
    return df


def write_report(df, path, th=THRESHOLDS, src=""):
    n = len(df)
    kept = int(df["qc_pass"].sum())
    lines = ["# QC Report (Weeks 3-4)", ""]
    if src:
        lines += ["Input: `%s`" % src, ""]
    lines += ["Instances audited: **%d**" % n,
              "", "## Policy applied", "",
              "| Tier | Rule | Effect |", "|---|---|---|",
              "| HARD EXCLUDE | mesh / point cloud / occupancy failed to load | row dropped from analysis |",
              "| FLAG | mesh not watertight | kept, caveat |",
              "| FLAG | `occ_fraction_inside` < %g | kept, caveat |" % th["min_occ_fraction_inside"],
              "| FLAG | raw point-cloud extent > %g%% from mesh extent | kept, use trimmed extent |" % th["max_pc_extent_pct_diff"],
              "| FLAG | fewer than %d renders in a configuration | kept, image features partial |" % th["expected_views_per_config"],
              "", "## Outcome", "",
              "- Hard exclusions: **%d**" % (n - kept),
              "- Retained for analysis: **%d** (%.1f%%)" % (kept, 100.0 * kept / max(n, 1)), ""]
    if n - kept:
        lines.append("### Excluded instances")
        lines.append("")
        for _, r in df[df["qc_exclude"]].iterrows():
            lines.append("- `%s` - %s" % (r["instance_id"], r["qc_exclude_reason"]))
        lines.append("")

    lines += ["### Flag counts", "", "| Flag | n | % of retained |", "|---|---|---|"]
    kept_df = df[df["qc_pass"]]
    allflags = sorted({f for s in kept_df["qc_flags"] for f in s.split(";") if f})
    for f in allflags:
        c = int(kept_df["qc_flags"].str.split(";").apply(lambda L: f in L).sum())
        lines.append("| `%s` | %d | %.2f%% |" % (f, c, 100.0 * c / max(kept, 1)))
    if not allflags:
        lines.append("| (none) | 0 | 0.00%% |")
    lines += ["", "Flagged instances are **kept**. Flags travel with the row into",
              "`paired_feature_table.csv` so downstream analysis can condition on them",
              "(e.g. the sphericity-conditioned equivalence margin uses",
              "`low_occupancy_sampling`).", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="QC filter for the 3DMSL audit table")
    ap.add_argument("--in", dest="inp", default="audit_full_descriptors.csv")
    ap.add_argument("--out", default="audit_descriptors_qc.csv")
    ap.add_argument("--report", default="qc_report.md")
    ap.add_argument("--keep-excluded", action="store_true",
                    help="write excluded rows too (marked qc_pass=False)")
    args = ap.parse_args()

    df = apply_qc(pd.read_csv(args.inp))
    write_report(df, args.report, src=args.inp)
    out = df if args.keep_excluded else df[df["qc_pass"]].copy()
    out.to_csv(args.out, index=False)

    print("QC: %d audited, %d hard-excluded, %d retained"
          % (len(df), int(df["qc_exclude"].sum()), int(df["qc_pass"].sum())))
    kept = df[df["qc_pass"]]
    for f in sorted({x for s in kept["qc_flags"] for x in s.split(";") if x}):
        c = int(kept["qc_flags"].str.split(";").apply(lambda L: f in L).sum())
        print("  flag %-24s %d" % (f, c))
    print("Wrote %s and %s" % (os.path.abspath(args.out), os.path.abspath(args.report)))


if __name__ == "__main__":
    main()