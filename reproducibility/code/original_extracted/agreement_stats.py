#!/usr/bin/env python3
"""
agreement_stats.py - Weeks 7-8: representation-agreement statistics on the paired table.

  ICC(2,1)  two-way random effects, absolute agreement, single measurement
            (Shrout & Fleiss 1979; McGraw & Wong 1996 ICC(A,1)), computed from the
            ANOVA variance components directly.
  Spearman  rank correlation between each pair of representations (rank-transform
            + Pearson, so no scipy dependency).
  Bootstrap 95% CI for each ICC, resampling INSTANCES (not measurements), because
            representations are paired within an instance.
  Equivalence-margin check: sphericity-conditioned margin taken from the Week 1-2
            measured error (15% when occ_fraction_inside < 2%, else 6%).
            *** Starting point for pre-registration - needs adviser sign-off. ***

  python3 agreement_stats.py --in paired_feature_table.csv --outdir .
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Descriptor families compared across representations. Add rows to extend the analysis.
PAIRS = [
    {"name": "extent_x",
     "label": "extent_x (mesh vs. point-cloud[trimmed] vs. occupancy)",
     "cols": ["mesh_extent_x", "pc_trimmed_extent_x", "occ_extent_x"],
     "methods": ["mesh", "pointcloud_trimmed", "occupancy"]},
    {"name": "volume",
     "label": "volume (mesh vs. occupancy-derived estimate)",
     "cols": ["mesh_volume", "occ_volume_est"],
     "methods": ["mesh", "occupancy_est"]},
]

MARGIN_LOW_OCC = 15.0    # percent, occ_fraction_inside < 2%
MARGIN_OTHER = 6.0       # percent, everything else
LOW_OCC_CUT = 0.02


def icc21(M):
    """ICC(2,1): M is (n subjects x k methods), no missing values."""
    M = np.asarray(M, dtype=np.float64)
    n, k = M.shape
    if n < 2 or k < 2:
        return np.nan
    grand = M.mean()
    row_m = M.mean(axis=1)
    col_m = M.mean(axis=0)
    SSR = k * ((row_m - grand) ** 2).sum()
    SSC = n * ((col_m - grand) ** 2).sum()
    SSE = ((M - row_m[:, None] - col_m[None, :] + grand) ** 2).sum()
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    denom = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return np.nan if denom == 0 else float((MSR - MSE) / denom)


def spearman(x, y):
    x = pd.Series(x).rank().to_numpy()
    y = pd.Series(y).rank().to_numpy()
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_icc(M, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = icc21(M[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def drop_zero_variance(df):
    dead = [c for c in df.columns
            if df[c].dtype.kind in "fi" and df[c].notna().any() and df[c].nunique(dropna=True) <= 1]
    if dead:
        print("Dropping zero-variance columns: %s" % ", ".join(dead))
    return df.drop(columns=dead), dead


def fig_icc(results, path):
    rows = [r for r in results if np.isfinite(r["icc"])]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 3.2 + 0.4 * len(rows)))
    y = np.arange(len(rows))
    icc = [r["icc"] for r in rows]
    lo = [max(r["icc"] - r["ci"][0], 0) for r in rows]
    hi = [max(r["ci"][1] - r["icc"], 0) for r in rows]
    ax.barh(y, icc, xerr=[lo, hi], color="#4C72B0", ecolor="#333333", capsize=4, height=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(["%s\n(n=%d)" % (r["name"], r["n"]) for r in rows])
    ax.set_xlim(0, 1.05)
    ax.axvline(0.90, ls="--", c="#888888", lw=1)
    ax.text(0.90, -0.7, "0.90 'excellent'", fontsize=8, color="#666666", ha="center")
    for yi, r in zip(y, rows):
        ax.text(min(r["icc"], 1.0) + 0.01, yi, "%.3f" % r["icc"], va="center", fontsize=9)
    ax.set_xlabel("ICC(2,1), absolute agreement (bars = 95% bootstrap CI)")
    ax.set_title("Cross-representation agreement")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bland_altman(df, path):
    panels = []
    if {"mesh_extent_x", "pc_trimmed_extent_x"} <= set(df.columns):
        panels.append(("extent_x: point-cloud[trimmed] vs. mesh", df["mesh_extent_x"], df["pc_trimmed_extent_x"]))
    if {"mesh_volume", "occ_volume_est"} <= set(df.columns):
        panels.append(("volume: occupancy estimate vs. mesh", df["mesh_volume"], df["occ_volume_est"]))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5))
    axes = np.atleast_1d(axes)
    for ax, (title, a, b) in zip(axes, panels):
        m = (a + b) / 2.0
        d = b - a
        ok = np.isfinite(m) & np.isfinite(d)
        m, d = m[ok], d[ok]
        ax.scatter(m, d, s=6, alpha=0.25, color="#4C72B0", edgecolors="none")
        mu, sd = d.mean(), d.std(ddof=1)
        ax.axhline(mu, color="#C44E52", lw=1.4, label="bias = %.3g" % mu)
        ax.axhline(mu + 1.96 * sd, color="#C44E52", ls="--", lw=1, label="+/-1.96 SD")
        ax.axhline(mu - 1.96 * sd, color="#C44E52", ls="--", lw=1)
        ax.set_xlabel("mean of the two methods")
        ax.set_ylabel("difference (second - first)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Bland-Altman style agreement plots", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="paired_feature_table.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.inp)
    df, dead = drop_zero_variance(df)

    results = []
    for spec in PAIRS:
        cols = [c for c in spec["cols"] if c in df.columns]
        if len(cols) < 2:
            print("skipping %s (missing columns)" % spec["name"])
            continue
        sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        M = sub.to_numpy()
        icc = icc21(M)
        ci = bootstrap_icc(M, n_boot=args.n_boot, seed=args.seed)
        sp = {}
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                sp["%s vs %s" % (spec["methods"][a], spec["methods"][b])] = spearman(M[:, a], M[:, b])
        results.append(dict(name=spec["name"], label=spec["label"], n=len(sub),
                            icc=icc, ci=ci, spearman=sp, cols=cols))

    # equivalence-margin check (occupancy vs mesh volume)
    eq = None
    if {"mesh_volume", "occ_volume_est", "occ_fraction_inside"} <= set(df.columns):
        d = df[["mesh_volume", "occ_volume_est", "occ_fraction_inside"]].dropna()
        err = (100.0 * (d["occ_volume_est"] - d["mesh_volume"]) / d["mesh_volume"]).abs()
        low = d["occ_fraction_inside"] < LOW_OCC_CUT
        margin = np.where(low, MARGIN_LOW_OCC, MARGIN_OTHER)
        within = err.to_numpy() <= margin
        eq = dict(n=len(d), overall=100.0 * within.mean(),
                  n_low=int(low.sum()),
                  low=100.0 * within[low.to_numpy()].mean() if low.any() else np.nan,
                  n_high=int((~low).sum()),
                  high=100.0 * within[(~low).to_numpy()].mean() if (~low).any() else np.nan)

    icc_png = os.path.join(args.outdir, "icc_summary.png")
    ba_png = os.path.join(args.outdir, "agreement_scatter.png")
    fig_icc(results, icc_png)
    fig_bland_altman(df, ba_png)

    lines = ["# Representation-Agreement Statistics (Weeks 7-8)", "",
             "Input: `%s` (n=%d rows)" % (os.path.basename(args.inp), len(df)), ""]
    if dead:
        lines += ["Zero-variance columns dropped before analysis: %s"
                  % ", ".join("`%s`" % c for c in dead), ""]
    lines += ["## ICC(2,1) and rank agreement", "",
              "| Pair | n | ICC(2,1) | 95% bootstrap CI |", "|---|---|---|---|"]
    for r in results:
        lines.append("| %s | %d | %.3f | [%.3f, %.3f] |"
                     % (r["label"], r["n"], r["icc"], r["ci"][0], r["ci"][1]))
    lines += ["", "Spearman rank correlations:", ""]
    for r in results:
        for k, v in r["spearman"].items():
            lines.append("- %s - %s: **%.4f**" % (r["name"], k, v))
    if eq:
        lines += ["", "## Equivalence-margin check (occupancy vs. mesh volume)", "",
                  "Margins: **%.0f%%** when `occ_fraction_inside` < %g, **%.0f%%** otherwise."
                  % (MARGIN_LOW_OCC, LOW_OCC_CUT, MARGIN_OTHER),
                  "These come from the Week 1-2 measured error and are a *starting point for",
                  "pre-registration*, not a finalized value - they need adviser/mentor sign-off.", "",
                  "- Overall: **%.1f%%** of %d instances within margin" % (eq["overall"], eq["n"]),
                  "- Low `occ_fraction_inside` (<%g, n=%d, %.0f%% margin): **%.1f%%** within"
                  % (LOW_OCC_CUT, eq["n_low"], MARGIN_LOW_OCC, eq["low"]),
                  "- Higher `occ_fraction_inside` (>=%g, n=%d, %.0f%% margin): **%.1f%%** within"
                  % (LOW_OCC_CUT, eq["n_high"], MARGIN_OTHER, eq["high"])]
    lines += ["", "## Figures", "", "- `icc_summary.png`", "- `agreement_scatter.png`", ""]
    rep = os.path.join(args.outdir, "agreement_report.md")
    with open(rep, "w") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines[:40]))
    print("\nWrote %s, %s, %s" % (rep, icc_png, ba_png))


if __name__ == "__main__":
    main()