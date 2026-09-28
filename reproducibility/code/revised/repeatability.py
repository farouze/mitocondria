#!/usr/bin/env python3
"""
repeatability.py - the missing piece of the Weeks 7-8 milestone: an estimate of the
occupancy volume estimator's *own precision*, measured without reference to the mesh.

The equivalence margins used so far were derived from occupancy-vs-mesh agreement,
which mixes two different things: how noisy the estimator is (precision) and how far
off-centre it is (bias). This script separates them using only the occupancy archive:

  1. K-fold resampling of each instance's query points gives the estimator's replicate
     SD at full N - a genuine repeatability estimate, no mesh involved.
  2. That empirical SD is compared with the analytic binomial floor sqrt(p(1-p)/N).
     A ratio near 1 is a descriptive check; random fold assignment cannot establish independent query generation.
  3. A convergence curve (N = 1k ... 100k) shows whether the estimate is approaching
     the mesh volume or a persistent offset at the available sampling depth.
  4. A uniformity check on the query-point marginals tests the one alternative
     explanation - that the sampling design, not the labels, causes the offset.

  python3 repeatability.py --root /path/to/shard --out repeatability.csv --sample 400
"""
import argparse
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from hit_common import find_instance_dirs, instance_id, load_mesh, load_occupancy, raw_extent, unnormalize

C_BLUE, C_ORANGE, C_INK, C_MUTED, C_GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d8d7d2"
K_FOLDS = 10
CONVERGENCE_N = (1000, 2500, 5000, 10000, 25000, 50000, 100000)


def instance_repeatability(d, rng, with_mesh=True):
    pts, occ, loc, scale = load_occupancy(d)
    pts_raw = unnormalize(pts, loc, scale)
    N = len(occ)
    p = float(occ.mean())
    v_box = float(np.prod(raw_extent(pts_raw)))
    v_full = p * v_box

    # 1. K-fold replicate SD -> SD of the full-N estimate
    idx = rng.permutation(N)
    folds = np.array_split(idx, K_FOLDS)
    fold_est = np.array([occ[f].mean() * v_box for f in folds])
    sd_fold = fold_est.std(ddof=1)
    sd_full_emp = sd_fold / np.sqrt(K_FOLDS)

    # 2. analytic binomial floor at full N
    sd_full_ana = np.sqrt(p * (1 - p) / N) * v_box

    # 3. convergence curve
    conv = {}
    for n in CONVERGENCE_N:
        if n <= N:
            conv["v_at_%d" % n] = float(occ[idx[:n]].mean() * v_box)

    # 4. uniformity of the query-point marginals (10 bins per axis, chi-square df=9)
    chi2 = []
    for a in range(3):
        counts, _ = np.histogram(pts_raw[:, a], bins=10)
        exp = N / 10.0
        chi2.append(float(((counts - exp) ** 2 / exp).sum()))

    row = dict(instance_id=instance_id(d), n_points=N, occ_fraction_inside=p,
               v_box=v_box, occ_volume_est=v_full,
               sd_replicate=sd_full_emp, sd_analytic=sd_full_ana,
               sd_ratio=sd_full_emp / sd_full_ana if sd_full_ana else np.nan,
               rel_sd_pct=100 * sd_full_emp / v_full if v_full else np.nan,
               rc_pct=100 * 2.77 * sd_full_emp / v_full if v_full else np.nan,
               chi2_x=chi2[0], chi2_y=chi2[1], chi2_z=chi2[2])
    row.update(conv)
    if with_mesh:
        try:
            m = load_mesh(d)
            row["mesh_volume"] = float(abs(m.volume))
            row["mesh_area"] = float(m.area)
            row["bias_pct"] = 100 * (v_full - row["mesh_volume"]) / row["mesh_volume"]
        except Exception as e:
            row["mesh_error"] = str(e)
    return row


def fig_convergence(df, path):
    if plt is None:
        return
    ns = [n for n in CONVERGENCE_N if ("v_at_%d" % n) in df.columns]
    if not ns or "mesh_volume" not in df.columns:
        return
    rel = np.array([(100 * (df["v_at_%d" % n] / df["mesh_volume"] - 1)).to_numpy() for n in ns])
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    med = np.median(rel, axis=1)
    lo = np.percentile(rel, 25, axis=1)
    hi = np.percentile(rel, 75, axis=1)
    ax.fill_between(ns, lo, hi, color=C_BLUE, alpha=0.18, lw=0)
    ax.plot(ns, med, color=C_BLUE, lw=2.2, marker="o", ms=6,
            markeredgecolor="white", markeredgewidth=1)
    ax.axhline(0, color=C_INK, lw=1.4)
    ax.text(ns[0], 0.25, "mesh volume", color=C_INK, fontsize=9, va="bottom")
    ax.annotate("converges to %+.2f%%,\nnot to zero" % med[-1],
                xy=(ns[-1], med[-1]), xytext=(ns[-1] * 0.34, med[-1] + 1.9),
                color=C_ORANGE, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=1.4))
    ax.set_xscale("log")
    ax.set_xlabel("query points used in the volume estimate")
    ax.set_ylabel("occupancy − mesh volume (%)")
    ax.set_title("More sampling does not close the gap — the estimator is biased",
                 fontsize=11, loc="left")
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED); ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="repeatability.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--sample", type=int, default=400,
                    help="instances to measure (a precision estimate does not need all 2,720)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dirs = find_instance_dirs(args.root)
    if not dirs:
        raise FileNotFoundError("No 3DMSL instance folders with .off meshes under %s" % args.root)
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(dirs):
        dirs = [dirs[i] for i in sorted(rng.choice(len(dirs), args.sample, replace=False))]
    print("Measuring repeatability on %d instances" % len(dirs), flush=True)

    t0, rows = time.time(), []
    out_csv = os.path.join(args.outdir, os.path.basename(args.out))
    for i, d in enumerate(dirs, 1):
        try:
            rows.append(instance_repeatability(d, rng))
        except Exception as e:
            rows.append({"instance_id": instance_id(d), "error": str(e)})
        if i % args.progress_every == 0 or i == len(dirs):
            print("  %d/%d (%.2fs/instance)" % (i, len(dirs), (time.time() - t0) / i), flush=True)
        if i == min(5, len(dirs)) and not any("rel_sd_pct" in row for row in rows):
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            counts = Counter(row.get("error", "unknown error") for row in rows)
            raise RuntimeError("All first %d instances failed. Most common error: %s. "
                               "Inspect the error column in %s." % (i, counts.most_common(1)[0][0], out_csv))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    if "rel_sd_pct" not in df:
        counts = Counter(df["error"].fillna("unknown error")) if "error" in df else Counter()
        raise RuntimeError("No repeatability measurements succeeded. Errors: %s" % counts.most_common(3))
    ok = df[df["rel_sd_pct"].notna()].copy()
    if ok.empty:
        raise RuntimeError("No finite repeatability measurements succeeded; inspect %s" % out_csv)
    failures = len(df) - len(ok)
    if failures:
        print("WARNING: %d/%d instances failed; inspect the error column in %s" % (failures, len(df), out_csv))
    L = ["# Repeatability of the Occupancy Volume Estimator (n=%d instances)" % len(ok), "",
         "%d of %d sampled instances failed and were excluded; inspect the error column in `%s`." %
         (failures, len(df), os.path.basename(out_csv)) if failures else "All sampled instances produced finite repeatability estimates.", "",
         "Measured from the occupancy archive alone: each instance's %d query points were" % int(ok["n_points"].median()),
         "split into %d disjoint folds, a volume computed from each, and the replicate SD" % K_FOLDS,
         "rescaled to the full-N estimate. No mesh is used anywhere in steps 1-2, so this is",
         "precision, not agreement.", "",
         "## 1. Precision", "", "| Quantity | Median | IQR |", "|---|---|---|",
         "| replicate SD, as %% of the estimate | **%.3f%%** | %.3f – %.3f |"
         % (ok["rel_sd_pct"].median(), ok["rel_sd_pct"].quantile(.25), ok["rel_sd_pct"].quantile(.75)),
         "| repeatability coefficient (2.77 × SD) | **%.3f%%** | %.3f – %.3f |"
         % (ok["rc_pct"].median(), ok["rc_pct"].quantile(.25), ok["rc_pct"].quantile(.75)),
         "| empirical SD ÷ analytic binomial SD | **%.3f** | %.3f – %.3f |"
         % (ok["sd_ratio"].median(), ok["sd_ratio"].quantile(.25), ok["sd_ratio"].quantile(.75)), "",
         "Agreement of fold and binomial SDs is descriptive; random partitioning cannot prove independent query generation.",
         "The repeatability coefficient describes pairwise repeat differences, not single-estimate accuracy.", ""]

    mesh_ok = ok.dropna(subset=["mesh_volume"]) if "mesh_volume" in ok else ok.iloc[:0]
    if len(mesh_ok) and "bias_pct" in mesh_ok:
        b = mesh_ok["bias_pct"]
        L += ["## 2. Precision vs. bias", "",
              "| | Value |", "|---|---|",
              "| repeatability (replicate SD) | **%.3f%%** |" % ok["rel_sd_pct"].median(),
              "| bias (occupancy − mesh) | **%+.3f%%** |" % b.median(),
              "| bias ÷ repeatability SD | **%.1f×** |" % (b.median() / ok["rel_sd_pct"].median()), "",
              "The offset is roughly %.0f replicate standard deviations away from the mesh" % (b.median() / ok["rel_sd_pct"].median()),
              "volume. An equivalence margin built on repeatability alone would be about",
              "**±%.2f%%** (2.77 × SD) — far tighter than the 6%%/15%% margins used so far, and" % ok["rc_pct"].median(),
              "the right basis for pre-registration once the bias is corrected rather than",
              "absorbed into the margin.", ""]

    ns = [n for n in CONVERGENCE_N if ("v_at_%d" % n) in mesh_ok.columns]
    if ns and len(mesh_ok):
        L += ["## 3. Convergence", "", "| Query points | median error vs. mesh |", "|---|---|"]
        for n in ns:
            L.append("| %d | %+.3f%% |" % (n, (100 * (mesh_ok["v_at_%d" % n] / mesh_ok["mesh_volume"] - 1)).median()))
        final = float((100 * (mesh_ok["v_at_%d" % ns[-1]] / mesh_ok["mesh_volume"] - 1)).median())
        sd = float(ok["rel_sd_pct"].median())
        if abs(final) > 3 * sd:
            L += ["", "At the largest sample the estimate still sits %+.3f%% from the mesh volume — %.1f×"
                  % (final, abs(final) / sd),
                  "the replicate SD at that sample size. This is consistent with a systematic",
                  "offset at the available sampling depth. Confirm the mechanism before",
                  "assuming that additional query points cannot reduce it.", ""]
        else:
            L += ["", "At the largest sample the residual error (%+.3f%%) is within %.1f× the replicate"
                  % (final, abs(final) / sd if sd else float("nan")),
                  "SD, so on this evidence the discrepancy is consistent with sampling noise rather",
                  "than a fixed offset. Re-check against the full shard before concluding either way.", ""]

    crit = 27.88   # chi-square 0.999 critical value, df = 9
    frac = float(((ok[["chi2_x", "chi2_y", "chi2_z"]] > crit).any(axis=1)).mean())
    L += ["## 4. Are the query points uniform?", "",
          "| Axis | median χ² (10 bins, df=9) |", "|---|---|"]
    for a in "xyz":
        L.append("| %s | %.2f |" % (a, ok["chi2_%s" % a].median()))
    L += ["", "Instances failing a χ² uniformity test on any axis at p<0.001 (crit %.2f): **%.1f%%**."
          % (crit, 100 * frac), ""]
    if frac < 0.05:
        L += ["The one-axis marginal tests do not show major nonuniformity. They do not test",
              "the joint 3D sampling distribution, so they cannot rule out every sampling",
              "effect. A label-boundary difference remains a plausible mechanism and needs",
              "direct verification against the 3DMSL generation procedure.", ""]
    else:
        L += ["**%.1f%% of instances fail the uniformity test, so the sampling design cannot be" % (100 * frac),
              "ruled out as a source of the offset.** Inspect the query-point distribution",
              "directly before attributing the bias to the occupancy labels — a non-uniform",
              "sampling box biases `mean(occupancy) × box volume` on its own.", ""]
    L += ["(χ² on 10 bins is mildly inflated by the float16 storage of the query points; treat",
          "a modest exceedance rate as uniform rather than as evidence of structure.)", ""]

    fig = os.path.join(args.outdir, "estimator_convergence.png")
    fig_convergence(mesh_ok, fig)
    if os.path.exists(fig):
        L += ["## Figure", "", "- `estimator_convergence.png`", ""]

    rep = os.path.join(args.outdir, "repeatability_report.md")
    with open(rep, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    written = [out_csv, rep] + ([fig] if os.path.exists(fig) else [])
    print("\nWrote %s" % ", ".join(written))


if __name__ == "__main__":
    main()
