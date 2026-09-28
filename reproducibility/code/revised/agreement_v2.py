#!/usr/bin/env python3
"""
agreement_v2.py - Weeks 7-8, second pass: separate systematic BIAS from measurement
NOISE, and report agreement statistics that do not depend on how heterogeneous the
sample happens to be.

Why this exists. The first pass reported ICC(2,1) = 0.994 for volume alongside "88.8%
of instances inside the equivalence margin". Those two statements are in tension, and
this script resolves it:

  1. The occupancy-vs-mesh volume discrepancy is almost perfectly one-signed, so it is
     bias, not noise.
  2. Its size is several times the Monte-Carlo sampling floor the estimator is entitled
     to, so sampling noise is a minor component.
  3. It scales with the shape's surface-to-volume ratio - the signature of a constant
     boundary offset, i.e. the occupancy labels describe a slightly dilated shape.
  4. Because it is bias, it is correctable, and correcting it moves the margin pass
     rate from 88.8% to ~99%.
  5. ICC is inflated here by the 68x spread of mitochondrial volumes in the shard.
     Recomputed inside narrow size strata it collapses - direct evidence that the
     "excellent agreement" reading is a property of the sample, not the method.

  python3 agreement_v2.py --in paired_feature_table.csv --outdir .
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-3 of the validated palette (all-pairs safe, light surface)
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#d8d7d2"

MARGIN_LOW_OCC, MARGIN_OTHER, LOW_OCC_CUT = 15.0, 6.0, 0.02


# ----------------------------------------------------------------------------- stats
def icc21(M):
    M = np.asarray(M, dtype=np.float64)
    n, k = M.shape
    if n < 2 or k < 2:
        return np.nan
    g = M.mean(); rm = M.mean(1); cm = M.mean(0)
    MSR = k * ((rm - g) ** 2).sum() / (n - 1)
    MSC = n * ((cm - g) ** 2).sum() / (k - 1)
    MSE = ((M - rm[:, None] - cm[None, :] + g) ** 2).sum() / ((n - 1) * (k - 1))
    d = MSR + (k - 1) * MSE + k * (MSC - MSE) / n
    return float((MSR - MSE) / d) if d else np.nan


def ols(y, *cols):
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in cols])
    b, *_ = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)
    pred = X @ b
    ss = ((np.asarray(y, float) - np.asarray(y, float).mean()) ** 2).sum()
    return b, pred, float(1 - ((y - pred) ** 2).sum() / ss)


def boot_ci(x, fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def agreement_block(a, b):
    """Difference-based agreement summary for method b against reference a."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a; m = (a + b) / 2.0
    sd = d.std(ddof=1)
    return dict(bias=d.mean(), bias_pct=100 * d.mean() / a.mean(),
                loa_lo=d.mean() - 1.96 * sd, loa_hi=d.mean() + 1.96 * sd,
                percent_error=100 * 1.96 * sd / m.mean(),
                percent_error_denominator="mean of both methods",
                median_abs_pct=float(np.median(np.abs(100 * (b / a - 1)))),
                prop_bias_r=float(np.corrcoef(m, d)[0, 1]),
                one_signed=float(max((d > 0).mean(), (d < 0).mean())))


# --------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED); ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_INK)


def fig_bias_not_noise(e, sv, fit, mc_floor, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    ax = axes[0]
    lo, hi = min(e.min(), -1.0), float(np.percentile(e, 99.5))
    ax.hist(e, bins=60, range=(lo, hi), color=C_BLUE, edgecolor="none")
    ax.set_xlim(lo - 0.5, hi + 0.5)
    n_beyond = int((e > hi).sum())
    if n_beyond:
        ax.text(hi, ax.get_ylim()[1] * 0.06, "%d instances\nbeyond →" % n_beyond,
                color=C_MUTED, fontsize=8.5, ha="right", va="bottom")
    ax.axvline(0, color=C_INK, lw=1.6)
    ax.axvline(e.mean(), color=C_ORANGE, lw=2)
    ax.annotate("mean bias\n%+.2f%%" % e.mean(),
                xy=(e.mean(), ax.get_ylim()[1] * 0.80),
                xytext=(e.mean() + 1.4, ax.get_ylim()[1] * 0.80),
                color=C_ORANGE, fontsize=9, va="center",
                arrowprops=dict(arrowstyle="-", color=C_ORANGE, lw=1.2))
    ax.annotate("zero", xy=(0, ax.get_ylim()[1] * 0.96), xytext=(0.6, ax.get_ylim()[1] * 0.96),
                color=C_INK, fontsize=9, va="center")
    ax.set_xlabel("occupancy − mesh volume (%)")
    ax.set_ylabel("instances")
    ax.set_title("%.2f%% of instances lie on one side of zero (all but %d)"
                 % (100 * (e > 0).mean(), int((e <= 0).sum())),
                 fontsize=10.5, loc="left")
    _style(ax)

    ax = axes[1]
    ax.scatter(sv, e, s=5, alpha=0.22, color=C_BLUE, edgecolors="none")
    xs = np.linspace(np.percentile(sv, 0.2), np.percentile(sv, 99.8), 50)
    ax.plot(xs, fit[0] + fit[1] * xs, color=C_ORANGE, lw=2)
    ax.text(0.03, 0.94, "R² = %.2f" % fit[2], transform=ax.transAxes,
            color=C_ORANGE, fontsize=10, va="top")
    ax.axhline(0, color=C_INK, lw=1)
    ax.axhspan(-mc_floor, mc_floor, color=C_MUTED, alpha=0.14, lw=0)
    ax.text(ax.get_xlim()[1], mc_floor, " Monte-Carlo\n noise floor (±%.2f%%)" % mc_floor,
            color=C_MUTED, fontsize=8.5, va="bottom", ha="right")
    ax.set_xlabel(r"shape complexity  $A\,/\,V^{2/3}$")
    ax.set_ylabel("occupancy − mesh volume (%)")
    ax.set_title("Bias tracks surface-to-volume — a boundary-offset signature",
                 fontsize=10.5, loc="left")
    _style(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_icc_collapse(whole_v, dec_v, whole_e, dec_e, path):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    groups = [("volume\nmesh vs. occupancy", whole_v, np.asarray(dec_v)),
              ("extent_x\n3 representations", whole_e, np.asarray(dec_e))]
    rng = np.random.default_rng(7)
    for i, (lbl, whole, dec) in enumerate(groups):
        xw, xd = i * 1.0 - 0.17, i * 1.0 + 0.17
        ax.scatter([xw], [whole], s=110, marker="D", color=C_ORANGE, zorder=4,
                   edgecolors="white", linewidths=1.2,
                   label="whole sample" if i == 0 else None)
        ax.annotate("%.3f" % whole, xy=(xw, whole), xytext=(xw - 0.055, whole),
                    ha="right", va="center", color=C_ORANGE, fontsize=10)
        ax.scatter(xd + rng.normal(0, 0.022, len(dec)), dec, s=36, color=C_BLUE,
                   alpha=0.8, zorder=3, edgecolors="white", linewidths=0.8,
                   label="one size decile" if i == 0 else None)
        med = float(np.median(dec))
        ax.plot([xd - 0.10, xd + 0.10], [med, med], color=C_BLUE, lw=2.4, zorder=5)
        ax.annotate("median %.3f" % med, xy=(xd + 0.10, med), xytext=(xd + 0.15, med),
                    ha="left", va="center", color=C_BLUE, fontsize=10)
        ax.plot([xw, xd], [whole, med], color=C_GRID, lw=1.2, ls=":", zorder=1)
    ax.axhline(0.90, ls="--", lw=1, color=C_MUTED, zorder=0)
    ax.text(-0.52, 0.902, "0.90  “excellent”", color=C_MUTED, fontsize=8.5,
            va="bottom", ha="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=10)
    ax.set_xlim(-0.55, 1.62)
    ax.set_ylim(0.70, 1.035)
    ax.set_ylabel("ICC(2,1)")
    ax.set_title("ICC falls apart once the sample's size spread is removed",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right", labelcolor=C_MUTED,
              handletextpad=0.4)
    _style(ax)
    ax.tick_params(axis="x", length=0, labelsize=10)
    for t in ax.get_xticklabels():
        t.set_color(C_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def fig_correction(rates, path):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    labels = [r[0] for r in rates]
    vals = [r[1] for r in rates]
    colors = [C_MUTED if i == 0 else C_BLUE for i in range(len(rates))]
    y = np.arange(len(rates))
    ax.barh(y, vals, height=0.52, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v - 1.2, yi, "%.1f%%" % v, va="center", ha="right",
                color="white", fontsize=10, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, color=C_MUTED, fontsize=9.5)
    ax.set_xlim(0, 104)
    ax.set_xlabel("instances inside exploratory candidate margins (%)")
    ax.set_title("The margin failures are removable bias, not measurement error",
                 fontsize=11, loc="left")
    ax.invert_yaxis()
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="paired_feature_table.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.inp)
    n = len(df)
    L = ["# Agreement, Second Pass: Bias vs. Noise (n=%d)" % n, "",
         "Supersedes the framing in `agreement_report.md`. Same data, same pipeline; the",
         "question changed from *do the representations agree?* to *what kind of",
         "disagreement is this, and is it removable?*", ""]

    # ---- 1. bias or noise -------------------------------------------------------
    e = df["vol_pct_err_occ_vs_mesh"].to_numpy()
    N = float(df["occ_n_points"].iloc[0])
    p = df["occ_fraction_inside"].to_numpy()
    mc = np.sqrt((1 - p) / (p * N)) * 100.0            # binomial MC relative SE, percent
    ci = boot_ci(e, np.mean, args.n_boot)
    L += ["## 1. This is bias, not noise", "",
          "| Quantity | Value |", "|---|---|",
          "| mean signed error (occupancy − mesh) | **%+.3f%%** (95%% CI %+.3f to %+.3f) |" % (e.mean(), ci[0], ci[1]),
          "| median signed error | %+.3f%% |" % np.median(e),
          "| SD of signed error | %.3f%% |" % e.std(ddof=1),
          "| share of instances on one side of zero | **%.2f%%** |" % (100 * max((e > 0).mean(), (e < 0).mean())),
          "| Monte-Carlo sampling floor (median, N=%d) | %.3f%% |" % (N, np.median(mc)),
          "| observed bias ÷ sampling floor | **%.1f×** |" % np.median(e / mc), "",
          "A Monte-Carlo volume estimator with %d query points is entitled to about" % N,
          "%.2f%% relative error at the median occupancy fraction. The observed" % np.median(mc),
          "discrepancy is %.1f× that and points the same direction for %.2f%% of instances." % (np.median(e / mc), 100 * (e > 0).mean()),
          "Sampling noise is therefore a minor component: the occupancy representation",
          "reports a *systematically larger* mitochondrion than the mesh does.", ""]

    # ---- 2. mechanism -----------------------------------------------------------
    sv = (df["mesh_area"] / df["mesh_volume"] ** (2.0 / 3.0)).to_numpy()
    b_sv, pred_sv, r2_sv = ols(e, sv)
    t = ((df["occ_volume_est"] - df["mesh_volume"]) / df["mesh_area"]).to_numpy()
    tn = t / df["occ_scale"].to_numpy()
    L += ["## 2. The bias has the signature of a constant boundary offset", "",
          "| Predictor of the signed error | Pearson r |", "|---|---|",
          "| surface-to-volume ratio `A / V^(2/3)` | **%+.3f** |" % np.corrcoef(sv, e)[0, 1],
          "| `mesh_sphericity` | %+.3f |" % np.corrcoef(df["mesh_sphericity"], e)[0, 1],
          "| `occ_fraction_inside` | %+.3f |" % np.corrcoef(p, e)[0, 1],
          "| `mesh_elongation` | %+.3f |" % np.corrcoef(df["mesh_elongation"], e)[0, 1], "",
          "If one representation describes a shape uniformly offset outward by a small",
          "distance *t*, the excess volume is ≈ *A·t*, so the percent error scales with",
          "surface-to-volume. That is exactly the observed pattern (R² = %.2f against" % r2_sv,
          "`A / V^(2/3)` alone). Solving for the implied offset:", "",
          "- `t = ΔV / A` in real units: median **%.4f**, CV %.2f" % (np.median(t), t.std() / t.mean()),
          "- the same offset divided by each instance's `scale`: median **%.5f**, CV %.2f" % (np.median(tn), tn.std() / tn.mean()),
          "",
          "The offset is far more consistent once expressed relative to `scale` (CV %.2f vs %.2f),"
          % (tn.std() / tn.mean(), t.std() / t.mean()),
          "i.e. it is constant in the *normalized* frame where the occupancy labels were",
          "generated, not in real units. An offset of %.5f of the normalized extent is about" % np.median(tn),
          "one cell of a %d³ grid (or half a cell of a %d³ grid) — consistent with the"
          % (round(1 / np.median(tn)), round(0.5 / np.median(tn))),
          "occupancy labels having been produced by a voxelization that counts boundary",
          "cells as interior. **This is a dataset property worth reporting in its own right",
          "and worth confirming against the 3DMSL generation code.**", ""]

    # ---- 3. correction ----------------------------------------------------------
    bbox = (df["occ_extent_x"] * df["occ_extent_y"] * df["occ_extent_z"]).to_numpy()
    fill = df["occ_volume_est"].to_numpy() / bbox
    _, pred_occ, r2_occ = ols(e, fill, p, np.log(df["occ_volume_est"]))
    margin = np.where(p < LOW_OCC_CUT, MARGIN_LOW_OCC, MARGIN_OTHER)
    within = lambda err: 100.0 * np.mean(np.abs(err) <= margin)
    rates = [("no correction", within(e)),
             ("subtract the global mean bias", within(e - e.mean())),
             ("occupancy-only model\n(no mesh needed)", within(e - pred_occ)),
             ("mesh-informed A/V model\n(upper bound)", within(e - pred_sv))]
    L += ["## 3. Because it is bias, it is correctable", "",
          "| Correction | R² | inside margin |", "|---|---|---|",
          "| none | — | **%.1f%%** |" % within(e),
          "| subtract the global mean (%+.2f%%) | — | **%.1f%%** |" % (e.mean(), within(e - e.mean())),
          "| occupancy-only regression (bbox fill, `occ_fraction_inside`, log V) | %.2f | **%.1f%%** |" % (r2_occ, within(e - pred_occ)),
          "| mesh-informed `A / V^(2/3)` regression | %.2f | **%.1f%%** |" % (r2_sv, within(e - pred_sv)), "",
          "The occupancy-only model matters most: it uses nothing but quantities available",
          "from the occupancy representation itself, so it is a usable estimator correction",
          "rather than a post-hoc fit against the answer. Subtracting even a single global",
          "constant recovers most of the gap, which is the clearest possible statement that",
          "the %.1f%% of instances outside the margin were never a precision problem." % (100 - within(e)), ""]

    # ---- 4. ICC and heterogeneity ----------------------------------------------
    V = df[["mesh_volume", "occ_volume_est"]].to_numpy()
    E = df[["mesh_extent_x", "pc_trimmed_extent_x", "occ_extent_x"]].to_numpy()
    qv = pd.qcut(df["mesh_volume"], 10, labels=False).to_numpy()
    qe = pd.qcut(df["mesh_extent_x"], 10, labels=False).to_numpy()
    dec_v = [icc21(V[qv == i]) for i in range(10)]
    dec_e = [icc21(E[qe == i]) for i in range(10)]
    whole_v, whole_e = icc21(V), icc21(E)
    L += ["## 4. The ICC is a property of this sample, not of the method", "",
          "The shard spans a %.0f× range of mesh volumes. ICC(2,1) is a ratio of"
          % (df.mesh_volume.max() / df.mesh_volume.min()),
          "between-instance variance to total variance, so that spread inflates it toward 1",
          "relative to the error variance. Recomputing ICC inside volume",
          "deciles — where instances are near-identical in size — removes that inflation:", "",
          "| Descriptor | whole sample | median within decile | decile range |", "|---|---|---|---|",
          "| volume (mesh vs. occupancy) | **%.4f** | **%.4f** | %.3f – %.3f |"
          % (whole_v, np.median(dec_v), min(dec_v), max(dec_v)),
          "| extent_x (3 representations) | **%.4f** | **%.4f** | %.3f – %.3f |"
          % (whole_e, np.median(dec_e), min(dec_e), max(dec_e)), "",
          "Volume agreement drops from “excellent” (%.3f) to the low end of “good”"
          % whole_v,
          "(%.3f) on the same measurements. Koo & Li (2016) warn about exactly this."
          % np.median(dec_v),
          "**Recommendation: report ICC with the sample's size range stated alongside it,",
          "and lead the results with the heterogeneity-free statistics below.**", ""]

    # ---- 5. heterogeneity-free agreement ---------------------------------------
    comps = [("volume: occupancy vs. mesh", df["mesh_volume"], df["occ_volume_est"]),
             ("extent_x: point-cloud (trimmed) vs. mesh", df["mesh_extent_x"], df["pc_trimmed_extent_x"]),
             ("extent_x: occupancy vs. mesh", df["mesh_extent_x"], df["occ_extent_x"])]
    L += ["## 5. Difference-based agreement statistics", "",
          "Bias here is the mean *absolute-unit* difference (and that difference as a share",
          "of the reference mean), so it differs slightly from the mean *per-instance* percent",
          "error in §1 — the latter weights small mitochondria equally with large ones.", "",
          "| Comparison | bias | 95% limits of agreement | percent error | median &#124;% diff&#124; | proportional bias r |",
          "|---|---|---|---|---|---|"]
    for lbl, a, b in comps:
        s = agreement_block(a, b)
        L.append("| %s | %+.3f (%+.2f%%) | %+.3f to %+.3f | %.2f%% | %.2f%% | %+.3f |"
                 % (lbl, s["bias"], s["bias_pct"], s["loa_lo"], s["loa_hi"],
                    s["percent_error"], s["median_abs_pct"], s["prop_bias_r"]))
    L += ["", "Percent error (Critchley & Critchley 1999) is the scale-free companion to the",
          "limits of agreement and is the number to quote when readers ask how good the",
          "method is. Volume comes out at %.1f%% — a very different message from ICC 0.994."
          % agreement_block(df["mesh_volume"], df["occ_volume_est"])["percent_error"], ""]

    # ---- 6. the point cloud has its own bias -----------------------------------
    L += ["## 6. The trimmed point-cloud extent carries its own (smaller) bias", "",
          "| Axis | mean error vs. mesh | SD | share one-signed |", "|---|---|---|---|"]
    for ax in "xyz":
        c = df["extent_%s_pct_err_pc_vs_mesh" % ax]
        L.append("| %s | **%+.3f%%** | %.3f | %.1f%% |"
                 % (ax, c.mean(), c.std(ddof=1), 100 * max((c > 0).mean(), (c < 0).mean())))
    L += ["", "This is the cost of the 1st–99th-percentile trim adopted in Week 1–2: it buys",
          "immunity to the sampling-noise outlier at the price of a consistent ~2% shrinkage,",
          "because trimming discards real extremal surface points along with the noise. The",
          "trade is worth making and the bias is stable enough to state as a known offset —",
          "but it must be declared in the methods, not left implicit. The occupancy extents,",
          "by contrast, are near-unbiased (%+.3f%% to %+.3f%%)."
          % (min(df["extent_%s_pct_err_occ_vs_mesh" % a].mean() for a in "xyz"),
             max(df["extent_%s_pct_err_occ_vs_mesh" % a].mean() for a in "xyz")), ""]

    # ---- figures ---------------------------------------------------------------
    f1 = os.path.join(args.outdir, "bias_not_noise.png")
    f2 = os.path.join(args.outdir, "icc_heterogeneity.png")
    f3 = os.path.join(args.outdir, "margin_after_correction.png")
    fig_bias_not_noise(e, sv, (b_sv[0], b_sv[1], r2_sv), float(np.median(mc)), f1)
    fig_icc_collapse(whole_v, dec_v, whole_e, dec_e, f2)
    fig_correction(rates, f3)
    L += ["## Figures", "",
          "- `bias_not_noise.png` — one-signed error distribution; error vs. surface-to-volume",
          "- `icc_heterogeneity.png` — whole-sample ICC vs. within-size-decile ICC",
          "- `margin_after_correction.png` — margin pass rate before and after bias correction", ""]

    out = os.path.join(args.outdir, "agreement_v2_report.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    print("\nWrote %s, %s, %s, %s" % (out, f1, f2, f3))


if __name__ == "__main__":
    main()