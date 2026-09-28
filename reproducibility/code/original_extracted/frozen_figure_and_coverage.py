#!/usr/bin/env python3
"""Paper fixes 1 and 2 (frozen second-shard outputs).

1. Re-export the frozen-error box plot (Fig. 1) at 300 dpi PNG plus vector PDF,
   sized for one IEEE column (3.5 in).
2. Check the identical coverage values in Table IV (uncorrected vs. global
   correction: 96.4 / 96.4 internal, 90.0 / 90.0 frozen).

Why identical coverage can be expected. The global correction divides every
occupancy volume by the same factor k = 1 + c/100, where c is the mean training
error. Its percentage residual is therefore (e_raw - c) / k, and an object is
covered when
        c - b_g*k  <=  e_raw  <=  c + b_g*k .
The uncorrected method covers an object when -b_u <= e_raw <= b_u. Because
almost every raw error is positive, and the calibration quantiles of |e_raw|
and |e_raw - c| usually come from the same calibration object, the upper limits
c + b_g*k and b_u nearly coincide. Both methods then cover almost exactly the
same objects. The script measures how closely this holds and lists every object
on which the two methods disagree.

Nothing here refits or changes the frozen model. It only reads saved outputs.
"""
import argparse, glob, json, os, re, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PAPER = {"bounds": {"uncorrected": 8.451, "global": 4.632, "occ_only": 2.661},
         "frozen_coverage": {"uncorrected": 90.0, "global": 90.0, "occ_only": 92.1},
         "internal_coverage": {"uncorrected": 96.4, "global": 96.4, "occ_only": 96.2}}
LABELS = {"uncorrected": "Uncorrected", "global": "Global", "occ_only": "Occupancy-only"}

# Column-name patterns for the signed percentage error of each method. The
# first matching column is used. Override with --col-uncorrected etc.
PATTERNS = {
    "uncorrected": [r"^(e|err|error|pct_err|signed_err)_?(raw|uncorr)", r"uncorr.*(err|pct|resid)",
                    r"(err|pct|resid).*uncorr", r"^raw_.*(err|pct)", r"(err|pct).*_raw$"],
    "global": [r"glob.*(err|pct|resid)", r"(err|pct|resid).*glob"],
    "occ_only": [r"occ.*only.*(err|pct|resid)", r"(err|pct|resid).*occ.*only", r"ols.*(err|pct|resid)",
                 r"(err|pct|resid).*ols", r"occupancy_only"],
}
EXCLUDE = re.compile(r"abs|ape|within|cover|pass|bound|_ci|lo$|hi$|pred|hat", re.I)
ID_PAT = re.compile(r"^(instance_?id|object_?id|id|instance)$", re.I)


def find_tables(root, exclude=None):
    out = []
    for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True) + \
             glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True):
        if exclude and re.search(exclude, p):
            continue
        try:
            df = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
        except Exception:
            continue
        out.append((p, df))
    return out


def detect_cols(df, overrides):
    cols = {}
    for m, pats in PATTERNS.items():
        if overrides.get(m):
            cols[m] = overrides[m]
            continue
        hit = None
        for pat in pats:
            cands = [c for c in df.columns if re.search(pat, str(c), re.I) and not EXCLUDE.search(str(c))
                     and pd.api.types.is_numeric_dtype(df[c])]
            if cands:
                hit = cands[0]
                break
        cols[m] = hit
    return cols


def pick_table(root, overrides, min_rows, explicit=None, exclude=None):
    tables = [(explicit, pd.read_csv(explicit))] if explicit else find_tables(root, exclude)
    best = None
    for p, df in tables:
        if len(df) < min_rows:
            continue
        cols = detect_cols(df, overrides)
        score = sum(v is not None for v in cols.values())
        if best is None or score > best[0] or (score == best[0] and len(df) > len(best[2])):
            best = (score, p, df, cols)
    return best


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def find_bounds_and_offset(roots):
    """Look through saved JSON files for the calibrated bounds and the global offset."""
    bounds, offset, sources = {}, None, []
    keys = {"uncorrected": r"uncorr|raw", "global": r"glob", "occ_only": r"occ.*only|ols|occupancy_only"}
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
            try:
                data = json.load(open(p))
            except Exception:
                continue
            for path, val in flatten(data):
                low = path.lower()
                if re.search(r"bound|q95|quantile|qhat|cal.*abs", low):
                    for m, kp in keys.items():
                        if m not in bounds and re.search(kp, low) and 0 < val < 50:
                            bounds[m] = (val, f"{p}:{path}")
                if offset is None and re.search(r"glob", low) and re.search(r"offset|mean|shift|c$|const", low) \
                        and not re.search(r"bound|q95|cover|rmse|mdape|bias", low) and 0 < val < 20:
                    offset = (val, f"{p}:{path}")
    return bounds, offset


def coverage_check(df, cols, bounds, c_json, tag):
    e_u, e_g, e_o = (df[cols[m]].to_numpy(float) for m in ("uncorrected", "global", "occ_only"))
    b_u, b_g, b_o = (bounds[m] for m in ("uncorrected", "global", "occ_only"))
    cov_u, cov_g, cov_o = np.abs(e_u) <= b_u, np.abs(e_g) <= b_g, np.abs(e_o) <= b_o
    n = len(df)
    # Offset implied by the stored errors: e_g = (e_u - c)/(1 + c/100)  =>  c = (e_u - e_g)/(1 + e_g/100)
    c_obj = (e_u - e_g) / (1 + e_g / 100)
    c_est = float(np.median(c_obj))
    algebra_spread = float(np.max(np.abs(c_obj - c_est)))
    c = c_json if c_json is not None else c_est
    k = 1 + c / 100
    lo_g, hi_g = c - b_g * k, c + b_g * k
    both = int(np.sum(cov_u & cov_g)); only_u = int(np.sum(cov_u & ~cov_g)); only_g = int(np.sum(~cov_u & cov_g))
    ids = df[[x for x in df.columns if ID_PAT.match(str(x))][0]].astype(str).to_numpy() \
        if any(ID_PAT.match(str(x)) for x in df.columns) else np.arange(n).astype(str)
    disagree = pd.DataFrame({"id": ids, "e_uncorrected": e_u, "e_global": e_g,
                             "covered_uncorrected": cov_u, "covered_global": cov_g})[cov_u != cov_g]
    res = {
        "evaluation": tag, "n": n,
        "coverage_pct": {"uncorrected": 100 * cov_u.mean(), "global": 100 * cov_g.mean(), "occ_only": 100 * cov_o.mean()},
        "covered_counts": {"uncorrected": int(cov_u.sum()), "global": int(cov_g.sum()), "occ_only": int(cov_o.sum())},
        "cross_tab": {"covered_by_both": both, "uncorrected_only": only_u, "global_only": only_g,
                      "neither": int(np.sum(~cov_u & ~cov_g))},
        "global_offset_c": c, "global_offset_source": "saved JSON" if c_json is not None else "implied by stored errors",
        "global_offset_implied_by_errors": c_est,
        "max_deviation_from_global_formula": algebra_spread,
        "raw_error_window_uncorrected": [-b_u, b_u],
        "raw_error_window_global": [lo_g, hi_g],
        "upper_limit_difference_pp": hi_g - b_u,
        "n_raw_errors_below_global_lower_limit": int(np.sum(e_u < lo_g)),
        "n_raw_errors_in_gap_between_upper_limits": int(np.sum((e_u > min(b_u, hi_g)) & (e_u <= max(b_u, hi_g)))),
        "n_negative_raw_errors": int(np.sum(e_u < 0)),
    }
    same = only_u == 0 and only_g == 0
    res["verdict"] = (
        ("IDENTICAL covered sets. " if same else f"Covered sets differ on {only_u + only_g} object(s), "
         f"which offset in the coverage count ({only_u} vs {only_g}). ")
        + f"The global method covers raw errors in [{lo_g:.3f}, {hi_g:.3f}] and the uncorrected method covers "
          f"[{-b_u:.3f}, {b_u:.3f}]. The upper limits differ by {hi_g - b_u:+.3f} percentage points and "
          f"{res['n_raw_errors_below_global_lower_limit']} raw error(s) fall below the global lower limit, so the "
          "equal coverage is an arithmetic consequence of a one-signed error distribution, not a copy error."
        if algebra_spread < 1e-3 else
        "WARNING: the stored global errors do not follow e_g = (e_u - c)/(1 + c/100) for a single c "
        f"(max deviation {algebra_spread:.4f}). Check which columns were read before interpreting coverage.")
    return res, disagree


def make_figure(df, cols, out_dir, n_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    data = [df[cols[m]].dropna().to_numpy(float) for m in ("uncorrected", "global", "occ_only")]
    fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
    bp = ax.boxplot(data, showfliers=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color="#bd632d", lw=1.4), whiskerprops=dict(lw=0.9),
                    capprops=dict(lw=0.9), boxprops=dict(lw=0.9))
    for b in bp["boxes"]:
        b.set_facecolor("#dce7ef"); b.set_edgecolor("#28648a")
    ax.axhline(0, color="#777777", ls="--", lw=0.8, zorder=0)
    ax.set_xticks([1, 2, 3], [LABELS[m] for m in ("uncorrected", "global", "occ_only")])
    ax.set_ylabel("Volume error relative to mesh (%)")
    ax.grid(axis="y", alpha=0.15); ax.set_axisbelow(True)
    stem = os.path.join(out_dir, "frozen_errors")
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    stats = {}
    for m, x in zip(("uncorrected", "global", "occ_only"), data):
        q1, med, q3 = np.percentile(x, [25, 50, 75]); iqr = q3 - q1
        stats[m] = {"n": int(len(x)), "q1": q1, "median": med, "q3": q3,
                    "whisker_lo": float(x[x >= q1 - 1.5 * iqr].min()), "whisker_hi": float(x[x <= q3 + 1.5 * iqr].max()),
                    "n_beyond_whiskers": int(np.sum((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)))}
    return stem, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-dir", required=True, help="frozen_validation_1_1_2729/<timestamp> folder")
    ap.add_argument("--frozen-table", help="per-object CSV (auto-detected if omitted)")
    ap.add_argument("--dev-dir", help="development outputs folder, for the internal-test check (optional)")
    ap.add_argument("--dev-table", help="development per-object CSV with a split column (optional)")
    ap.add_argument("--split-col", default=None, help="column holding train/cal/test (auto-detected)")
    ap.add_argument("--out", required=True)
    for m in PATTERNS:
        ap.add_argument(f"--col-{m.replace('_', '-')}", dest=f"col_{m}")
    for m in PATTERNS:
        ap.add_argument(f"--bound-{m.replace('_', '-')}", dest=f"bound_{m}", type=float)
    ap.add_argument("--global-offset", type=float, help="global correction c (%%), if not found in JSON")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    overrides = {m: getattr(a, f"col_{m}") for m in PATTERNS}

    best = pick_table(a.frozen_dir, overrides, 2000, a.frozen_table)
    if best is None or best[0] < 3:
        print("Could not find a per-object table with signed errors for all three methods.")
        if best:
            print("Best candidate:", best[1]); print("Columns:", list(best[2].columns))
        print("Rerun with --frozen-table and --col-uncorrected/--col-global/--col-occ-only.")
        sys.exit(1)
    _, tpath, fdf, cols = best
    print(f"Frozen table: {tpath}  (n={len(fdf)})")
    for m, c in cols.items():
        print(f"  {LABELS[m]:15s} <- column '{c}'  (mean {fdf[c].mean():+.3f}, median |e| {fdf[c].abs().median():.3f})")

    found, off = find_bounds_and_offset([a.frozen_dir] + ([a.dev_dir] if a.dev_dir else []))
    bounds, bsrc = {}, {}
    for m in PATTERNS:
        if getattr(a, f"bound_{m}") is not None:
            bounds[m], bsrc[m] = getattr(a, f"bound_{m}"), "command line"
        elif m in found:
            bounds[m], bsrc[m] = found[m]
        else:
            bounds[m], bsrc[m] = PAPER["bounds"][m], "paper value (rounded to 3 dp)"
        print(f"  bound {LABELS[m]:15s} = {bounds[m]:.6g}   [{bsrc[m]}]")
    c_json = a.global_offset if a.global_offset is not None else (off[0] if off else None)
    if off and a.global_offset is None:
        print(f"  global offset c = {off[0]:.6g}   [{off[1]}]")

    report = {"run_utc": datetime.now(timezone.utc).isoformat(), "frozen_table": tpath, "columns": cols,
              "bounds": bounds, "bound_sources": bsrc, "paper_values": PAPER}
    res, dis = coverage_check(fdf, cols, bounds, c_json, "frozen second shard")
    report["frozen"] = res
    dis.to_csv(os.path.join(a.out, "coverage_disagreements_frozen.csv"), index=False)

    # Optional internal-test check
    dev = None
    if a.dev_table:
        dev = pd.read_csv(a.dev_table)
    elif a.dev_dir:
        cand = pick_table(a.dev_dir, overrides, 400, exclude=r"frozen_validation|remaining_analyses|paper_fixes")
        if cand and cand[0] == 3:
            dev = cand[2]; print(f"Development table: {cand[1]}")
    if dev is not None:
        scol = a.split_col or next((c for c in dev.columns if re.search(r"split|partition|fold|set$", str(c), re.I)), None)
        if scol is None and 400 <= len(dev) <= 700:
            print(f"Development table has {len(dev)} rows and no split column; treating it as the internal test set.")
            test = dev
        elif scol is None:
            print("Development table has no split column; skipping internal-test check (use --split-col).")
            test = dev.iloc[0:0]
        else:
            test = dev[dev[scol].astype(str).str.lower().str.startswith("test")]
        if len(test):
            dcols = detect_cols(dev, overrides)
            if all(dcols.values()):
                r2, d2 = coverage_check(test.reset_index(drop=True), dcols, bounds, c_json, "internal test")
                report["internal_test"] = r2
                d2.to_csv(os.path.join(a.out, "coverage_disagreements_internal.csv"), index=False)

    stem, stats = make_figure(fdf, cols, a.out, len(fdf))
    report["figure"] = {"png": stem + ".png", "pdf": stem + ".pdf", "dpi": 300, "width_in": 3.5, "box_stats": stats}
    json.dump(report, open(os.path.join(a.out, "coverage_check.json"), "w"), indent=2, default=float)

    print("\n=== Coverage check ===")
    for key in ("internal_test", "frozen"):
        if key not in report:
            continue
        r = report[key]
        pv = PAPER["internal_coverage" if key == "internal_test" else "frozen_coverage"]
        print(f"\n{r['evaluation']} (n={r['n']})")
        for m in PATTERNS:
            print(f"  {LABELS[m]:15s} {r['coverage_pct'][m]:6.2f}%  ({r['covered_counts'][m]}/{r['n']})   paper: {pv[m]}%")
        print("  cross-tab:", r["cross_tab"])
        print("  " + r["verdict"])
    print(f"\nFigure written: {stem}.png (300 dpi) and {stem}.pdf")
    print("Copy frozen_errors.pdf into the Overleaf figures/ folder (main.tex can point to the .pdf).")


if __name__ == "__main__":
    main()
