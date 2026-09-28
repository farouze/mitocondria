#!/usr/bin/env python3
"""Paper fix 3: rerun the E5 label-boundary fits with a convergence check.

For each object:
  * near-boundary occupancy query points are selected as in the original run
    (approximate distance from the 32 nearest triangle centroids, |d| < band,
    at most 3,000 points, fixed seed);
  * signed distances are computed twice: with the original 32-centroid
    approximation and EXACTLY (brute-force point-to-triangle distance over all
    faces, sign from the generalized winding number). This also answers the
    second E5 limitation, that the approximation was never validated;
  * P(inside | d) = 1 / (1 + exp((d - t50)/w)) is fitted by maximum likelihood
    (Bernoulli), from several starting points, and each fit is checked:
      - optimizer reports success
      - projected gradient small
      - t50 and w not on a parameter bound
      - all starts that reach the best likelihood agree on t50
      - both labels present (>= 20 points each) and the classes overlap
    Fits that fail any check are reported, not silently used.

Positive distance = outside the raw mesh, in normalized coordinates.
The volume-implied offset is t_vol = (V_o - V_m) / (A_m * s).
"""
import argparse, glob, json, os, re, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr

ORIGINAL = {"n": 60, "t50_median": 0.00304, "t50_iqr": [0.00285, 0.00317], "t50_cv": 0.09,
            "w_median": 0.00027, "t_vol_median": 0.00311, "ratio": 0.98, "per_object_r": 0.567}
PIPELINE = {"full_ray_offset": 1.5 / 256 / 0.9, "half_ray_offset": 0.5 * 1.5 / 256 / 0.9}


# ----------------------------------------------------------------- loading
def load_instance(d):
    import trimesh
    mesh = trimesh.load(os.path.join(d, "raw_mesh.off"), process=False, force="mesh")
    z = np.load(os.path.join(d, "occupancies.npz"))
    pts = z["points"].astype(np.float64)
    occ = z["occupancies"]
    if occ.dtype == np.uint8 and occ.size * 8 >= len(pts) and occ.size != len(pts):
        occ = np.unpackbits(occ)[: len(pts)]
    occ = occ.astype(bool).reshape(-1)
    loc = np.asarray(z["loc"], dtype=np.float64).reshape(3)
    scale = float(np.asarray(z["scale"]).reshape(()))
    return mesh, pts, occ, loc, scale


# ----------------------------------------------------------------- geometry
def closest_point_on_triangles(p, a, b, c):
    """Exact closest points (Ericson, Real-Time Collision Detection 5.1.5), vectorized.
    p: (m,1,3); a,b,c: (1,f,3) -> squared distances (m,f)."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = np.einsum("...k,...k", ab, ap); d2 = np.einsum("...k,...k", ac, ap)
    bp = p - b
    d3 = np.einsum("...k,...k", ab, bp); d4 = np.einsum("...k,...k", ac, bp)
    cp = p - c
    d5 = np.einsum("...k,...k", ab, cp); d6 = np.einsum("...k,...k", ac, cp)
    va = d3 * d6 - d5 * d4; vb = d5 * d2 - d1 * d6; vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        v = vb / denom; w = vc / denom
    q = a + ab * v[..., None] + ac * w[..., None]              # interior
    # edges
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ab = d1 / (d1 - d3); t_ac = d2 / (d2 - d6); t_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    q = np.where(m_bc[..., None], b + (c - b) * t_bc[..., None], q)
    q = np.where(m_ac[..., None], a + ac * t_ac[..., None], q)
    q = np.where(m_ab[..., None], a + ab * t_ab[..., None], q)
    # vertices (checked last so they take precedence)
    q = np.where(((d6 >= 0) & (d5 <= d6))[..., None], np.broadcast_to(c, q.shape), q)
    q = np.where(((d3 >= 0) & (d4 <= d3))[..., None], np.broadcast_to(b, q.shape), q)
    q = np.where(((d1 <= 0) & (d2 <= 0))[..., None], np.broadcast_to(a, q.shape), q)
    diff = p - q
    return np.einsum("...k,...k", diff, diff)


def winding_number(P, V, F, chunk=None):
    """Generalized winding number (Jacobson et al. 2013) via the Van Oosterom-Strackee solid angle."""
    A, B, C = V[F[:, 0]][None], V[F[:, 1]][None], V[F[:, 2]][None]
    chunk = chunk or max(1, int(1.5e6 // len(F)))          # keeps memory near 0.5 GB
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk, None, :]
        a, b, c = A - p, B - p, C - p
        la, lb, lc = (np.linalg.norm(x, axis=-1) for x in (a, b, c))
        det = np.einsum("...k,...k", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("...k,...k", a, b) * lc + np.einsum("...k,...k", b, c) * la \
            + np.einsum("...k,...k", c, a) * lb
        out[i:i + chunk] = np.arctan2(det, den).sum(axis=1) / (2 * np.pi)
    return out


def exact_unsigned(P, V, F, chunk=None):
    A, B, C = V[F[:, 0]][None], V[F[:, 1]][None], V[F[:, 2]][None]
    chunk = chunk or max(1, int(4e5 // len(F)))
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = np.sqrt(closest_point_on_triangles(P[i:i + chunk, None, :], A, B, C).min(axis=1))
    return out


def exact_unsigned_pruned(P, V, F, tree, upper):
    """Exact distance using only triangles that can possibly be closer than a known upper bound.
    A triangle within distance u of p has its centroid within u + r_max of p, where r_max is the
    largest centroid-to-vertex distance, so no candidate is missed."""
    tri = V[F]
    cent = tri.mean(axis=1)
    r_max = float(np.linalg.norm(tri - cent[:, None, :], axis=-1).max())
    out = np.empty(len(P))
    for i, (p, u) in enumerate(zip(P, upper)):
        idx = np.asarray(tree.query_ball_point(p, u + r_max + 1e-12), dtype=int)
        t = tri[idx]
        out[i] = np.sqrt(closest_point_on_triangles(p[None, None, :], t[None, :, 0], t[None, :, 1], t[None, :, 2]).min())
    return out


def approx_unsigned(P, V, F, tree, k=32):
    """Original method: exact distance to the k triangles with the nearest centroids."""
    _, idx = tree.query(P, k=min(k, len(F)))
    idx = np.atleast_2d(idx)
    a, b, c = V[F[idx, 0]], V[F[idx, 1]], V[F[idx, 2]]
    return np.sqrt(closest_point_on_triangles(P[:, None, :], a, b, c).min(axis=1))


# ----------------------------------------------------------------- fitting
def nll(theta, d, y):
    t, logw = theta
    w = np.exp(logw)
    z = (d - t) / w                     # P(inside) = sigmoid(-z)
    # -log L = sum y*log(1+e^z) + (1-y)*log(1+e^-z)
    return float(np.sum(y * np.logaddexp(0, z) + (1 - y) * np.logaddexp(0, -z)))


def grad(theta, d, y):
    t, logw = theta
    w = np.exp(logw)
    z = (d - t) / w
    s = expit(z)                        # = 1 - P(inside)
    r = s - (1 - y)                     # d nll / dz
    return np.array([np.sum(r * (-1 / w)), np.sum(r * (-z))])


def fit_logistic(d, y, band, starts=None):
    y = y.astype(float)
    lo = np.array([-band, np.log(1e-6)]); hi = np.array([band, np.log(band)])
    if starts is None:
        # crude midpoint: where the running inside-fraction crosses 0.5
        order = np.argsort(d); cum = np.cumsum(y[order]) / np.arange(1, len(d) + 1)
        t0 = float(d[order][np.argmin(np.abs(cum - 0.5))]) if len(d) else 0.0
        starts = [(t0, np.log(3e-4)), (0.0, np.log(1e-3)), (band / 4, np.log(1e-4)), (-band / 4, np.log(1e-3)),
                  (0.003, np.log(3e-4))]
    fits = []
    for s in starts:
        x0 = np.clip(np.array(s, float), lo + 1e-9, hi - 1e-9)
        r = minimize(nll, x0, args=(d, y), jac=grad, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10})
        fits.append(r)
    best = min(fits, key=lambda r: r.fun)
    near = [r for r in fits if r.fun - best.fun < 1e-6 * max(1.0, abs(best.fun))]
    t_spread = float(np.ptp([r.x[0] for r in near])) if len(near) > 1 else 0.0
    t, logw = best.x
    g = grad(best.x, d, y)
    at_lo = best.x <= lo + 1e-3 * (hi - lo); at_hi = best.x >= hi - 1e-3 * (hi - lo)
    pg = np.where((at_lo & (g > 0)) | (at_hi & (g < 0)), 0.0, g)      # projected gradient
    pg = pg * np.array([np.exp(logw), 1.0]) / max(1, len(d))         # scale-free, per point
    n_in, n_out = int(y.sum()), int(len(y) - y.sum())
    # class overlap: any inside point beyond the outermost outside point's inner edge
    overlap = n_in > 0 and n_out > 0 and d[y == 1].max() > d[y == 0].min()
    reasons = []
    if not best.success: reasons.append("optimizer_not_success")
    if np.max(np.abs(pg)) > 1e-4: reasons.append("gradient_not_small")
    if at_lo[0] or at_hi[0]: reasons.append("t50_on_bound")
    if at_lo[1]: reasons.append("width_on_lower_bound")
    if at_hi[1]: reasons.append("width_on_upper_bound")
    if t_spread > 1e-5: reasons.append("starts_disagree")
    if min(n_in, n_out) < 20: reasons.append("too_few_points_in_one_class")
    if not overlap: reasons.append("perfectly_separable")
    # standard error of t50 from the observed information (analytic Hessian of the Bernoulli NLL)
    w = np.exp(logw); z = (d - t) / w; p = expit(z); v = p * (1 - p)
    H = np.array([[np.sum(v) / w**2, np.sum(v * z) / w], [np.sum(v * z) / w, np.sum(v * z * z)]])
    # (logw block uses dz/dlogw = -z; cross terms ignore the residual term, i.e. expected information)
    try:
        se_t = float(np.sqrt(np.linalg.inv(H)[0, 0]))
    except np.linalg.LinAlgError:
        se_t = float("nan")
    return {"t50": float(t), "w": float(w), "nll": float(best.fun), "success": bool(best.success),
            "n_iter": int(best.nit), "max_proj_grad": float(np.max(np.abs(pg))), "t50_spread_across_starts": t_spread,
            "n_starts_at_optimum": len(near), "se_t50": se_t, "n_inside": n_in, "n_outside": n_out,
            "converged": len(reasons) == 0, "fail_reasons": ";".join(reasons)}


# ----------------------------------------------------------------- per object
def process(d, band, max_points, seed, k_approx, brute_force=False):
    mesh, pts, occ, loc, s = load_instance(d)
    Vn = (np.asarray(mesh.vertices, float) - loc) / s          # mesh -> normalized frame
    F = np.asarray(mesh.faces, int)
    cent = Vn[F].mean(axis=1)
    tree = cKDTree(cent)
    # preselect by nearest-vertex distance (an upper bound on the true distance, loosened by edge length)
    edges = np.linalg.norm(Vn[F[:, [1, 2, 0]]] - Vn[F], axis=-1)
    dv, _ = cKDTree(Vn).query(pts)
    pre = np.where(dv < band + edges.max())[0]
    d_apx_all = approx_unsigned(pts[pre], Vn, F, tree, k_approx)
    keep = pre[d_apx_all < band]
    rng = np.random.default_rng(seed)
    if len(keep) > max_points:
        keep = np.sort(rng.choice(keep, max_points, replace=False))
    P, y = pts[keep], occ[keep]
    wn = winding_number(P, Vn, F)
    sign = np.where(wn > 0.5, -1.0, 1.0)                     # inside -> negative
    u_apx = approx_unsigned(P, Vn, F, tree, k_approx)
    d_apx = sign * u_apx
    d_ex = sign * (exact_unsigned(P, Vn, F) if brute_force else exact_unsigned_pruned(P, Vn, F, tree, u_apx))
    diff = np.abs(np.abs(d_apx) - np.abs(d_ex))
    fe = fit_logistic(d_ex, y, band)
    fa = fit_logistic(d_apx, y, band)
    # volume-implied offset
    V_m, A_m = abs(float(mesh.volume)), float(mesh.area)
    praw = pts * s + loc
    box = np.prod(praw.max(axis=0) - praw.min(axis=0))
    V_o = occ.mean() * box
    t_vol = (V_o - V_m) / (A_m * s)
    # label sanity: points clearly inside / outside
    # label sanity relative to the fitted transition: deep inside = d < t50 - 10w, far outside = d > t50 + 10w
    marg = 10 * max(fe["w"], 1e-4)
    deep_in, far_out = d_ex < fe["t50"] - marg, d_ex > fe["t50"] + marg
    row = {"n_faces": len(F), "n_points_used": len(P),
           "frac_deep_inside_labeled_out": float(np.mean(~y[deep_in])) if deep_in.any() else np.nan,
           "frac_far_outside_labeled_in": float(np.mean(y[far_out])) if far_out.any() else np.nan,
           "approx_minus_exact_max": float(diff.max()), "approx_minus_exact_median": float(np.median(diff)),
           "frac_points_approx_differs_gt_1e-6": float(np.mean(diff > 1e-6)),
           "t_vol": float(t_vol), "V_mesh": V_m, "A_mesh": A_m, "V_occ": float(V_o), "scale": s}
    row.update({f"exact_{k}": v for k, v in fe.items()})
    row.update({f"approx_{k}": v for k, v in fa.items()})
    return row


def find_instance_dirs(root):
    return {os.path.basename(os.path.dirname(p)): os.path.dirname(p)
            for p in glob.glob(os.path.join(root, "**", "raw_mesh.off"), recursive=True)
            if "__MACOSX" not in p}


def ids_from_previous(path_or_dir):
    """Return object IDs (and the original per-object results) from the earlier mechanism run."""
    cands = [path_or_dir] if os.path.isfile(path_or_dir) else \
        sorted(glob.glob(os.path.join(path_or_dir, "**", "*.csv"), recursive=True),
               key=lambda q: (0 if re.search(r"mech|boundary|offset", q, re.I) else 1, q))
    for p in cands:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        idc = next((c for c in df.columns if re.match(r"^(instance_?id|object_?id|id|instance)$", str(c), re.I)), None)
        if idc and 1 <= len(df) <= 500 and any(re.search(r"t50|midpoint|offset", str(c), re.I) for c in df.columns):
            return p, df, idc
    return None


def summarize(df, col_t, col_w):
    t = df[col_t].to_numpy(float); w = df[col_w].to_numpy(float); tv = df["t_vol"].to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(tv)
    q1, q3 = np.percentile(t[ok], [25, 75]) if ok.any() else (np.nan, np.nan)
    return {"n": int(ok.sum()), "t50_median": float(np.median(t[ok])), "t50_iqr": [float(q1), float(q3)],
            "t50_cv": float(np.std(t[ok], ddof=1) / np.mean(t[ok])) if ok.sum() > 1 else np.nan,
            "w_median": float(np.median(w[ok])), "t_vol_median": float(np.median(tv[ok])),
            "ratio_t50_to_tvol": float(np.median(t[ok]) / np.median(tv[ok])),
            "pearson_r": float(pearsonr(t[ok], tv[ok])[0]) if ok.sum() > 2 else np.nan,
            "spearman_rho": float(spearmanr(t[ok], tv[ok])[0]) if ok.sum() > 2 else np.nan,
            "t50_over_half_ray": float(np.median(t[ok]) / PIPELINE["half_ray_offset"]),
            "t50_over_full_ray": float(np.median(t[ok]) / PIPELINE["full_ray_offset"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-root", required=True, help="extracted 10_24553_27272 folder")
    ap.add_argument("--previous", help="earlier mechanism_offset output (CSV or folder) to reuse its 60 IDs")
    ap.add_argument("--n", type=int, default=60, help="objects to sample if --previous is not given")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band", type=float, default=0.02, help="max |distance| of selected points (normalized)")
    ap.add_argument("--max-points", type=int, default=3000)
    ap.add_argument("--k-approx", type=int, default=32)
    ap.add_argument("--brute-force", action="store_true", help="exact distance over all faces (slow; for checking)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dirs = find_instance_dirs(a.shard_root)
    if not dirs:
        sys.exit(f"No raw_mesh.off files found under {a.shard_root}")
    prev = ids_from_previous(a.previous) if a.previous else None
    if prev:
        ppath, pdf_, idc = prev
        ids = [str(x) for x in pdf_[idc]]
        print(f"Reusing {len(ids)} IDs from {ppath}")
    else:
        if a.previous:
            print("No usable per-object table found in --previous; sampling new IDs instead.")
        ids = sorted(np.random.default_rng(a.seed).choice(sorted(dirs), size=min(a.n, len(dirs)), replace=False))
        pdf_ = None
    missing = [i for i in ids if i not in dirs]
    if missing:
        print(f"WARNING: {len(missing)} IDs not found in the shard, e.g. {missing[:5]}")
    ids = [i for i in ids if i in dirs]

    out_csv = os.path.join(a.out, "e5_refit_per_object.csv")
    done = pd.read_csv(out_csv, dtype={"id": str}) if os.path.exists(out_csv) else pd.DataFrame()
    rows = done.to_dict("records")
    have = set(done["id"]) if len(done) else set()
    for j, i in enumerate(ids):
        if i in have:
            continue
        t0 = time.time()
        try:
            r = process(dirs[i], a.band, a.max_points, a.seed + j, a.k_approx, a.brute_force)
            r["id"] = i; r["error"] = ""
        except Exception as e:                      # keep going; record the failure
            r = {"id": i, "error": repr(e), "exact_converged": False, "exact_fail_reasons": "processing_error"}
        rows.append(r)
        pd.DataFrame(rows).to_csv(out_csv, index=False)   # checkpoint after every object
        print(f"[{j + 1}/{len(ids)}] {i}: t50={r.get('exact_t50', float('nan')):.5f} "
              f"converged={r.get('exact_converged')} {r.get('exact_fail_reasons', '')} ({time.time() - t0:.1f}s)")

    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    conv = df[df["exact_converged"] == True]
    fails = df[df["exact_converged"] != True]
    reasons = {}
    for s in fails.get("exact_fail_reasons", pd.Series(dtype=str)).fillna(""):
        for k in filter(None, str(s).split(";")):
            reasons[k] = reasons.get(k, 0) + 1
    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(), "settings": vars(a),
        "n_objects": len(df), "n_converged": len(conv), "n_failed": len(fails), "fail_reason_counts": reasons,
        "failed_ids": fails["id"].tolist(),
        "exact_all": summarize(df.dropna(subset=["exact_t50"]), "exact_t50", "exact_w") if "exact_t50" in df else None,
        "exact_converged_only": summarize(conv, "exact_t50", "exact_w") if len(conv) > 2 else None,
        "approx_converged_only": summarize(df[df.get("approx_converged") == True], "approx_t50", "approx_w")
        if "approx_converged" in df and (df["approx_converged"] == True).sum() > 2 else None,
        "approx_vs_exact_distance": {
            "max_abs_diff": float(df["approx_minus_exact_max"].max()),
            "median_of_object_medians": float(df["approx_minus_exact_median"].median()),
            "median_frac_points_differing": float(df["frac_points_approx_differs_gt_1e-6"].median()),
            "t50_abs_change_median": float((df["approx_t50"] - df["exact_t50"]).abs().median()),
            "t50_abs_change_max": float((df["approx_t50"] - df["exact_t50"]).abs().max())},
        "label_sanity": {"max_frac_deep_inside_labeled_out": float(df["frac_deep_inside_labeled_out"].max()),
                         "max_frac_far_outside_labeled_in": float(df["frac_far_outside_labeled_in"].max())},
        "original_reported": ORIGINAL, "pipeline_reference": PIPELINE,
    }
    if pdf_ is not None:
        tcol = next((c for c in pdf_.columns if re.search(r"t50|midpoint", str(c), re.I)), None)
        if tcol:
            m = df.merge(pdf_[[idc, tcol]].rename(columns={idc: "id", tcol: "t50_original"}).astype({"id": str}), on="id")
            summary["vs_original_run"] = {"n_matched": len(m),
                                          "median_abs_change_t50": float((m["exact_t50"] - m["t50_original"]).abs().median()),
                                          "max_abs_change_t50": float((m["exact_t50"] - m["t50_original"]).abs().max())}
    json.dump(summary, open(os.path.join(a.out, "e5_refit_summary.json"), "w"), indent=2, default=float)

    s = summary["exact_converged_only"] or summary["exact_all"]
    ad = summary["approx_vs_exact_distance"]
    print("\n=== E5 refit ===")
    print(f"objects {len(df)}, converged {len(conv)}, failed {len(fails)} {reasons if reasons else ''}")
    print(f"converged fits: t50 median {s['t50_median']:.5f} (IQR {s['t50_iqr'][0]:.5f}-{s['t50_iqr'][1]:.5f}, "
          f"CV {s['t50_cv']:.2f}), w median {s['w_median']:.5f}")
    print(f"volume-implied offset median {s['t_vol_median']:.5f}; ratio {s['ratio_t50_to_tvol']:.2f}; "
          f"per-object r {s['pearson_r']:.3f}")
    print(f"t50 / half-ray reference {s['t50_over_half_ray']:.2f}; / full-ray {s['t50_over_full_ray']:.2f}")
    print(f"approx vs exact distance: max diff {ad['max_abs_diff']:.2e}; median t50 change {ad['t50_abs_change_median']:.2e}")
    print(f"original paper values: {ORIGINAL}")
    print("\nSuggested Methods/Limitations wording:")
    print(f"  \"Each fit was accepted only if the optimizer converged, the projected gradient was small, neither "
          f"parameter lay on a bound, and five starting points agreed on t50; {len(conv)} of {len(df)} fits met all "
          f"criteria. Distances were computed exactly (all faces, winding-number sign); the 32-centroid approximation "
          f"differed from the exact distance by at most {ad['max_abs_diff']:.1e} normalized units and changed t50 by a "
          f"median of {ad['t50_abs_change_median']:.1e}.\"")


if __name__ == "__main__":
    main()
