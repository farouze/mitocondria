#!/usr/bin/env python3
"""
mechanism_offset.py - direct test of the boundary-offset hypothesis for the occupancy
volume bias.

For a random sample of instances it measures, for every occupancy query point near the
surface, its exact signed distance to raw_mesh.off (normalized frame, positive = outside
the mesh) and fits P(labelled inside | distance) with a logistic curve. The distance
where that curve crosses 0.5 is where the occupancy labels place the surface. If the
labels come from a slightly inflated copy of the mesh, the crossing is positive and
roughly constant in the normalized frame.

The measured offset is then compared with the offset implied by the volume error of
the same instances, t = (V_occ - V_mesh) / (A_mesh * scale). Agreement means the whole
volume bias is explained by where the labels put the boundary.

No rtree/embree needed: distances use exact point-triangle distances to the k nearest
triangles, and inside/outside uses the generalized winding number.

  python3 mechanism_offset.py --root /path/to/shard --outdir out --sample 60
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from hit_common import find_instance_dirs, instance_id, load_mesh, load_occupancy, raw_extent, unnormalize

BAND = 0.02            # normalized units either side of the surface
K_TRI = 32             # candidate triangles per point for the exact distance
MAX_BAND_POINTS = 3000 # random subset per instance (enough for the logistic fit)
BINS = np.arange(-0.012, 0.0121, 0.0005)


def winding_number(points, tris, chunk=48):
    """Generalized winding number (~1 inside, ~0 outside) for a closed mesh."""
    out = np.empty(len(points))
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        a = tris[None, :, 0, :] - p[:, None, :]
        b = tris[None, :, 1, :] - p[:, None, :]
        c = tris[None, :, 2, :] - p[:, None, :]
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum("pfi,pfi->pf", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("pfi,pfi->pf", a, b) * lc
               + np.einsum("pfi,pfi->pf", b, c) * la + np.einsum("pfi,pfi->pf", c, a) * lb)
        out[s:s + chunk] = (2.0 * np.arctan2(det, den)).sum(axis=1) / (4 * np.pi)
    return out


def unsigned_distance(points, tris, centroid_tree):
    import trimesh
    k = min(K_TRI, len(tris))
    _, idx = centroid_tree.query(points, k=k)
    idx = idx.reshape(len(points), k)
    cand = tris[idx.ravel()]
    reps = np.repeat(points, k, axis=0)
    closest = trimesh.triangles.closest_point(cand, reps)
    d = np.linalg.norm(closest - reps, axis=1).reshape(len(points), k)
    return d.min(axis=1)


def fit_logistic(d, y):
    """MLE of P(inside) = 1 / (1 + exp((d - t) / s)); returns (t, s)."""
    if len(d) < 50 or y.all() or not y.any():
        return np.nan, np.nan

    def nll(theta):
        t, log_s = theta
        z = np.clip((d - t) / np.exp(log_s), -50, 50)
        p = 1.0 / (1.0 + np.exp(z))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    res = minimize(nll, x0=[0.0, np.log(0.001)], method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-6, maxiter=2000))
    return float(res.x[0]), float(np.exp(res.x[1]))


def one_instance(d):
    m = load_mesh(d)
    pts, occ, loc, scale = load_occupancy(d)
    s = float(np.asarray(scale).reshape(-1)[0])
    verts_n = (np.asarray(m.vertices, float) - loc) / s          # mesh in the normalized frame
    tris = verts_n[np.asarray(m.faces)]
    tree = cKDTree(tris.mean(axis=1))

    # cheap prefilter: distance to nearest centroid, then exact distance in the band
    dc, _ = tree.query(pts, k=1)
    edge = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1).max()
    near = dc < BAND + edge
    p_near = pts[near]
    dist = unsigned_distance(p_near, tris, tree)
    keep = dist < BAND
    p_band, dist, lab = p_near[keep], dist[keep], occ[near][keep].astype(bool)
    if len(p_band) > MAX_BAND_POINTS:
        sel = np.random.default_rng(len(p_band)).choice(len(p_band), MAX_BAND_POINTS, replace=False)
        p_band, dist, lab = p_band[sel], dist[sel], lab[sel]
    wn = winding_number(p_band, tris)
    signed = np.where(wn > 0.5, -dist, dist)                     # + outside the mesh

    t50, width = fit_logistic(signed, lab.astype(float))
    # volume-implied offset for the same instance (same formula as audit_3dmsl.py)
    pts_raw = unnormalize(pts, loc, scale)
    v_occ = float(occ.mean()) * float(np.prod(raw_extent(pts_raw)))
    v_mesh, a_mesh = float(abs(m.volume)), float(m.area)
    far_in = signed < -0.004
    far_out = signed > 0.008
    return dict(instance_id=instance_id(d), scale=s, n_band_points=int(len(signed)),
                offset_t50=t50, transition_width=width,
                offset_volume_implied=(v_occ - v_mesh) / a_mesh / s,
                vol_err_pct=100 * (v_occ / v_mesh - 1),
                inside_pts_labelled_empty=float((~lab[far_in]).mean()) if far_in.any() else np.nan,
                outside_pts_labelled_full=float(lab[far_out].mean()) if far_out.any() else np.nan,
                q_min=float(pts.min()), q_max=float(pts.max()),
                n_query=int(len(pts))), signed, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=790)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dirs = find_instance_dirs(args.root)
    rng = np.random.default_rng(args.seed)
    if args.sample and args.sample < len(dirs):
        dirs = [dirs[i] for i in sorted(rng.choice(len(dirs), args.sample, replace=False))]
    print("Mechanism test on %d instances" % len(dirs), flush=True)

    rows, all_d, all_y, t0 = [], [], [], time.time()
    for i, d in enumerate(dirs, 1):
        try:
            row, sd, lab = one_instance(d)
            rows.append(row); all_d.append(sd); all_y.append(lab)
        except Exception as e:
            rows.append(dict(instance_id=instance_id(d), error="%s: %s" % (type(e).__name__, e)))
        if i % 10 == 0 or i == len(dirs):
            print("  %d/%d (%.1fs/instance)" % (i, len(dirs), (time.time() - t0) / i), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "mechanism_offset_instances.csv"), index=False)
    ok = df.dropna(subset=["offset_t50"]) if "offset_t50" in df else df.iloc[:0]
    if ok.empty:
        raise RuntimeError("No instance produced an offset estimate; see mechanism_offset_instances.csv")

    D = np.concatenate(all_d); Y = np.concatenate(all_y)
    idx = np.digitize(D, BINS)
    prof = pd.DataFrame([dict(d_lo=BINS[i - 1], d_hi=BINS[i], n=int((idx == i).sum()),
                              p_inside=float(Y[idx == i].mean()) if (idx == i).any() else np.nan)
                         for i in range(1, len(BINS))])
    prof.to_csv(os.path.join(args.outdir, "mechanism_offset_profile.csv"), index=False)
    t_pool, w_pool = fit_logistic(D, Y.astype(float))

    t50 = ok["offset_t50"]; tv = ok["offset_volume_implied"]
    r = float(np.corrcoef(t50, tv)[0, 1]) if len(ok) > 2 else np.nan
    cv = lambda x: float(x.std() / abs(x.mean())) if x.mean() else np.nan
    q_lo, q_hi = float(ok["q_min"].min()), float(ok["q_max"].max())

    L = ["# Mechanism Test: Where Do the Occupancy Labels Put the Surface? (n=%d)" % len(ok), "",
         "Each occupancy query point within ±%.3f (normalized units, scale = largest mesh extent) of" % BAND,
         "`raw_mesh.off` was given its exact signed distance to the mesh (positive = outside).",
         "A logistic curve P(labelled inside | distance) was fitted per instance; its 50% point is",
         "where the labels place the surface relative to the mesh.", "",
         "## 1. Measured label offset", "",
         "| Quantity | Median | IQR | CV |", "|---|---|---|---|",
         "| label offset t50 (normalized) | **%.5f** | %.5f – %.5f | %.2f |"
         % (t50.median(), t50.quantile(.25), t50.quantile(.75), cv(t50)),
         "| transition width (logistic scale) | %.5f | %.5f – %.5f | |"
         % (ok.transition_width.median(), ok.transition_width.quantile(.25), ok.transition_width.quantile(.75)),
         "| offset implied by the volume error | **%.5f** | %.5f – %.5f | %.2f |"
         % (tv.median(), tv.quantile(.25), tv.quantile(.75), cv(tv)),
         "| pooled logistic fit, all points | %.5f (width %.5f) | | |" % (t_pool, w_pool), "",
         "Correlation between the measured and volume-implied offsets across instances: r = **%.3f**." % r,
         "Points well inside the mesh (d < −0.004) labelled empty: %.3f%% (median per instance)."
         % (100 * ok.inside_pts_labelled_empty.median()),
         "Points well outside (d > +0.008) labelled inside: %.3f%%." % (100 * ok.outside_pts_labelled_full.median()), ""]
    ratio = t50.median() / tv.median() if tv.median() else np.nan
    if t50.median() > 0 and 0.75 <= ratio <= 1.33:
        L += ["The labels put the surface **outside** the mesh by about the same distance the volume",
              "error implies (ratio %.2f). The occupancy volume bias is therefore explained by a boundary" % ratio,
              "offset of the labels relative to `raw_mesh.off`, not by the Monte-Carlo estimator.", ""]
    elif t50.median() > 0:
        L += ["The labels put the surface outside the mesh, but the measured offset is %.2f× the" % ratio,
              "volume-implied one, so a boundary offset explains only part of the volume bias.", ""]
    else:
        L += ["The labels do **not** place the surface outside the mesh, so a boundary offset does",
              "not explain the volume bias. Look for another mechanism.", ""]

    full = 1.5 / 256.0 / 0.9          # depth offset in the normalized frame (1_scale padding 0.1)
    L += ["## 2. Comparison with the documented 3DMSL generation pipeline", "",
          "From github.com/bioailab/3DMSL (`occupancy_networks/scripts/dataset_mito/build.sh`):",
          "each mesh is scaled with mesh-fusion `1_scale.py` (padding 0.1, so the largest extent",
          "becomes 0.9) and fused into a watertight mesh with `2_fusion.py` (defaults: 256³,",
          "`depth_offset_factor` 1.5, which moves every depth map 1.5 voxels toward the camera). Then",
          "`sample_mesh.py --resize --bbox_in_folder 0_in` labels 100,000 uniform points in the ±0.55",
          "box against that watertight mesh (`check_mesh_contains`), with `loc`/`scale` taken from",
          "the original mesh's bounding box. So the labels describe a slightly inflated copy of",
          "`raw_mesh.off`.", "",
          "| Reference | Offset (normalized) | Measured t50 ÷ reference |", "|---|---|---|",
          "| full depth offset along the viewing ray, 1.5/256/0.9 | %.5f | %.2f |" % (full, t50.median() / full),
          "| half of it (mean cos θ = ½ over the visible hemisphere) | %.5f | %.2f |" % (full / 2, t50.median() / (full / 2)), "",
          "The depth offset acts along each camera ray, so the outward shift of the fused surface",
          "along its normal is a fraction of 1.5 voxels that depends on the viewing angles averaged",
          "by the fusion step. A measured offset of the same order (roughly 0.3–1× the full ray",
          "offset) is consistent with the documented pipeline.", "",
          "Query points span %.3f to %.3f in the normalized frame (the pipeline uses ±0.55). Each" % (q_lo, q_hi),
          "archive holds %d query points; the data paper's figure of 10,000 does not match the files." % int(ok.n_query.median()), "",
          "Files: `mechanism_offset_instances.csv`, `mechanism_offset_profile.csv`, `mechanism_offset.png`.", ""]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        mid = (prof.d_lo + prof.d_hi) / 2
        axes[0].plot(mid, prof.p_inside, "o-", color="#2a78d6", ms=4)
        axes[0].axvline(0, color="#0b0b0b", lw=1.2); axes[0].axvline(t50.median(), color="#eb6834", lw=1.6)
        axes[0].axhline(0.5, color="#888888", lw=0.8, ls="--")
        axes[0].set_xlabel("signed distance to raw mesh (normalized; + outside)")
        axes[0].set_ylabel("share of query points labelled inside")
        axes[0].set_title("Label boundary vs mesh surface (median t50 = %.4f)" % t50.median(), fontsize=10, loc="left")
        axes[1].scatter(tv, t50, s=18, color="#2a78d6", alpha=0.8, edgecolors="none")
        lim = [min(tv.min(), t50.min()), max(tv.max(), t50.max())]
        axes[1].plot(lim, lim, color="#888888", lw=1, ls="--")
        axes[1].set_xlabel("offset implied by volume error"); axes[1].set_ylabel("measured label offset t50")
        axes[1].set_title("Per instance (r = %.2f)" % r, fontsize=10, loc="left")
        for a in axes:
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "mechanism_offset.png"), dpi=180); plt.close(fig)
    except Exception as e:
        L += ["(figure skipped: %s)" % e, ""]

    with open(os.path.join(args.outdir, "mechanism_offset_report.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()