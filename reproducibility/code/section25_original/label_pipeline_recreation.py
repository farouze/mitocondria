#!/usr/bin/env python3
"""Controlled test of the 3DMSL label-generation mechanism (plan item 2).

This is a re-implementation in NumPy of the steps the 3DMSL build uses
(bioailab/3DMSL -> occupancy_networks/external/mesh-fusion, read 2026-09-26).
It is not the original compiled code, so it is a CONTROLLED RE-IMPLEMENTATION,
not a reproduction of the historical run.
  1_scale.py : centre on the mesh bounding box, scale so the largest extent is
               1 - padding = 0.9, i.e. grid coordinates g = 0.9 * n, where n are
               the normalized 3DMSL coordinates (loc = bbox centre, scale = max extent).
  2_fusion.py: 100 Fibonacci views (get_points / get_views, copied exactly);
               camera = R @ v + [0, 0, 1]; pinhole f = 640, c = 320, 640 x 640;
               z-near/far 0.25 / 1.75 (background depth = 1.75);
               depth -= depth_offset_factor / 256; 3 x 3 grey erosion;
               TSDF at 256^3, truncation 10 voxels: for each voxel, average of
               clip(depth - z, +-trunc) over views where depth - z >= -trunc,
               -trunc if no view qualifies (libfusioncpu TsdfFusionFunctor);
               marching cubes on -tsdf at level 0.
  sample_mesh.py: the stored query points are labeled inside the fused surface.

Only the depth offset changes between conditions (default 0, 0.75, 1.5 voxels);
the mesh, views, depth maps, erosion, grid and query points are identical, so the
comparison is paired. Offsets are applied to the same rendered depth maps
(erosion commutes with subtracting a constant).

Two label conventions are computed, because the code places marching-cubes
vertices at index/256 - 0.5 while voxel centres are at (index + 0.5)/256 - 0.5,
i.e. a -0.5-voxel shift. Which one the released files follow is decided
empirically by agreement with the released labels ("literal" = with the shift,
"centred" = without).

Endpoints per object and condition (relative to the raw mesh volume):
  occupancy-volume error (%)  = 100 * (p_recreated * box / V_mesh - 1)
  fused-mesh volume error (%)
  agreement with the released labels (all points; points within 0.01 of the surface)
  label-transition midpoint t50 (exact distances, e5_boundary_refit.py fit)
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES, TRUNC_F, NVIEWS, F, C, W, ZFAR, PAD = 256, 10, 100, 640.0, 320.0, 640, 1.75, 0.1


# ------------------------------------------------------------ views (copied)
def get_points(n_views=NVIEWS):
    rnd, pts = 1., []
    offset = 2. / n_views
    inc = math.pi * (3. - math.sqrt(5.))
    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - y ** 2)
        phi = ((i + rnd) % n_views) * inc
        pts.append([math.cos(phi) * r, y, math.sin(phi) * r])
    return np.array(pts)


def get_views(n_views=NVIEWS):
    Rs = []
    for p in get_points(n_views):
        lon = -math.atan2(p[0], p[1]); lat = math.atan2(p[2], math.sqrt(p[0] ** 2 + p[1] ** 2))
        Rx = np.array([[1, 0, 0], [0, math.cos(lat), -math.sin(lat)], [0, math.sin(lat), math.cos(lat)]])
        Ry = np.array([[math.cos(lon), 0, math.sin(lon)], [0, 1, 0], [-math.sin(lon), 0, math.cos(lon)]])
        Rs.append(Ry @ Rx)
    return Rs


# ------------------------------------------------------------ rendering
def render_depth(V, Fc, R):
    """Z-buffer render of camera-space depth (perspective-correct), background = ZFAR.
    Pixel (row v, col u) sees the ray through u = f x/z + c, v = f y/z + c."""
    cam = V @ R.T + np.array([0, 0, 1.0])
    z = cam[:, 2]
    u = F * cam[:, 0] / z + C; v = F * cam[:, 1] / z + C
    tu, tv, tz = u[Fc], v[Fc], z[Fc]                      # (f,3)
    depth = np.full(W * W, ZFAR, dtype=np.float64)
    umin = np.clip(np.ceil(tu.min(1)), 0, W - 1).astype(int); umax = np.clip(np.floor(tu.max(1)), 0, W - 1).astype(int)
    vmin = np.clip(np.ceil(tv.min(1)), 0, W - 1).astype(int); vmax = np.clip(np.floor(tv.max(1)), 0, W - 1).astype(int)
    nu, nv = umax - umin + 1, vmax - vmin + 1
    ok = (nu > 0) & (nv > 0)
    idx = np.where(ok)[0]
    cnt = (nu * nv)[idx]
    # process triangles in batches to bound memory
    order = np.cumsum(cnt)
    start = 0
    B = 4_000_000
    while start < len(idx):
        end = np.searchsorted(order, (order[start - 1] if start else 0) + B, side="right")
        end = max(end, start + 1)
        ti = idx[start:end]; c = cnt[start:end]
        rep = np.repeat(np.arange(len(ti)), c)
        loc = np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c)
        nuu = np.repeat(nu[ti], c)
        pu = np.repeat(umin[ti], c) + loc % nuu
        pv = np.repeat(vmin[ti], c) + loc // nuu
        a = ti[rep]
        x0, x1, x2 = tu[a, 0], tu[a, 1], tu[a, 2]; y0, y1, y2 = tv[a, 0], tv[a, 1], tv[a, 2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        with np.errstate(divide="ignore", invalid="ignore"):
            w0 = ((y1 - y2) * (pu - x2) + (x2 - x1) * (pv - y2)) / den
            w1 = ((y2 - y0) * (pu - x2) + (x0 - x2) * (pv - y2)) / den
        w2 = 1 - w0 - w1
        e = -1e-9
        m = (w0 >= e) & (w1 >= e) & (w2 >= e) & np.isfinite(w0) & (np.abs(den) > 1e-12)
        invz = w0 / tz[a, 0] + w1 / tz[a, 1] + w2 / tz[a, 2]
        zz = 1.0 / invz[m]
        pix = (pv[m] * W + pu[m])
        keep = (zz > 0.25) & (zz < ZFAR)
        np.minimum.at(depth, pix[keep], zz[keep])
        start = end
    return depth.reshape(W, W)


# ------------------------------------------------------------ fusion
def fuse(depths, Rs, offsets, lo_idx, hi_idx):
    """TSDF over the voxel sub-box [lo_idx, hi_idx) (per axis), one volume per offset.
    Voxels outside the sub-box are free space (+trunc)."""
    vs = 1.0 / RES; trunc = TRUNC_F * vs
    ax = [((np.arange(lo_idx[k], hi_idx[k]) + 0.5) * vs - 0.5).astype(np.float32) for k in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    n = len(P)
    sums = np.zeros((len(offsets), n), np.float32); cnts = np.zeros((len(offsets), n), np.int16)
    offs = np.asarray(offsets, np.float32) * vs
    for dm, R in zip(depths, Rs):
        cam = P @ R.T.astype(np.float32) + np.float32([0, 0, 1])
        zc = cam[:, 2]
        u = np.floor(F * cam[:, 0] / zc + C + 0.5).astype(np.int32)
        v = np.floor(F * cam[:, 1] / zc + C + 0.5).astype(np.int32)
        inside = (u >= 0) & (v >= 0) & (u < W) & (v < W)
        d = np.full(n, np.nan, np.float32)
        d[inside] = dm[v[inside], u[inside]]
        base = d - zc
        for k, o in enumerate(offs):
            dist = base - o
            val = inside & (dist >= -trunc)
            sums[k, val] += np.clip(dist[val], -trunc, trunc)
            cnts[k, val] += 1
    shape = tuple(int(h - l) for l, h in zip(lo_idx, hi_idx))
    out = []
    for k in range(len(offsets)):
        t = np.where(cnts[k] > 0, sums[k] / np.maximum(cnts[k], 1), -trunc).reshape(shape)
        full = np.full((RES, RES, RES), trunc, np.float32)
        full[lo_idx[0]:hi_idx[0], lo_idx[1]:hi_idx[1], lo_idx[2]:hi_idx[2]] = t
        out.append(full)
    return out


def fused_mesh_volume(tsdf, shift):
    from skimage.measure import marching_cubes
    import trimesh
    T = np.pad(tsdf, 1, constant_values=1e6)
    verts, faces, _, _ = marching_cubes(-T, 0.0)
    verts = verts - 1
    verts = verts / RES - 0.5 if shift else (verts + 0.5) / RES - 0.5
    m = trimesh.Trimesh(verts, faces[:, ::-1], process=False)
    return abs(float(m.volume)), bool(m.is_watertight), verts, faces[:, ::-1]


def labels_mesh(tsdf, g, shift, verts, faces, band_vox=2.0):
    """Mesh-containment labels against the marching-cubes surface (as sample_mesh.py does).
    Points farther than band_vox voxels from the trilinear zero level cannot change side
    (the two surfaces differ by less than one voxel), so only the band is tested exactly."""
    import e5_boundary_refit as e5
    idx = (g + 0.5) * RES - 0.5 + (0.5 if shift else 0.0)
    val = ndimage.map_coordinates(tsdf, idx.T, order=1, mode="nearest")
    lab = val < 0
    band = np.where(np.abs(val) < band_vox / RES)[0]
    if len(band):
        lab[band] = e5.winding_number(g[band], verts, faces) > 0.5
    return lab, int(len(band))


def labels(tsdf, g, shift):
    """inside = trilinear TSDF < 0 at grid coordinates g (literal code: mesh shifted by -0.5 voxel)."""
    idx = (g + 0.5) * RES - 0.5 + (0.5 if shift else 0.0)
    val = ndimage.map_coordinates(tsdf, idx.T, order=1, mode="nearest")
    return val < 0


# ------------------------------------------------------------ per object
def run_object(d, offsets, do_t50=True, mesh_check=()):
    import e5_boundary_refit as e5
    from scipy.spatial import cKDTree
    t0 = time.time()
    mesh, pts, occ_rel, loc, s = e5.load_instance(d)
    Vn = (np.asarray(mesh.vertices, float) - loc) / s
    Fc = np.asarray(mesh.faces, int)
    Vg = 0.9 * Vn                                                   # grid frame
    Rs = get_views()
    depths = [ndimage.grey_erosion(render_depth(Vg, Fc, R), size=(3, 3)) for R in Rs]
    t_render = time.time() - t0
    margin = TRUNC_F + 4
    lo = np.clip(np.floor((Vg.min(0) + 0.5) * RES).astype(int) - margin, 0, RES)
    hi = np.clip(np.ceil((Vg.max(0) + 0.5) * RES).astype(int) + margin, 0, RES)
    tsdfs = fuse(depths, Rs, offsets, lo, hi)
    t_fuse = time.time() - t0 - t_render

    V_m = abs(float(mesh.volume)) / s ** 3                          # normalized units
    box = np.prod(pts.max(0) - pts.min(0))
    g = 0.9 * pts
    # distances for agreement near the surface and for t50 (same selection as the refit)
    tree = cKDTree(Vn[Fc].mean(1))
    dv, _ = cKDTree(Vn).query(pts)
    edges = np.linalg.norm(Vn[Fc[:, [1, 2, 0]]] - Vn[Fc], axis=-1)
    pre = np.where(dv < 0.02 + edges.max())[0]
    ua = e5.approx_unsigned(pts[pre], Vn, Fc, tree, 32)
    near = pre[ua < 0.02]
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(near, 3000, replace=False)) if len(near) > 3000 else near
    sign = np.where(e5.winding_number(pts[sel], Vn, Fc) > 0.5, -1.0, 1.0)
    dsel = sign * e5.exact_unsigned_pruned(pts[sel], Vn, Fc, tree, e5.approx_unsigned(pts[sel], Vn, Fc, tree, 32))
    near01 = sel[np.abs(dsel) < 0.01]

    rows = []
    released = {"p": float(occ_rel.mean())}
    released["occ_err_pct"] = 100 * (released["p"] * box / V_m - 1)
    if do_t50:
        released["t50"] = e5.fit_logistic(dsel, occ_rel[sel], 0.02)["t50"]
    for off, T in zip(offsets, tsdfs):
        for shift in (True, False):
            lab = labels(T, g, shift)
            vf, wt, mv, mf = fused_mesh_volume(T, shift)
            r = {"offset_vox": off, "convention": "literal" if shift else "centred",
                 "p": float(lab.mean()), "occ_err_pct": 100 * (lab.mean() * box / V_m - 1),
                 "fused_mesh_err_pct": 100 * (vf / 0.9 ** 3 / V_m - 1), "fused_watertight": wt,
                 "agree_all": float(np.mean(lab == occ_rel)),
                 "agree_near_0.01": float(np.mean(lab[near01] == occ_rel[near01])) if len(near01) else np.nan}
            if off in mesh_check and not shift:          # centred convention only
                labm, nb = labels_mesh(T, g, shift, mv, mf)
                r.update({"mesh_occ_err_pct": 100 * (labm.mean() * box / V_m - 1), "mesh_band_points": nb,
                          "mesh_vs_trilinear_agree_all": float(np.mean(labm == lab)),
                          "mesh_vs_trilinear_agree_near": float(np.mean(labm[near01] == lab[near01])) if len(near01) else np.nan,
                          "mesh_agree_near_0.01": float(np.mean(labm[near01] == occ_rel[near01])) if len(near01) else np.nan})
            if do_t50:
                fit = e5.fit_logistic(dsel, lab[sel], 0.02)
                r.update({"t50": fit["t50"], "w": fit["w"], "t50_converged": fit["converged"]})
            rows.append(r)
    return rows, released, {"render_s": t_render, "fuse_s": t_fuse, "total_s": time.time() - t0,
                            "n_faces": int(len(Fc)), "grid_box": [lo.tolist(), hi.tolist()]}


def paired_summary(df, conv, a, b, col, n_boot=5000, seed=0):
    x = df[(df.convention == conv) & (df.offset_vox == a)].set_index("id")[col]
    y = df[(df.convention == conv) & (df.offset_vox == b)].set_index("id")[col]
    d = (x - y).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(d, len(d))) for _ in range(n_boot)] if len(d) > 1 else [np.nan]
    return {"n": int(len(d)), "mean": float(np.mean(d)), "median": float(np.median(d)), "sd": float(np.std(d, ddof=1)) if len(d) > 1 else np.nan,
            "ci95_mean": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_positive": int(np.sum(d > 0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--ids-csv", required=True, help="CSV with the 60 E5 IDs (column instance_id or id)")
    ap.add_argument("--n", type=int, default=0, help="use only the first n IDs (pilot); 0 = all")
    ap.add_argument("--offsets", default="0,0.75,1.5")
    ap.add_argument("--no-t50", action="store_true")
    ap.add_argument("--mesh-check-offsets", default="", help="e.g. 0,1.5: also label by mesh containment (centred convention)")
    ap.add_argument("--skip-first", type=int, default=0, help="skip the first k IDs (e.g. the 5 pilot objects)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import e5_boundary_refit as e5
    offsets = [float(x) for x in a.offsets.split(",")]
    ids_df = pd.read_csv(a.ids_csv, dtype=str)
    idc = next(c for c in ids_df.columns if c.lower() in ("instance_id", "id", "object_id", "instance"))
    ids = list(ids_df[idc])[a.skip_first:]
    ids = ids[: a.n or None]
    mesh_check = tuple(float(x) for x in a.mesh_check_offsets.split(",") if x.strip())
    dirs = e5.find_instance_dirs(a.shard_root)
    out_csv = os.path.join(a.out, "recreation_per_condition.csv")
    rel_csv = os.path.join(a.out, "released_per_object.csv")
    done = pd.read_csv(out_csv, dtype={"id": str}) if os.path.exists(out_csv) else pd.DataFrame()
    rel = pd.read_csv(rel_csv, dtype={"id": str}) if os.path.exists(rel_csv) else pd.DataFrame()
    rows, rrows = done.to_dict("records"), rel.to_dict("records")
    have = set(done["id"]) if len(done) else set()
    json.dump({"protocol": "controlled re-implementation of the 3DMSL mesh-fusion label step",
               "offsets_vox": offsets, "ids": ids, "resolution": RES, "truncation_vox": TRUNC_F,
               "n_views": NVIEWS, "image": W, "focal": F, "erosion": "3x3 grey", "started_utc": datetime.now(timezone.utc).isoformat(),
               "primary_endpoint": "paired change in occupancy-volume error, offset 1.5 minus offset 0 (convention chosen by agreement with released labels)"},
              open(os.path.join(a.out, "protocol.json"), "w"), indent=1)
    for j, i in enumerate(ids):
        if i in have:
            continue
        if i not in dirs:
            print(f"missing {i}"); continue
        try:
            rs, relv, timing = run_object(dirs[i], offsets, not a.no_t50, mesh_check)
        except Exception as ex:
            print(f"[{j + 1}/{len(ids)}] {i}: ERROR {ex!r}"); rows.append({"id": i, "error": repr(ex)}); continue
        for r in rs:
            r["id"] = i
        rows += rs
        relv["id"] = i; relv.update(timing); rrows.append(relv)
        pd.DataFrame(rows).to_csv(out_csv, index=False); pd.DataFrame(rrows).to_csv(rel_csv, index=False)
        lit = {r["offset_vox"]: r for r in rs if r["convention"] == "literal"}
        cen = {r["offset_vox"]: r for r in rs if r["convention"] == "centred"}
        print(f"[{j + 1}/{len(ids)}] {i}: released err {relv['occ_err_pct']:+.2f}% | recreated err "
              + " ".join(f"{o}:{cen[o]['occ_err_pct']:+.2f}%" for o in offsets)
              + f" | agree@1.5 literal {lit[1.5]['agree_near_0.01']:.3f} centred {cen[1.5]['agree_near_0.01']:.3f}"
              f" ({timing['total_s']:.0f}s)")

    df = pd.DataFrame(rows).dropna(subset=["offset_vox"])
    rel = pd.DataFrame(rrows)
    if df.empty:
        sys.exit("no results")
    m15 = df[df.offset_vox == 1.5].groupby("convention")["agree_near_0.01"].median()
    best = m15.idxmax()
    S = {"n_objects": int(df["id"].nunique()), "convention_agreement_near_surface_at_1.5": m15.to_dict(),
         "convention_used": best}
    for conv in ("literal", "centred"):
        sub = df[df.convention == conv]
        S[conv] = {
            "by_offset": {str(o): {"occ_err_median": float(g["occ_err_pct"].median()), "occ_err_mean": float(g["occ_err_pct"].mean()),
                                   "fused_mesh_err_median": float(g["fused_mesh_err_pct"].median()),
                                   "agree_all_median": float(g["agree_all"].median()),
                                   "agree_near_median": float(g["agree_near_0.01"].median()),
                                   "t50_median": float(g["t50"].median()) if "t50" in g else None,
                                   "n_watertight": int(g["fused_watertight"].sum())}
                          for o, g in sub.groupby("offset_vox")},
            "primary_paired_occ_err_1.5_minus_0": paired_summary(df, conv, 1.5, 0.0, "occ_err_pct"),
            "paired_fused_mesh_err_1.5_minus_0": paired_summary(df, conv, 1.5, 0.0, "fused_mesh_err_pct"),
        }
        if "t50" in df:
            S[conv]["paired_t50_1.5_minus_0"] = paired_summary(df, conv, 1.5, 0.0, "t50")
    r15 = df[(df.convention == best) & (df.offset_vox == 1.5)].set_index("id")
    rr = rel.set_index("id")
    common = r15.index.intersection(rr.index)
    S["recreated_1.5_vs_released"] = {
        "occ_err_released_median": float(rr.loc[common, "occ_err_pct"].median()),
        "occ_err_recreated_median": float(r15.loc[common, "occ_err_pct"].median()),
        "median_abs_diff_pp": float((r15.loc[common, "occ_err_pct"] - rr.loc[common, "occ_err_pct"]).abs().median()),
        "t50_released_median": float(rr.loc[common, "t50"].median()) if "t50" in rr else None,
        "t50_recreated_median": float(r15.loc[common, "t50"].median()) if "t50" in r15 else None}
    json.dump(S, open(os.path.join(a.out, "recreation_summary.json"), "w"), indent=1, default=float)

    # figure: paired dose-response
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    sub = df[df.convention == best]
    fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
    for i, gi in sub.groupby("id"):
        gi = gi.sort_values("offset_vox")
        ax.plot(gi.offset_vox, gi.occ_err_pct, color="#28648a", alpha=0.25, lw=0.7)
    med = sub.groupby("offset_vox")["occ_err_pct"].median()
    ax.plot(med.index, med.values, "o-", color="#bd632d", lw=1.6, label="median")
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    if len(common):
        ax.scatter([1.5], [rr.loc[common, "occ_err_pct"].median()], marker="x", color="black", zorder=5, label="released labels (median)")
    ax.set_xticks(offsets); ax.set_xlabel("Depth offset (voxels at 256$^3$)")
    ax.set_ylabel("Occupancy volume error vs mesh (%)"); ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(a.out, "offset_dose_response.pdf")); fig.savefig(os.path.join(a.out, "offset_dose_response.png"), dpi=300)

    c = S[best]; p = c["primary_paired_occ_err_1.5_minus_0"]
    print("\n=== Label-pipeline recreation ===")
    print(f"objects: {S['n_objects']}; convention chosen by agreement with released labels: {best} {m15.to_dict()}")
    for o, v in c["by_offset"].items():
        print(f"  offset {o}: occupancy error median {v['occ_err_median']:+.3f}% (mean {v['occ_err_mean']:+.3f}%), "
              f"fused-mesh error {v['fused_mesh_err_median']:+.3f}%, t50 {v['t50_median']}, agreement near surface {v['agree_near_median']:.4f}")
    print(f"  PRIMARY paired change 1.5 - 0: mean {p['mean']:+.3f} pp (95% CI {p['ci95_mean'][0]:+.3f} to {p['ci95_mean'][1]:+.3f}), "
          f"median {p['median']:+.3f}, positive in {p['n_positive']}/{p['n']}")
    print(f"  recreated (1.5) vs released: {S['recreated_1.5_vs_released']}")


if __name__ == "__main__":
    main()
