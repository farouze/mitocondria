#!/usr/bin/env python3
"""
imaging_v2.py - rebuilt imaging-condition analysis for the 3DMSL renders.

Why the first version failed (imaging_condition_reliability.py):
  * The three configurations use different pixel sizes (Conf1 70 nm, Conf2 48 nm,
    Epi1 109 nm; 3DMSL data paper), so "foreground fraction of the image" measures
    a different physical area in each one. Absolute-agreement ICC across them could
    not work.
  * The renders label the mitochondrial SURFACE. A confocal optical section through
    a surface label is a ring, so Otsu picks out membrane pixels, not the object.
  * Nothing was compared with the known geometry, so real orientation effects
    (an elongated object looks different end-on) could not be told apart from
    measurement error.

What this version does:
  1. Records bit depth, image size and value range of every render.
  2. Segments each view by thresholding at half the peak above a border-estimated
     background, closing and filling holes, and keeping the largest component
     (primary, fixed before looking at results). Otsu is kept for comparison.
  3. Converts every size measure to physical units with each configuration's pixel size.
     It also computes a threshold-free size: the intensity-weighted radius of gyration.
  4. Computes geometric targets from the mesh for every view, following the 3DMSL
     simulator (github.com/bioailab/3DMSL, occupancy_networks/scripts/dataset_mito/
     simulate_img.py): mesh units are 24 nm; emitters are centred on their centroid and
     rotated with the simulator's own rotate(); the optical axis is the third coordinate;
     and emitters below the centroid plane fall outside the PSF lookup (map_coordinates
     with zero fill for negative depth), so only the upper half is imaged. The targets
     are the silhouette of the upper half, the silhouette of the whole object, and
     cross-sections at several heights above the centroid. The focal depth depends on the
     Gibson-Lanni model with the stage 1 um toward the objective, so it is identified
     from the data (exploratory).
  5. Reports validity (image measure vs geometric target), orientation effects (how
     much of the across-view variation the geometry explains), and cross-configuration
     agreement in physical units (absolute and consistency ICC).

  python3 imaging_v2.py --root /path/to/shard --outdir out --sample 300
"""
import argparse
import os
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from hit_common import IMG_CONFIGS, find_instance_dirs, images_for_config, instance_id, load_mesh, otsu_threshold

PIXEL_UM = {"Conf1": 0.070, "Conf2": 0.0488, "Epi1": 0.109}     # data paper table; Conf2 from repo conf2.py
# (the repo's only widefield config, epi2.py, uses 80 nm pixels; the paper lists 109 nm for Epi1.
#  The geometry ratio in section 2 of the report checks which one fits.)
MODALITY = {"Conf1": "confocal", "Conf2": "confocal", "Epi1": "widefield"}
VIEWS = ("0", "x90", "x180", "y90", "y180", "z90")
HEIGHTS_UM = (0.0, 0.1, 0.2, 0.3, 0.5)    # candidate focal heights above the emitter centroid
BORDER = 6                   # pixels used to estimate background
AXES = np.eye(3)


def sim_rotate(X, theta, axis):
    """Exact copy of rotate() in 3DMSL simulate_img.py (row vectors times matrix)."""
    c, s_ = np.cos(theta), np.sin(theta)
    M = {"x": np.array([[1., 0, 0], [0, c, -s_], [0, s_, c]]),
         "y": np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]]),
         "z": np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.]])}[axis]
    return np.dot(X, M)


def view_transform(X, view):
    if view == "0":
        return X.copy()
    return sim_rotate(X, np.deg2rad(float(view[1:])), view[0])


def view_name(path):
    m = re.search(r"_(0|x90|x180|y90|y180|z90)$", os.path.splitext(os.path.basename(path))[0])
    return m.group(1) if m else None


# ------------------------------------------------------------------ geometry targets
def _raster_area(uv, cell):
    """Area of the region covered by dense 2-D points, via a filled occupancy grid."""
    if len(uv) < 3:
        return 0.0
    lo = uv.min(axis=0) - 3 * cell
    ij = np.floor((uv - lo) / cell).astype(int)
    shape = ij.max(axis=0) + 4
    g = np.zeros(shape, bool)
    g[ij[:, 0], ij[:, 1]] = True
    g = ndi.binary_closing(g, structure=np.ones((3, 3)), iterations=2)
    g = ndi.binary_fill_holes(g)
    return float(g.sum()) * cell * cell


def _rg(uv):
    return float(np.sqrt(((uv - uv.mean(0)) ** 2).sum(1).mean())) if len(uv) > 2 else np.nan


def mesh_targets(mesh, unit_um, n_points=120000, seed=0):
    """Per view: geometric targets in um / um^2, following the simulator's geometry."""
    import trimesh
    pts, _ = trimesh.sample.sample_surface(mesh, n_points, seed=seed)
    pts = np.asarray(pts, float)
    centre = pts.mean(axis=0)                           # emitter centroid (area-weighted)
    P0 = (pts - centre) * unit_um
    V0 = (np.asarray(mesh.vertices, float) - centre) * unit_um
    faces = np.asarray(mesh.faces)
    cell = float(np.ptp(P0, axis=0).max()) / 160.0
    out = {}
    for v in VIEWS:
        P = view_transform(P0, v)
        up = P[P[:, 2] >= 0]
        t = {"sil_upper": _raster_area(up[:, :2], cell), "sil_all": _raster_area(P[:, :2], cell),
             "rg_upper": _rg(up[:, :2]), "rg_all": _rg(P[:, :2])}
        rm = trimesh.Trimesh(vertices=view_transform(V0, v), faces=faces, process=False)
        for h in HEIGHTS_UM:
            key = "sec_%03dnm" % round(1000 * h)
            try:
                seg = trimesh.intersections.mesh_plane(rm, plane_normal=[0, 0, 1], plane_origin=[0, 0, h])
                if len(seg) == 0:
                    t[key] = 0.0
                else:
                    seg = np.asarray(seg, float)
                    tt = np.linspace(0, 1, 12)[None, :, None]
                    line = (seg[:, :1, :] * (1 - tt) + seg[:, 1:2, :] * tt).reshape(-1, 3)[:, :2]
                    t[key] = _raster_area(line, cell)
            except Exception:
                t[key] = np.nan
        out[v] = t
    return out


# ----------------------------------------------------------------------- image side
def read_render(path):
    im = Image.open(path)
    arr = np.asarray(im)
    info = dict(mode=im.mode, dtype=str(arr.dtype), h=arr.shape[0], w=arr.shape[1],
                vmin=float(arr.min()), vmax=float(arr.max()))
    a = arr.astype(np.float64)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a, info


def segment(a):
    """Primary mask: half-peak above border background, closed, hole-filled, largest component."""
    border = np.concatenate([a[:BORDER].ravel(), a[-BORDER:].ravel(), a[:, :BORDER].ravel(), a[:, -BORDER:].ravel()])
    bg = float(np.median(border))
    sig = float(1.4826 * np.median(np.abs(border - bg))) or float(border.std()) or 1e-9
    peak = float(np.percentile(a, 99.8))
    thr = bg + max(0.5 * (peak - bg), 3 * sig)
    m = a > thr
    m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=2)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m, bg, sig, peak


def mask_stats(m, px):
    if not m.any():
        return dict(area_um2=0.0, major_um=np.nan, minor_um=np.nan, touches_border=False, mask_empty=True)
    ys, xs = np.nonzero(m)
    cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1].clip(min=0)
    return dict(area_um2=float(m.sum()) * px * px,
                major_um=4 * np.sqrt(ev[0]) * px, minor_um=4 * np.sqrt(ev[1]) * px,
                touches_border=bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()),
                mask_empty=False)


def one_view(path, config):
    a, info = read_render(path)
    px = PIXEL_UM[config]
    m, bg, sig, peak = segment(a)
    st = mask_stats(m, px)
    # threshold-free size: intensity-weighted radius of gyration, restricted to a generous
    # neighbourhood of the object so background noise across the frame does not dominate
    near = ndi.binary_dilation(m, iterations=4) if m.any() else np.zeros_like(m)
    w = np.where(near, np.clip(a - bg - 2 * sig, 0, None), 0.0)
    if w.sum() > 0:
        yy, xx = np.indices(a.shape)
        cy, cx = (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()
        rg = float(np.sqrt((w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / w.sum())) * px
    else:
        rg = np.nan
    lo, hi = a.min(), a.max()
    a8 = np.zeros_like(a, np.uint8) if hi == lo else ((a - lo) / (hi - lo) * 255).astype(np.uint8)
    otsu_m = a8 > otsu_threshold(a8)
    row = dict(config=config, view=view_name(path), file=os.path.basename(path), pixel_um=px,
               bg=bg, bg_sigma=sig, peak=peak, snr=(peak - bg) / sig if sig else np.nan,
               area_um2=st["area_um2"], major_um=st["major_um"], minor_um=st["minor_um"],
               touches_border=st["touches_border"], mask_empty=st["mask_empty"], rg_um=rg,
               otsu_area_um2=float(otsu_m.sum()) * px * px, otsu_fraction=float(otsu_m.mean()))
    row.update({"img_" + k: v for k, v in info.items()})
    return row, m, otsu_m, a


def one_instance(d, n_points, unit_um):
    iid = instance_id(d)
    mesh = load_mesh(d)
    tg = mesh_targets(mesh, unit_um, n_points=n_points, seed=zlib.crc32(iid.encode()))
    rows = []
    for c in IMG_CONFIGS:
        for f in images_for_config(d, c):
            row, *_ = one_view(f, c)
            row["instance_id"] = iid
            if row["view"] in tg:
                row.update({"target_" + k: val for k, val in tg[row["view"]].items()})
            rows.append(row)
    return rows


def safe(d, n_points, unit_um):
    try:
        return one_instance(d, n_points, unit_um)
    except Exception as e:
        return [dict(instance_id=instance_id(d), error="%s: %s" % (type(e).__name__, e))]


# ------------------------------------------------------------------------ statistics
def icc(M, kind="A"):
    """ICC(A,1) absolute agreement or ICC(C,1) consistency, two-way model."""
    M = np.asarray(M, float); n, k = M.shape
    if n < 3 or k < 2:
        return np.nan
    g = M.mean(); rm = M.mean(1); cm = M.mean(0)
    MSR = k * ((rm - g) ** 2).sum() / (n - 1)
    MSC = n * ((cm - g) ** 2).sum() / (k - 1)
    MSE = ((M - rm[:, None] - cm[None, :] + g) ** 2).sum() / ((n - 1) * (k - 1))
    den = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n if kind == "A" else 0.0)
    return float((MSR - MSE) / den) if den else np.nan


def spearman(x, y):
    x = pd.Series(x).rank().to_numpy(); y = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else np.nan


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else np.nan


def overlay_figure(dirs, outpath, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dirs = dirs[:n]
    fig, axes = plt.subplots(len(dirs), 3, figsize=(7.5, 2.5 * len(dirs)))
    axes = np.atleast_2d(axes)
    for r, d in enumerate(dirs):
        for c_i, c in enumerate(IMG_CONFIGS):
            ax = axes[r, c_i]; ax.axis("off")
            files = [f for f in images_for_config(d, c) if view_name(f) == "0"] or images_for_config(d, c)
            if not files:
                continue
            _, m, om, a = one_view(files[0], c)
            ax.imshow(a, cmap="gray")
            ax.contour(m, levels=[0.5], colors="#eb6834", linewidths=1.0)
            ax.contour(om, levels=[0.5], colors="#2a78d6", linewidths=0.6, linestyles="dotted")
            if r == 0:
                ax.set_title("%s (%.0f nm px)" % (c, 1000 * PIXEL_UM[c]), fontsize=9)
        axes[r, 0].text(-4, 10, instance_id(d), fontsize=8, ha="right")
    fig.suptitle("orange = primary mask (half-peak, filled); blue dotted = Otsu (old)", fontsize=9)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=790)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mesh-unit-um", type=float, default=0.024,
                    help="physical size of one mesh unit (24 nm: roi_to_mesh.py meshes the 24 nm stack "
                         "and simulate_img.py multiplies by resolution=24 nm)")
    ap.add_argument("--n-surface-points", type=int, default=120000)
    ap.add_argument("--epi-pixel-um", type=float, default=None,
                    help="override the Epi1 pixel size (paper 0.109 um; repo epi2.py 0.080 um)")
    args = ap.parse_args()
    if args.epi_pixel_um:
        PIXEL_UM["Epi1"] = args.epi_pixel_um      # inherited by forked worker processes
    os.makedirs(args.outdir, exist_ok=True)

    dirs = find_instance_dirs(args.root)
    if args.sample and args.sample < len(dirs):
        pick = np.random.default_rng(args.seed).choice(len(dirs), args.sample, replace=False)
        dirs = [dirs[i] for i in sorted(pick)]
    print("imaging_v2 on %d instances" % len(dirs), flush=True)
    t0, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(safe, d, args.n_surface_points, args.mesh_unit_um) for d in dirs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.extend(f.result())
            if i % 25 == 0 or i == len(futs):
                print("  %d/%d (%.1fs)" % (i, len(futs), time.time() - t0), flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(args.outdir, "imaging_v2_views.csv"), index=False)
    n_err = int(raw["error"].notna().sum()) if "error" in raw else 0
    v = raw.dropna(subset=["config", "view"]).copy() if "config" in raw else raw.iloc[:0]
    if v.empty:
        raise RuntimeError("No views were analyzed; see imaging_v2_views.csv")
    usable = v[~v.touches_border.astype(bool) & ~v.mask_empty.astype(bool)].copy()

    L = ["# Imaging-Condition Reliability, Rebuilt (n=%d instances)" % v.instance_id.nunique(), "",
         "Replaces `imaging_condition_report.md`. Sizes are in physical units (pixel sizes from the",
         "3DMSL data paper: Conf1 70 nm, Conf2 48 nm, Epi1 109 nm), and every image measure is checked",
         "against the matching geometric quantity computed from the mesh for the same view.", ""]
    if n_err:
        L += ["%d instance(s) failed and were skipped (see the `error` column)." % n_err, ""]

    # 1. render diagnostics
    L += ["## 1. What the renders are", "", "| Config | mode / dtype | size (px) | field of view | value range | median SNR |",
          "|---|---|---|---|---|---|"]
    for c in IMG_CONFIGS:
        s = v[v.config == c]
        if s.empty:
            continue
        L.append("| %s | %s / %s | %d × %d | %.1f µm | %.0f – %.0f | %.1f |"
                 % (c, s.img_mode.mode().iat[0], s.img_dtype.mode().iat[0], s.img_h.median(), s.img_w.median(),
                    s.img_w.median() * PIXEL_UM[c], s.img_vmin.min(), s.img_vmax.max(), s.snr.median()))
        if s.snr.median() > 1e4:
            L[-1] = L[-1].rsplit("|", 2)[0] + "| noise-free |"
    L += ["", "| Config | views | empty mask | object cut by image edge | used |", "|---|---|---|---|---|"]
    for c in IMG_CONFIGS:
        s = v[v.config == c]
        if s.empty:
            continue
        L.append("| %s | %d | %.1f%% | %.1f%% | %d |" % (c, len(s), 100 * s.mask_empty.astype(bool).mean(),
                                                     100 * s.touches_border.astype(bool).mean(),
                                                     int((usable.config == c).sum())))
    L += ["", "Views where the object touches the image edge are excluded from size measures",
          "(part of the object is outside the field of view).", ""]

    # 2. validity against geometry
    tkeys = ["sil_upper", "sil_all"] + ["sec_%03dnm" % round(1000 * h) for h in HEIGHTS_UM]
    tlabel = {"sil_upper": "silhouette, upper half", "sil_all": "silhouette, whole object"}
    tlabel.update({"sec_%03dnm" % round(1000 * h): "slice at +%.0f nm" % (1000 * h) for h in HEIGHTS_UM})
    L += ["## 2. Does the image measure track the true geometry?", "",
          "Targets are computed per view with the simulator's own rotation and centring. Only the",
          "upper half of each object reaches the image (emitters below the centroid plane fall outside",
          "the PSF lookup), and the focal height is not stated, so every candidate target is shown.",
          "Spearman correlation between the filled image area and each target:", "",
          "| Config | " + " | ".join(tlabel[k] for k in tkeys) + " | Otsu area (old) vs best |",
          "|---|" + "---|" * (len(tkeys) + 1)]
    best = {}
    for c in IMG_CONFIGS:
        s = usable[usable.config == c]
        if len(s) < 10 or not all("target_" + k in s for k in tkeys):
            continue
        rs = [spearman(s.area_um2, s["target_" + k]) for k in tkeys]
        b = tkeys[int(np.nanargmax(rs))]; best[c] = b
        ro = spearman(s.otsu_area_um2, s["target_" + b])
        L.append("| %s | " % c + " | ".join("%.3f" % r for r in rs) + " | %.3f |" % ro)
    ratios, rgs = {}, {}
    for c, b in best.items():
        s = usable[usable.config == c]
        rg_key = "target_rg_upper" if b != "sil_all" else "target_rg_all"
        ok = s["target_" + b] > 0
        ratios[c] = float((s.area_um2[ok] / s["target_" + b][ok]).median())
        rgs[c] = spearman(s.rg_um, s[rg_key])
    conf_ref = [ratios[c] for c in ratios if MODALITY[c] == "confocal"]
    ref = float(np.mean(conf_ref)) if conf_ref else np.nan
    L += ["", "| Config | best-matching target | Spearman, radius of gyration vs target Rg | median image ÷ geometric area | pixel size that would give the confocal ratio |",
          "|---|---|---|---|---|"]
    for c, b in best.items():
        imp = 1000 * PIXEL_UM[c] * np.sqrt(ref / ratios[c]) if ratios[c] > 0 and np.isfinite(ref) else np.nan
        L.append("| %s | %s | %.3f | %.2f | %.0f nm (used: %.0f nm) |" % (c, tlabel[b], rgs[c], ratios[c], imp, 1000 * PIXEL_UM[c]))
    L += ["", "The best-matching target is chosen from the data, so treat it as an exploratory",
          "comparison of candidate geometric targets. Different optical configurations need not share an area ratio.",
          "The implied pixel size above matches the confocal ratio; matching unit geometric agreement uses pixel_size/sqrt(area_ratio).",
          "If Epi1's implied pixel size is far from the 109 nm used (the repo's epi2.py uses 80 nm),",
          "verify the actual simulator configuration before interpreting this discrepancy as a calibration error.", ""]

    # 3. orientation: geometry vs measurement noise
    L += ["## 3. Orientation effects: real geometry or measurement error?", "",
          "Within each instance, the six views are compared on log area against the best-matching",
          "target. *Geometry share* is the squared correlation between the image's and the geometry's",
          "view-to-view deviations; the residual SD is the variation the geometry does not explain.", "",
          "| Config | median CV, image | median CV, geometry | geometry share (R²) | residual SD (log) |",
          "|---|---|---|---|---|"]
    orient = {}
    for c, b in best.items():
        s = usable[usable.config == c].copy()
        s["tg"] = s["target_" + b]
        s = s[(s.tg > 0) & (s.area_um2 > 0)]
        g = s.groupby("instance_id")
        cv_img = (g.area_um2.std() / g.area_um2.mean()).median()
        cv_geo = (g.tg.std() / g.tg.mean()).median()
        li, lt = np.log(s.area_um2), np.log(s.tg)
        di = li - li.groupby(s.instance_id).transform("mean")
        dt = lt - lt.groupby(s.instance_id).transform("mean")
        r2 = pearson(di, dt) ** 2
        slope = float((di * dt).sum() / (dt * dt).sum()) if (dt * dt).sum() else np.nan
        resid = float(np.sqrt(((di - slope * dt) ** 2).mean()))
        orient[c] = (cv_img, cv_geo, r2, resid)
        L.append("| %s | %.3f | %.3f | %.2f | %.3f |" % (c, cv_img, cv_geo, r2, resid))
    L += [""]
    for c, (ci, cg, r2, rs) in orient.items():
        if r2 >= 0.5:
            L.append("- %s: most of the orientation variation is real geometry (R² %.2f): the image"
                     " follows how the visible part of the object changes with viewing angle." % (c, r2))
        else:
            L.append("- %s: geometry explains only %.0f%% of the orientation variation; the rest is"
                     " measurement error from rendering/segmentation." % (c, 100 * r2))
    L += [""]

    # 4. cross-configuration agreement in physical units
    L += ["## 4. Agreement across configurations (physical units, mean over usable views)", "",
          "| Pair | n | ICC(A,1) absolute | ICC(C,1) consistency | Spearman | median ratio |",
          "|---|---|---|---|---|---|"]
    inst = usable.groupby(["instance_id", "config"]).area_um2.mean().unstack()
    for a_, b_ in (("Conf1", "Conf2"), ("Conf1", "Epi1"), ("Conf2", "Epi1")):
        if a_ in inst and b_ in inst:
            w = inst[[a_, b_]].dropna()
            if len(w) >= 3:
                L.append("| %s vs %s | %d | %.3f | %.3f | %.3f | %.2f |"
                         % (a_, b_, len(w), icc(w.values, "A"), icc(w.values, "C"),
                            spearman(w[a_], w[b_]), (w[b_] / w[a_]).median()))
    L += ["", "Conf1 and Conf2 are both confocal sections and can be compared directly. Epi1 is a",
          "widefield image (a blurred silhouette), which is a different quantity from a section, so",
          "for Epi1 read the consistency ICC and Spearman; absolute agreement is not expected.", ""]
    L += ["## Interpretation guardrails", "",
          "- These are simulated renders from the same meshes; they test how imaging configuration and",
          "  viewing angle change image-based morphometry, not biological variation.",
          "- Geometry follows the public simulator code (rotation, centring, upper-half visibility,",
          "  24 nm mesh unit); the focal height is identified from the data and the released images",
          "  may have been made with a slightly different code version. Confirm with the 3DMSL authors.",
          "- Conf1 files are the Poisson-noise versions (N_*); Conf2 and Epi1 files are noise-free and",
          "  normalized to a maximum of 255 per image, so intensities are not physical in any config.",
          "- The primary segmentation rule was fixed before looking at results; Otsu is shown only to",
          "  explain the first analysis.", ""]

    try:
        overlay_figure(dirs, os.path.join(args.outdir, "imaging_v2_overlays.png"))
        L += ["Figure: `imaging_v2_overlays.png` (masks on the 0° view of the first 8 instances).", ""]
    except Exception as e:
        L += ["(overlay figure skipped: %s)" % e, ""]
    with open(os.path.join(args.outdir, "imaging_v2_report.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()