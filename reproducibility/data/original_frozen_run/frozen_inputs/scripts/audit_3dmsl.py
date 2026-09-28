#!/usr/bin/env python3
"""
audit_3dmsl.py - Weeks 1-2 milestone: load every 3DMSL instance in a shard across all
four representations (mesh / point cloud / occupancy / rendered images) and write one
row of structural descriptors per instance.

  python3 audit_3dmsl.py --root /path/to/10_24553_27272 --out audit_full_descriptors.csv
  python3 audit_3dmsl.py --root ... --sample 15 --out audit_sample.csv
  python3 audit_3dmsl.py --root ... --start 24553 --end 24852 --workers 4
"""
import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from hit_common import (IMG_CONFIGS, count_images, elongation, find_instance_dirs,
                        instance_id, load_mesh, load_occupancy, load_pointcloud,
                        load_resume_table, pct_diff, raw_extent, sphericity, trimmed_extent,
                        unnormalize, write_checkpoint)


def audit_instance(d):
    row = {"instance_id": instance_id(d), "path": d,
           "mesh_ok": False, "pc_ok": False, "occ_ok": False, "error": ""}
    loc = np.zeros(3)
    scale = np.ones(1)

    # ---- mesh (real units, ground truth) ------------------------------------
    try:
        m = load_mesh(d)
        ext = np.asarray(m.extents, dtype=np.float64)
        vol = float(abs(m.volume))
        area = float(m.area)
        row.update(mesh_ok=True,
                   mesh_watertight=bool(m.is_watertight),
                   mesh_n_vertices=int(len(m.vertices)),
                   mesh_n_faces=int(len(m.faces)),
                   mesh_volume=vol,
                   mesh_area=area,
                   mesh_sphericity=sphericity(vol, area),
                   mesh_extent_x=ext[0], mesh_extent_y=ext[1], mesh_extent_z=ext[2],
                   mesh_elongation=elongation(ext))
        mesh_center = np.asarray(m.bounds, dtype=np.float64).mean(axis=0)
        mesh_extent = ext
    except Exception as e:
        row["error"] += "mesh:%s; " % e
        mesh_center, mesh_extent = None, None

    # ---- occupancy (normalized frame + the loc/scale everything else needs) --
    try:
        pts, occ, loc, scale = load_occupancy(d)
        pts_raw = unnormalize(pts, loc, scale)
        frac = float(occ.mean())
        qbox = raw_extent(pts_raw)
        vol_est = frac * float(np.prod(qbox))
        inside = pts_raw[occ]
        occ_ext = raw_extent(inside) if len(inside) >= 2 else np.full(3, np.nan)
        s = np.asarray(scale).reshape(-1)
        row.update(occ_ok=True,
                   occ_n_points=int(pts.shape[0]),
                   occ_fraction_inside=frac,
                   occ_volume_est=vol_est,
                   occ_extent_x=occ_ext[0], occ_extent_y=occ_ext[1], occ_extent_z=occ_ext[2],
                   occ_loc_x=loc[0], occ_loc_y=loc[1], occ_loc_z=loc[2],
                   occ_scale=float(s[0]) if s.size == 1 else float(np.max(s)))
        # the coordinate-frame check: loc/scale should reproduce the mesh's own bbox
        if mesh_center is not None:
            row["loc_vs_mesh_center_err"] = float(np.max(np.abs(loc - mesh_center)))
            s_full = np.repeat(s, 3) if s.size == 1 else s[:3]
            row["scale_vs_mesh_extent_err"] = float(
                np.min(np.abs(s_full - mesh_extent)) / max(mesh_extent.max(), 1e-12))
    except Exception as e:
        row["error"] += "occ:%s; " % e

    # ---- point cloud (normalized frame -> un-normalize before comparing) -----
    try:
        pc, _ = load_pointcloud(d)
        pc_raw = unnormalize(pc, loc, scale)
        ext_raw = raw_extent(pc_raw)
        ext_trim = trimmed_extent(pc_raw)
        row.update(pc_ok=True,
                   pc_n_points=int(pc.shape[0]),
                   pc_extent_x=ext_raw[0], pc_extent_y=ext_raw[1], pc_extent_z=ext_raw[2],
                   pc_trimmed_extent_x=ext_trim[0],
                   pc_trimmed_extent_y=ext_trim[1],
                   pc_trimmed_extent_z=ext_trim[2])
        if mesh_extent is not None:
            row["pc_extent_pct_diff_max"] = float(np.max(np.abs(
                [pct_diff(ext_raw[i], mesh_extent[i]) for i in range(3)])))
            row["pc_trimmed_extent_pct_diff_max"] = float(np.max(np.abs(
                [pct_diff(ext_trim[i], mesh_extent[i]) for i in range(3)])))
    except Exception as e:
        row["error"] += "pc:%s; " % e

    # ---- rendered images (presence only; features come from image_features.py)
    try:
        counts, files = count_images(d)
        for c in IMG_CONFIGS:
            row["n_images_" + c] = counts[c]
        row["n_images_total"] = len(files)
    except Exception as e:
        row["error"] += "img:%s; " % e
    return row


def _safe(d):
    try:
        return audit_instance(d)
    except Exception:
        return {"instance_id": instance_id(d), "path": d, "mesh_ok": False,
                "pc_ok": False, "occ_ok": False, "error": traceback.format_exc(limit=1)}


def main():
    ap = argparse.ArgumentParser(description="3DMSL structural audit")
    ap.add_argument("--root", required=True, help="unzipped shard directory")
    ap.add_argument("--out", default="audit_full_descriptors.csv")
    ap.add_argument("--sample", type=int, default=0, help="audit only the first N instances")
    ap.add_argument("--start", type=int, default=None, help="numeric instance id lower bound")
    ap.add_argument("--end", type=int, default=None, help="numeric instance id upper bound")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true",
                    help="skip instances already written to --out; failed rows are retried. "
                         "Partial results are checkpointed, so this survives a Colab disconnect")
    ap.add_argument("--checkpoint-every", type=int, default=200,
                    help="write partial results to --out every N instances")
    args = ap.parse_args()

    dirs = find_instance_dirs(args.root)
    if args.start is not None or args.end is not None:
        lo = args.start if args.start is not None else -10 ** 18
        hi = args.end if args.end is not None else 10 ** 18
        dirs = [d for d in dirs if instance_id(d).isdigit() and lo <= int(instance_id(d)) <= hi]
    if args.sample:
        dirs = dirs[: args.sample]
    if not dirs:
        sys.exit("No instance directories found under %s" % args.root)
    prev, done = None, set()
    if args.resume:
        prev, done = load_resume_table(
            args.out, ok_fn=lambda t: t["mesh_ok"].astype(bool) & t["pc_ok"].astype(bool) & t["occ_ok"].astype(bool))
        wanted = {instance_id(d) for d in dirs}
        if prev is not None:
            prev = prev[prev["instance_id"].isin(wanted)]
            done &= wanted
        dirs = [d for d in dirs if instance_id(d) not in done]
        print("Resuming: %d already done, %d remaining" % (len(done), len(dirs)))
    print("Auditing %d instances with %d worker(s)" % (len(dirs), args.workers), flush=True)

    t0 = time.time()
    rows = []

    def progress(i):
        if i % args.progress_every == 0 or i == len(dirs):
            print("  %d/%d  (%.2fs/instance)" % (i, len(dirs), (time.time() - t0) / i), flush=True)
        if args.checkpoint_every and i % args.checkpoint_every == 0 and i < len(dirs):
            write_checkpoint(prev, rows, args.out)

    if args.workers <= 1:
        for i, d in enumerate(dirs, 1):
            rows.append(_safe(d))
            progress(i)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_safe, d): d for d in dirs}
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                progress(i)

    df = write_checkpoint(prev, rows, args.out)

    n = len(df)
    print("\n=== AUDIT SUMMARY (n=%d, %.1fs total) ===" % (n, time.time() - t0))
    for k in ("mesh_ok", "pc_ok", "occ_ok"):
        col = df[k].fillna(False).astype(bool)
        print("  %-9s %d/%d (%.1f%%)" % (k, int(col.sum()), n, 100.0 * col.mean()))
    if "mesh_watertight" in df:
        print("  watertight %d/%d" % (int(df["mesh_watertight"].fillna(False).astype(bool).sum()), n))
    for c in IMG_CONFIGS:
        col = "n_images_" + c
        if col in df:
            bad = df[df[col] < 6]
            print("  %s renders: %d instance(s) with <6 views %s"
                  % (c, len(bad), list(bad["instance_id"])[:10]))
    if "occ_fraction_inside" in df:
        print("  occ_fraction_inside < 2%%: %d" % int((df["occ_fraction_inside"] < 0.02).sum()))
    if "mesh_volume" in df and "occ_volume_est" in df:
        err = 100.0 * (df["occ_volume_est"] - df["mesh_volume"]).abs() / df["mesh_volume"]
        print("  occupancy volume error vs mesh: mean %.2f%% / median %.2f%%"
              % (err.mean(), err.median()))
    if "loc_vs_mesh_center_err" in df:
        print("  coordinate-frame check: max |loc - mesh bbox center| = %.3g"
              % df["loc_vs_mesh_center_err"].max())
    print("\nWrote %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()
