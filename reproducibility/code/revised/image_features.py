#!/usr/bin/env python3
"""
image_features.py - Weeks 5-6: descriptors from the simulated-microscopy renders.

For each instance and each configuration (Conf1, Conf2, Epi1) it reads every view,
computes intensity statistics and an Otsu-threshold foreground segmentation, then
aggregates across views (mean and max).

Columns produced per configuration C:
    img_C_n_views
    img_C_intensity_mean / img_C_intensity_std
    img_C_fg_fraction        / img_C_fg_fraction_max
    img_C_fg_bbox_area_frac  / img_C_fg_bbox_area_frac_max
    img_C_otsu_threshold

NOTE (full-shard finding): img_Epi1_fg_bbox_area_frac_max is exactly 1.0 for every
instance - Otsu never finds a clean split on the blurred epifluorescence renders.
It is a zero-variance column and is dropped automatically by agreement_stats.py.

  python3 image_features.py --root /path/to/shard --out image_features.csv
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

from hit_common import IMG_CONFIGS, find_instance_dirs, images_for_config, instance_id, otsu_threshold


def _view_features(path):
    im = Image.open(path)
    if im.mode not in ("L", "I;16", "I"):
        im = im.convert("L")
    a = np.asarray(im).astype(np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.max() > 255 or a.dtype != np.uint8:
        rng = a.max() - a.min()
        a8 = np.zeros_like(a, dtype=np.uint8) if rng == 0 else \
            ((a - a.min()) / rng * 255.0).astype(np.uint8)
    else:
        a8 = a.astype(np.uint8)

    thr = otsu_threshold(a8)
    fg = a8 > thr
    frac = float(fg.mean())
    if fg.any():
        rs, cs = np.where(fg)
        bbox = (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)
        bbox_frac = float(bbox) / float(fg.size)
    else:
        bbox_frac = 0.0
    return dict(mean=float(a8.mean()), std=float(a8.std()),
                fg_fraction=frac, fg_bbox_area_frac=bbox_frac, otsu=thr)


def instance_image_features(d):
    row = {"instance_id": instance_id(d)}
    for c in IMG_CONFIGS:
        files = images_for_config(d, c)
        per = []
        for f in files:
            try:
                per.append(_view_features(f))
            except Exception:
                pass
        row["img_%s_n_views" % c] = len(per)
        if not per:
            for k in ("intensity_mean", "intensity_std", "fg_fraction", "fg_fraction_max",
                      "fg_bbox_area_frac", "fg_bbox_area_frac_max", "otsu_threshold"):
                row["img_%s_%s" % (c, k)] = np.nan
            continue
        g = lambda k: np.array([p[k] for p in per], dtype=np.float64)
        row["img_%s_intensity_mean" % c] = float(g("mean").mean())
        row["img_%s_intensity_std" % c] = float(g("std").mean())
        row["img_%s_fg_fraction" % c] = float(g("fg_fraction").mean())
        row["img_%s_fg_fraction_max" % c] = float(g("fg_fraction").max())
        row["img_%s_fg_bbox_area_frac" % c] = float(g("fg_bbox_area_frac").mean())
        row["img_%s_fg_bbox_area_frac_max" % c] = float(g("fg_bbox_area_frac").max())
        row["img_%s_otsu_threshold" % c] = float(g("otsu").mean())
    return row


def _safe(d):
    try:
        return instance_image_features(d)
    except Exception as e:
        return {"instance_id": instance_id(d), "img_error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="Image-derived descriptors for 3DMSL renders")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="image_features.csv")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--progress-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dirs = find_instance_dirs(args.root)
    if args.sample:
        dirs = dirs[: args.sample]
    done = set()
    if args.resume and os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=["instance_id"])["instance_id"].astype(str))
        dirs = [d for d in dirs if instance_id(d) not in done]
        print("Resuming: %d already done, %d remaining" % (len(done), len(dirs)))
        if not dirs:
            return
    print("Image features for %d instances" % len(dirs), flush=True)

    t0, rows = time.time(), []
    if args.workers <= 1:
        for i, d in enumerate(dirs, 1):
            rows.append(_safe(d))
            if i % args.progress_every == 0 or i == len(dirs):
                print("  %d/%d" % (i, len(dirs)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_safe, d) for d in dirs]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % args.progress_every == 0 or i == len(dirs):
                    print("  %d/%d" % (i, len(dirs)), flush=True)

    df = pd.DataFrame(rows)
    if done:
        df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
    df = df.sort_values("instance_id", key=lambda s: s.astype(str).str.zfill(12))
    df.to_csv(args.out, index=False)
    ok = int((df[[c for c in df.columns if c.endswith("_n_views")]].sum(axis=1) > 0).sum())
    print("Extracted features for %d/%d instances (%.1fs)" % (ok, len(df), time.time() - t0))
    for c in IMG_CONFIGS:
        col = "img_%s_fg_bbox_area_frac_max" % c
        if col in df and df[col].notna().any() and df[col].nunique(dropna=True) == 1:
            print("  NOTE: %s is constant (%.3f) - zero-variance column"
                  % (col, df[col].dropna().iloc[0]))
    print("Wrote %s" % os.path.abspath(args.out))


if __name__ == "__main__":
    main()