#!/usr/bin/env python3
"""
tests.py - unit tests for the 3DMSL reliability pipeline.

  python3 tests.py                                  # 7 pure-function tests, no data needed
  python3 tests.py --shard-root /path/to/shard      # + the coordinate-frame test on real data

The data-dependent test re-verifies the single most load-bearing correctness claim in
the pipeline: that the occupancy npz's loc/scale reproduce the mesh's own bounding-box
centre and extent, i.e. that normalized = (raw - loc)/scale.
"""
import argparse
import sys

import numpy as np

from hit_common import (elongation, find_instance_dirs, instance_id, load_mesh,
                        load_occupancy, otsu_threshold, pct_diff, raw_extent,
                        sphericity, trimmed_extent, unnormalize)

FAILS = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  " + detail) if detail else ""))
    if not cond:
        FAILS.append(name)


def test_sphericity_sphere():
    R = 3.0
    V = 4.0 / 3.0 * np.pi * R ** 3
    A = 4.0 * np.pi * R ** 2
    check("sphericity of a perfect sphere == 1", abs(sphericity(V, A) - 1.0) < 1e-12)


def test_sphericity_bounded():
    # an elongated box must score below a sphere of the same volume
    V, A = 8.0, 2 * (1 * 8 + 1 * 1 + 8 * 1)
    s = sphericity(V, A)
    check("sphericity of a non-sphere is in (0,1)", 0 < s < 1, "s=%.4f" % s)


def test_sphericity_degenerate():
    check("sphericity handles zero area/volume",
          np.isnan(sphericity(0, 1)) and np.isnan(sphericity(1, 0)))


def test_trimmed_extent_robust():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(5000, 3))
    pts[0] = [50.0, 0.0, 0.0]                      # one sampling-noise outlier
    raw = raw_extent(pts)
    trim = trimmed_extent(pts)
    check("raw extent is corrupted by a single outlier", raw[0] > 20, "raw_x=%.1f" % raw[0])
    check("trimmed (1-99 pct) extent rejects it", trim[0] < 2.2, "trim_x=%.3f" % trim[0])


def test_elongation():
    check("elongation = longest/shortest axis", abs(elongation([4.0, 2.0, 1.0]) - 4.0) < 1e-12)


def test_otsu_separates():
    img = np.full((64, 64), 40, dtype=np.uint8)
    img[:, 32:] = 200
    t = otsu_threshold(img)
    fg = (img > t).mean()
    check("Otsu splits a bimodal image at ~50% foreground",
          40 <= t < 200 and abs(fg - 0.5) < 0.02, "thr=%.0f fg=%.3f" % (t, fg))


def test_unnormalize_roundtrip():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(100, 3)) * 7 + 30
    loc = raw.mean(axis=0)
    scale = np.array([5.0])
    norm = (raw - loc) / scale
    check("unnormalize inverts (raw-loc)/scale",
          np.allclose(unnormalize(norm, loc, scale), raw))


def test_pct_diff():
    check("pct_diff basic + zero guard",
          abs(pct_diff(110, 100) - 10.0) < 1e-12 and np.isnan(pct_diff(1, 0)))


def test_coordinate_frame(root):
    dirs = find_instance_dirs(root)
    if not dirs:
        check("coordinate-frame check (real data)", False, "no instances under " + root)
        return
    d = dirs[0]
    m = load_mesh(d)
    pts, occ, loc, scale = load_occupancy(d)
    center = np.asarray(m.bounds, dtype=np.float64).mean(axis=0)
    ext = np.asarray(m.extents, dtype=np.float64)
    s = np.asarray(scale).reshape(-1)
    s_full = np.repeat(s, 3) if s.size == 1 else s[:3]
    center_err = float(np.max(np.abs(loc - center)) / max(ext.max(), 1e-12))
    scale_err = float(np.min(np.abs(s_full - ext)) / max(ext.max(), 1e-12))
    check("occupancy `loc` == mesh bbox centre (instance %s)" % instance_id(d),
          center_err < 0.01, "rel err %.2e" % center_err)
    check("occupancy `scale` matches a mesh bbox extent (instance %s)" % instance_id(d),
          scale_err < 0.01, "rel err %.2e" % scale_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-root", default=None,
                    help="run the data-dependent coordinate-frame test against a real shard")
    args = ap.parse_args()

    print("Pure-function tests:")
    for t in (test_sphericity_sphere, test_sphericity_bounded, test_sphericity_degenerate,
              test_trimmed_extent_robust, test_elongation, test_otsu_separates,
              test_unnormalize_roundtrip, test_pct_diff):
        t()
    if args.shard_root:
        print("Data-dependent tests:")
        test_coordinate_frame(args.shard_root)
    else:
        print("Data-dependent tests: skipped (pass --shard-root to run them)")

    print("\n%s" % ("ALL TESTS PASSED" if not FAILS else "FAILURES: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()