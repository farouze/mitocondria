#!/usr/bin/env python3
"""MitoEM-R external-domain morphometry audit from the original ZIP archives.

The labeled validation section (z=400..499) is read without extracting either
archive. Voxel descriptors are calculated for all complete instances, then a
size-balanced sample is converted to meshes and surface point clouds. These
representations all derive from the same annotation, so the experiment measures
pipeline portability and representation sensitivity, not independent biological
ground truth or the specific 3DMSL occupancy-generation error.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image


SPACING_UM = (0.030, 0.008, 0.008)  # z, y, x; MitoEM-R native resolution
VOXEL_UM3 = float(np.prod(SPACING_UM))
IMAGE_RE = re.compile(r"(?:^|/)im(\d{4})\.png$")
LABEL_RE = re.compile(r"(?:^|/)seg(\d{4})\.tif$")


def catalog(archive, pattern):
    found = {}
    for name in archive.namelist():
        if name.startswith("__MACOSX/") or "/._" in name:
            continue
        match = pattern.search(name)
        if match:
            index = int(match.group(1))
            if index in found:
                raise ValueError(f"Duplicate slice {index} in {archive.filename}")
            found[index] = name
    return found


def read_slice(archive, name):
    with archive.open(name) as stream:
        image = Image.open(stream)
        image.load()
        return np.asarray(image).copy()


def pixel_bounds(labels, active_ids):
    """Return per-label XY bounds; scipy is fast, NumPy remains a fallback."""
    try:
        from scipy import ndimage as ndi

        objects = ndi.find_objects(labels)
        for instance_id in active_ids:
            box = objects[instance_id - 1]
            if box is not None:
                yield int(instance_id), box[0].start, box[0].stop, box[1].start, box[1].stop
    except ImportError:
        y, x = np.nonzero(labels)
        ids = labels[y, x]
        size = int(ids.max()) + 1
        min_y = np.full(size, labels.shape[0], dtype=np.int32)
        min_x = np.full(size, labels.shape[1], dtype=np.int32)
        max_y = np.zeros(size, dtype=np.int32)
        max_x = np.zeros(size, dtype=np.int32)
        np.minimum.at(min_y, ids, y)
        np.minimum.at(min_x, ids, x)
        np.maximum.at(max_y, ids, y + 1)
        np.maximum.at(max_x, ids, x + 1)
        for instance_id in active_ids:
            yield int(instance_id), int(min_y[instance_id]), int(max_y[instance_id]), int(min_x[instance_id]), int(max_x[instance_id])


def audit_pair(image_zip, label_zip, image_names, label_names, indices, outdir):
    samples = sorted(set((indices[0], indices[len(indices) // 2], indices[-1])))
    audit = []
    for z in samples:
        image = read_slice(image_zip, image_names[z])
        label = read_slice(label_zip, label_names[z])
        if image.shape != label.shape:
            raise ValueError(f"Image/label shape mismatch at z={z}: {image.shape} vs {label.shape}")
        if image.shape != (4096, 4096):
            raise ValueError(f"Unexpected slice shape at z={z}: {image.shape}")
        if not np.issubdtype(label.dtype, np.integer):
            raise ValueError(f"Labels are not integer instance IDs at z={z}: {label.dtype}")
        values = np.unique(label)
        audit.append({"z": z, "shape": list(label.shape), "image_dtype": str(image.dtype),
                      "label_dtype": str(label.dtype), "nonzero_labels": int(np.count_nonzero(values)),
                      "labeled_fraction": float(np.mean(label > 0))})
        if z == samples[1]:
            scale = 8
            gray = image[::scale, ::scale]
            binary = label[::scale, ::scale] > 0
            rgb = np.repeat(gray[..., None], 3, axis=2)
            rgb[binary, 0] = np.maximum(rgb[binary, 0], 210)
            rgb[binary, 1] = (rgb[binary, 1] // 2)
            rgb[binary, 2] = (rgb[binary, 2] // 2)
            Image.fromarray(rgb.astype(np.uint8)).save(outdir / f"mitoem_R_overlay_z{z:04d}.png")
    return audit


def scan_instances(label_zip, label_names, indices, min_voxels, min_z_planes, max_roi_voxels):
    records = {}
    for n, z in enumerate(indices, 1):
        label = read_slice(label_zip, label_names[z])
        counts = np.bincount(label.ravel())
        active = np.flatnonzero(counts[1:]) + 1
        for instance_id, y0, y1, x0, x1 in pixel_bounds(label, active):
            count = int(counts[instance_id])
            if instance_id not in records:
                records[instance_id] = dict(instance_id=instance_id, voxel_count=0,
                                            z0=z, z1=z + 1, y0=y0, y1=y1, x0=x0, x1=x1)
            row = records[instance_id]
            row["voxel_count"] += count
            row["z0"] = min(row["z0"], z)
            row["z1"] = max(row["z1"], z + 1)
            row["y0"] = min(row["y0"], y0)
            row["y1"] = max(row["y1"], y1)
            row["x0"] = min(row["x0"], x0)
            row["x1"] = max(row["x1"], x1)
        if n % 10 == 0 or n == len(indices):
            print(f"Scanned label slices: {n}/{len(indices)}", flush=True)
    rows = pd.DataFrame.from_records(list(records.values())).sort_values("instance_id")
    if rows.empty:
        raise ValueError("No labeled mitochondria found in the requested slice range")
    rows["volume_um3"] = rows.voxel_count * VOXEL_UM3
    rows["z_extent_um"] = (rows.z1 - rows.z0) * SPACING_UM[0]
    rows["y_extent_um"] = (rows.y1 - rows.y0) * SPACING_UM[1]
    rows["x_extent_um"] = (rows.x1 - rows.x0) * SPACING_UM[2]
    rows["bbox_fill"] = rows.volume_um3 / (rows.z_extent_um * rows.y_extent_um * rows.x_extent_um)
    rows["touch_z_boundary"] = (rows.z0 == indices[0]) | (rows.z1 == indices[-1] + 1)
    rows["touch_xy_boundary"] = (rows.y0 == 0) | (rows.x0 == 0) | (rows.y1 == 4096) | (rows.x1 == 4096)
    rows["roi_voxels"] = (rows.z1 - rows.z0 + 2) * (rows.y1 - rows.y0 + 2) * (rows.x1 - rows.x0 + 2)
    rows["complete_for_morphometry"] = (
        ~rows.touch_z_boundary & ~rows.touch_xy_boundary &
        (rows.voxel_count >= min_voxels) &
        (rows.z1 - rows.z0 >= min_z_planes) &
        (rows.roi_voxels <= max_roi_voxels)
    )
    return rows


def balanced_sample(rows, n, seed):
    eligible = rows[rows.complete_for_morphometry].copy()
    if eligible.empty or n <= 0:
        return eligible.iloc[:0]
    if len(eligible) <= n:
        return eligible
    eligible["size_quartile"] = pd.qcut(eligible.voxel_count, 4, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    pools = {int(q): rng.permutation(group.index.to_numpy()).tolist()
             for q, group in eligible.groupby("size_quartile")}
    chosen = []
    while len(chosen) < n and any(pools.values()):
        for q in sorted(pools):
            if pools[q] and len(chosen) < n:
                chosen.append(pools[q].pop())
    return eligible.loc[chosen].sort_values("instance_id")


def build_masks(label_zip, label_names, selected):
    objects = {}
    for row in selected.itertuples():
        shape = (row.z1 - row.z0 + 2, row.y1 - row.y0 + 2, row.x1 - row.x0 + 2)
        objects[int(row.instance_id)] = (row, np.zeros(shape, dtype=bool))
    if not objects:
        return objects
    first = min(row.z0 for row, _ in objects.values())
    last = max(row.z1 for row, _ in objects.values())
    for z in range(first, last):
        current = [(row, mask) for row, mask in objects.values() if row.z0 <= z < row.z1]
        if not current:
            continue
        label = read_slice(label_zip, label_names[z])
        for row, mask in current:
            mask[z - row.z0 + 1, 1:-1, 1:-1] = (label[row.y0:row.y1, row.x0:row.x1] == row.instance_id)
    return objects


def voxel_surface_um2(mask):
    z_area = SPACING_UM[1] * SPACING_UM[2]
    y_area = SPACING_UM[0] * SPACING_UM[2]
    x_area = SPACING_UM[0] * SPACING_UM[1]
    return float(np.count_nonzero(mask[1:] != mask[:-1]) * z_area +
                 np.count_nonzero(mask[:, 1:] != mask[:, :-1]) * y_area +
                 np.count_nonzero(mask[:, :, 1:] != mask[:, :, :-1]) * x_area)


def point_extent_error(vertices, faces, mesh_extent, rng, count, repeats):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=1)
    valid = area > 0
    if not np.any(valid):
        return np.nan, np.nan
    triangles = triangles[valid]
    probabilities = area[valid] / area[valid].sum()
    raw, trimmed = [], []
    for _ in range(repeats):
        t = triangles[rng.choice(len(triangles), size=count, p=probabilities)]
        u = np.sqrt(rng.random(count))[:, None]
        v = rng.random(count)[:, None]
        points = (1 - u) * t[:, 0] + u * (1 - v) * t[:, 1] + u * v * t[:, 2]
        raw_extent = np.ptp(points, axis=0)
        low, high = np.quantile(points, [0.025, 0.975], axis=0)
        raw.append(np.mean(100 * (raw_extent / mesh_extent - 1)))
        trimmed.append(np.mean(100 * ((high - low) / mesh_extent - 1)))
    return float(np.mean(raw)), float(np.mean(trimmed))


def mesh_comparison(objects, seed, point_count, point_repeats):
    if not objects:
        return pd.DataFrame()
    try:
        from skimage.measure import marching_cubes
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Mesh comparison needs scikit-image and trimesh; run the notebook's dependency cell") from exc
    results = []
    rng = np.random.default_rng(seed)
    for index, (instance_id, (row, mask)) in enumerate(objects.items(), 1):
        record = {"instance_id": instance_id, "voxel_count": int(row.voxel_count),
                  "voxel_volume_um3": float(row.volume_um3),
                  "voxel_surface_um2": voxel_surface_um2(mask)}
        try:
            vertices, faces, _, _ = marching_cubes(mask.astype(np.uint8), level=0.5, spacing=SPACING_UM)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            record["mesh_watertight"] = bool(mesh.is_watertight)
            record["mesh_faces"] = int(len(faces))
            if not mesh.is_watertight:
                raise ValueError("Mesh was not watertight")
            record["mesh_volume_um3"] = float(abs(mesh.volume))
            record["mesh_surface_um2"] = float(mesh.area)
            record["mesh_vs_voxel_volume_pct"] = 100 * (record["mesh_volume_um3"] / row.volume_um3 - 1)
            record["mesh_vs_voxel_surface_pct"] = 100 * (record["mesh_surface_um2"] / record["voxel_surface_um2"] - 1)
            extent = np.ptp(vertices, axis=0)
            raw, trimmed = point_extent_error(vertices, faces, extent, rng, point_count, point_repeats)
            record["point_raw_extent_error_pct"] = raw
            record["point_trimmed_extent_error_pct"] = trimmed
            record["error"] = ""
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        results.append(record)
        print(f"Meshed instances: {index}/{len(objects)}", flush=True)
    return pd.DataFrame(results)


def metric_line(values):
    x = pd.Series(values, dtype=float).dropna().to_numpy()
    if not len(x):
        return "unavailable"
    return f"median {np.median(x):+.2f}%; 5th–95th percentile {np.percentile(x, 5):+.2f}% to {np.percentile(x, 95):+.2f}%"


def make_report(audit, rows, selected, agreement, args, outdir):
    success = agreement[agreement.error.eq("")] if not agreement.empty else agreement
    phase = "validation" if args.start >= 400 else "training-slice pilot"
    lines = [f"# MitoEM-R {phase} morphometry audit", "",
             f"Analyzed labeled slices **{args.start}–{args.stop - 1}** at 30 × 8 × 8 nm voxel spacing (z × y × x).",
             "Image and label ZIPs were read directly; no full-volume extraction was needed.", "",
             "## Image–label audit", "",
             "| Slice | Shape | Image type | Label type | Instance IDs | Labeled fraction |",
             "|---:|---:|---|---|---:|---:|"]
    for row in audit:
        lines.append(f"| {row['z']} | {row['shape'][0]} × {row['shape'][1]} | {row['image_dtype']} | "
                     f"{row['label_dtype']} | {row['nonzero_labels']} | {row['labeled_fraction']:.3f} |")
    lines += ["", "## Instance selection", "",
              f"- Unique instance IDs intersecting this slab: **{len(rows)}**.",
              f"- Complete instances passing ≥{args.min_voxels:,} voxels, ≥{args.min_z_planes} z planes, "
              f"no slab/image-edge contact, and ROI ≤{args.max_roi_voxels:,} voxels: **{int(rows.complete_for_morphometry.sum())}**.",
              f"- Size-balanced sample requested: **{args.n_mesh}**; selected: **{len(selected)}**; successful mesh comparisons: **{len(success)}**.", ""]
    if len(success):
        lines += ["## Representation agreement in the selected instances", "",
                  f"- Mesh versus source voxel volume: {metric_line(success.mesh_vs_voxel_volume_pct)}.",
                  f"- Mesh versus voxel-face surface area: {metric_line(success.mesh_vs_voxel_surface_pct)}.",
                  f"- Point-cloud raw extent versus mesh extent: {metric_line(success.point_raw_extent_error_pct)}.",
                  f"- Point-cloud 2.5–97.5% trimmed extent versus mesh extent: {metric_line(success.point_trimmed_extent_error_pct)}.", "",
                  "| Candidate absolute volume margin | Fraction of successful mesh comparisons within margin |",
                  "|---:|---:|"]
        for margin in (2.17, 5.0, 6.0, 10.0):
            fraction = np.mean(np.abs(success.mesh_vs_voxel_volume_pct) <= margin)
            lines.append(f"| {margin:.2f}% | {fraction:.1%} |")
    lines += ["", "## Interpretation", "",
              "The voxel mask, mesh, and sampled surface points all derive from the same MitoEM annotation. "
              "Their agreement measures representation sensitivity of this processing pipeline on real EM data. "
              "It does not test independent acquisition agreement or reproduce the 3DMSL occupancy query process.", "",
              "Candidate margins above are exploratory sensitivity checks. Choose and freeze a primary margin "
              "with the adviser before a confirmatory external analysis. The 2.17% value originates in the "
              "3DMSL sampling repeatability calculation and is not a biological threshold.", "",
              "This is one rat cortex tissue volume; instances within it are spatially dependent. "
              "No mitochondrial function, cardiovascular outcome, or independent animal replicate is present. "
              "Annotation errors and the challenge's minimum annotated size can alter the observed size distribution.", ""]
    (outdir / "mitoem_R_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-zip", type=Path, required=True)
    parser.add_argument("--label-zip", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=400)
    parser.add_argument("--stop", type=int, default=500, help="exclusive z index")
    parser.add_argument("--n-mesh", type=int, default=40)
    parser.add_argument("--min-voxels", type=int, default=2000)
    parser.add_argument("--min-z-planes", type=int, default=3)
    parser.add_argument("--max-roi-voxels", type=int, default=2_000_000)
    parser.add_argument("--point-count", type=int, default=10_000)
    parser.add_argument("--point-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not (0 <= args.start < args.stop <= 500):
        raise ValueError("Labeled slices must be inside 0–499")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.image_zip) as image_zip, ZipFile(args.label_zip) as label_zip:
        image_names = catalog(image_zip, IMAGE_RE)
        label_names = catalog(label_zip, LABEL_RE)
        if set(image_names) != set(range(1000)):
            raise ValueError(f"Expected image slices 0–999; found {len(image_names)}")
        if set(label_names) != set(range(500)):
            raise ValueError(f"Expected label slices 0–499; found {len(label_names)}")
        for z in range(400):
            if "mito-train-v2" not in label_names[z]:
                raise ValueError(f"Incorrect training label path at z={z}")
        for z in range(400, 500):
            if "mito-val-v2" not in label_names[z]:
                raise ValueError(f"Incorrect validation label path at z={z}")
        indices = list(range(args.start, args.stop))
        audit = audit_pair(image_zip, label_zip, image_names, label_names, indices, args.outdir)
        rows = scan_instances(label_zip, label_names, indices, args.min_voxels,
                              args.min_z_planes, args.max_roi_voxels)
        rows.to_csv(args.outdir / "mitoem_R_instance_features.csv", index=False)
        selected = balanced_sample(rows, args.n_mesh, args.seed)
        if len(selected):
            objects = build_masks(label_zip, label_names, selected)
            agreement = mesh_comparison(objects, args.seed, args.point_count, args.point_repeats)
        else:
            agreement = pd.DataFrame(columns=["instance_id", "error"])
        agreement.to_csv(args.outdir / "mitoem_R_representation_agreement.csv", index=False)
        metadata = {"image_zip": str(args.image_zip), "label_zip": str(args.label_zip),
                    "image_slice_count": len(image_names), "label_slice_count": len(label_names),
                    "analysis_z_start": args.start, "analysis_z_stop_exclusive": args.stop,
                    "voxel_spacing_um_zyx": SPACING_UM, "seed": args.seed,
                    "n_instances_intersecting_slab": len(rows),
                    "n_complete_eligible": int(rows.complete_for_morphometry.sum()),
                    "n_mesh_selected": len(selected), "audit_samples": audit,
                    "settings": {key: value for key, value in vars(args).items() if key not in ("image_zip", "label_zip", "outdir")}}
        (args.outdir / "mitoem_R_audit.json").write_text(json.dumps(metadata, indent=2))
        make_report(audit, rows, selected, agreement, args, args.outdir)


if __name__ == "__main__":
    main()