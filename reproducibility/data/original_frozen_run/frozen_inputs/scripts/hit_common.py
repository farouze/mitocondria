"""
hit_common.py — shared loaders and descriptor math for the 3DMSL reliability pipeline.

Coordinate-frame convention (verified in the Week 1-2 audit):
    raw_mesh.off       -> REAL units
    pointcloud.ply     -> NORMALIZED frame
    occupancies.npz    -> query points in NORMALIZED frame
    normalized = (raw - loc) / scale      (loc, scale stored per-instance in the npz)
so anything from the point cloud or the occupancy grid must be un-normalized
(raw = normalized * scale + loc) before it is compared with the mesh.
"""
import os
import glob
import numpy as np

IMG_CONFIGS = ("Conf1", "Conf2", "Epi1")
EXPECTED_VIEWS = 6

# -----------------------------------------------------------------------------
# instance discovery
# -----------------------------------------------------------------------------

def find_instance_dirs(root):
    """Any directory that directly contains a .off mesh is an instance directory."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Mac-created ZIPs include an __MACOSX tree containing AppleDouble
        # `._raw_mesh.off` metadata. These files are not 3DMSL meshes.
        if "__MACOSX" in os.path.normpath(dirpath).split(os.sep):
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name != "__MACOSX" and not name.startswith("._")]
        if any(f.lower().endswith(".off") and not f.startswith("._") for f in filenames):
            out.append(dirpath)
            dirnames[:] = []          # do not descend into image/render subfolders
    return sorted(out, key=lambda p: _id_key(os.path.basename(p)))


def _id_key(name):
    return (0, int(name)) if name.isdigit() else (1, name)


def instance_id(d):
    return os.path.basename(os.path.normpath(d))


def _first(d, patterns):
    for pat in patterns:
        # Search recursively within directory d for the pattern
        hits = sorted(glob.glob(os.path.join(d, '**', pat), recursive=True))
        if hits:
            return hits[0]
    return None

# -----------------------------------------------------------------------------
# loaders
# -----------------------------------------------------------------------------

def load_mesh(d):
    import trimesh
    p = _first(d, ("raw_mesh.off", "model.off", "mesh.off", "*.off"))
    if p is None:
        raise FileNotFoundError("no .off mesh in %s" % d)
    m = trimesh.load(p, process=False, force="mesh")
    if m is None or len(getattr(m, "faces", [])) == 0:
        raise ValueError("empty mesh: %s" % p)
    return m


def load_pointcloud(d):
    """Returns (N,3) points in whatever frame the file stores (normalized for 3DMSL)."""
    p = _first(d, ("pointcloud.ply", "*.ply"))
    if p is not None:
        import trimesh
        pc = trimesh.load(p, process=False)
        pts = np.asarray(pc.vertices, dtype=np.float64)
        if pts.size:
            return pts, p
    p = _first(d, ("pointcloud.npz", "points.npz"))
    if p is not None:
        z = np.load(p)
        for k in ("points", "point", "pointcloud"):
            if k in z:
                return np.asarray(z[k], dtype=np.float64), p
    raise FileNotFoundError("no point cloud in %s" % d)


def load_occupancy(d):
    """Returns (points, occ_bool, loc, scale). Points stay in the normalized frame."""
    p = _first(d, ("occupancies.npz", "points.npz", "occupancy.npz", "*.npz"))
    if p is None:
        raise FileNotFoundError("no occupancy npz in %s" % d)
    z = np.load(p)
    keys = set(z.files)
    pkey = next((k for k in ("points", "point", "query_points") if k in keys), None)
    okey = next((k for k in ("occupancies", "occupancy", "occ", "labels") if k in keys), None)
    if pkey is None or okey is None:
        raise ValueError("npz %s lacks points/occupancies (has %s)" % (p, sorted(keys)))
    pts = np.asarray(z[pkey], dtype=np.float64)
    occ = np.asarray(z[okey])
    if occ.dtype == np.uint8 and occ.size != pts.shape[0]:
        occ = np.unpackbits(occ)[: pts.shape[0]]          # Occupancy-Networks bit packing
    occ = occ.astype(bool).reshape(-1)[: pts.shape[0]]
    loc = np.asarray(z["loc"], dtype=np.float64).reshape(-1) if "loc" in keys else np.zeros(3)
    scale = np.asarray(z["scale"], dtype=np.float64).reshape(-1) if "scale" in keys else np.ones(1)
    if loc.size == 1:
        loc = np.repeat(loc, 3)
    return pts, occ, loc, scale


def unnormalize(pts, loc, scale):
    """normalized -> raw. scale may be a scalar or a per-axis vector."""
    s = np.asarray(scale, dtype=np.float64).reshape(-1)
    s = np.repeat(s, 3) if s.size == 1 else s[:3]
    return pts * s[None, :] + np.asarray(loc, dtype=np.float64).reshape(1, 3)


def count_images(d):
    """Per-configuration render counts. Config name is matched anywhere in the path."""
    files = [f for ext in ("png", "jpg", "jpeg", "tif", "tiff")
             for f in glob.glob(os.path.join(d, "**", "*." + ext), recursive=True)]
    counts = {c: 0 for c in IMG_CONFIGS}
    for f in files:
        rel = os.path.relpath(f, d).lower()
        for c in IMG_CONFIGS:
            if c.lower() in rel:
                counts[c] += 1
                break
    return counts, files


def images_for_config(d, config):
    files = [f for ext in ("png", "jpg", "jpeg", "tif", "tiff")
             for f in glob.glob(os.path.join(d, "**", "*." + ext), recursive=True)]
    return sorted(f for f in files
                  if config.lower() in os.path.relpath(f, d).lower())

# -----------------------------------------------------------------------------
# descriptor math (pure functions - these are what tests.py exercises)
# -----------------------------------------------------------------------------

def sphericity(volume, area):
    """Wadell sphericity: 1.0 for a perfect sphere, < 1.0 otherwise."""
    if area <= 0 or volume <= 0:
        return np.nan
    return (np.pi ** (1.0 / 3.0)) * ((6.0 * volume) ** (2.0 / 3.0)) / area


def raw_extent(pts):
    pts = np.asarray(pts, dtype=np.float64)
    return pts.max(axis=0) - pts.min(axis=0)


def trimmed_extent(pts, lo=1.0, hi=99.0):
    """Percentile-trimmed bounding-box extent - robust to point-cloud sampling noise."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.percentile(pts, hi, axis=0) - np.percentile(pts, lo, axis=0)


def elongation(extent):
    e = np.asarray(extent, dtype=np.float64)
    return float(e.max() / e.min()) if e.min() > 0 else np.nan


def otsu_threshold(img):
    """Otsu's method on a uint8 image. Returns the threshold level (0-255)."""
    img = np.asarray(img)
    hist = np.bincount(img.reshape(-1).astype(np.uint8), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(np.argmax(sigma_b))


def pct_diff(a, b):
    """Percent difference of a relative to b."""
    return np.nan if b == 0 else 100.0 * (a - b) / b

# -----------------------------------------------------------------------------
# checkpointing (so a Colab disconnect loses at most one checkpoint interval)
# -----------------------------------------------------------------------------

def load_resume_table(path, ok_fn=None):
    """Return (previous_rows_df or None, set_of_done_ids).

    Rows for which ok_fn(df) is False are dropped so those instances are retried."""
    import pandas as pd
    if not os.path.exists(path):
        return None, set()
    prev = pd.read_csv(path, dtype={"instance_id": str})
    if ok_fn is not None and len(prev):
        keep = ok_fn(prev).fillna(False).astype(bool)
        n_retry = int((~keep).sum())
        if n_retry:
            print("Resume: %d instance(s) failed or were computed by an older version and will be recomputed" % n_retry)
        prev = prev[keep].copy()
    return prev, set(prev["instance_id"].astype(str))


def write_checkpoint(prev, rows, path):
    """Write previous + new rows to `path` atomically, sorted by instance id."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if prev is not None and len(prev):
        df = pd.concat([prev, df], ignore_index=True)
    if "instance_id" in df:
        df["instance_id"] = df["instance_id"].astype(str)
        df = df.drop_duplicates("instance_id", keep="last")
        df = df.sort_values("instance_id", key=lambda s: s.str.zfill(12))
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return df
