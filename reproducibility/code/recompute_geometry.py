"""Reproduce mesh/occupancy descriptors directly from the two original ZIPs.

This deliberately reconstructs development artifacts; it does not claim that a
newly reconstructed model is the original, time-stamped frozen model.
"""
import argparse, io, json, zipfile, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import trimesh

ARCHIVE = None
MEMBERS = None

def init(path):
    global ARCHIVE, MEMBERS
    ARCHIVE = zipfile.ZipFile(path)
    MEMBERS = {}
    for name in ARCHIVE.namelist():
        p = Path(name)
        if '__MACOSX' not in p.parts and p.name in ('raw_mesh.off', 'occupancies.npz'):
            MEMBERS.setdefault(p.parent.name, {})[p.name] = name

def one(iid):
    files = MEMBERS[iid]
    raw = ARCHIVE.read(files['raw_mesh.off'])
    m = trimesh.load(io.BytesIO(raw), file_type='off', process=False, force='mesh')
    with np.load(io.BytesIO(ARCHIVE.read(files['occupancies.npz']))) as z:
        pts = np.asarray(z['points'], dtype=float)
        occ = z['occupancies']
        if len(occ) != len(pts): occ = np.unpackbits(occ)[:len(pts)]
        occ = occ.astype(bool)
        loc = np.asarray(z['loc'], dtype=float)
        scale = float(np.asarray(z['scale']).reshape(()))
    praw = pts * scale + loc
    ext = np.ptp(praw[occ], axis=0)
    vol, area = abs(float(m.volume)), float(m.area)
    row = dict(instance_id=int(iid), mesh_volume=vol, mesh_area=area,
               mesh_watertight=bool(m.is_watertight),
               mesh_sphericity=np.pi**(1/3)*(6*vol)**(2/3)/area,
               mesh_elongation=float(max(m.extents)/min(m.extents)),
               occ_scale=scale, occ_n_points=len(pts), occ_fraction_inside=float(occ.mean()),
               occ_volume_est=float(occ.mean()*np.prod(np.ptp(praw, axis=0))),
               occ_volume_known_box=float(occ.mean()*(1.1*scale)**3),
               mesh_sha256=hashlib.sha256(raw).hexdigest())
    for k, v in zip('xyz',ext): row['occ_extent_'+k]=v
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    init(a.zip);ids=sorted(MEMBERS,key=int)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(a.workers,initializer=init,initargs=(a.zip,)) as ex:
        rows=[]
        for i,row in enumerate(ex.map(one,ids,chunksize=12),1):
            rows.append(row)
            if i%250==0:print(f'{i}/{len(ids)}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out,index=False)
    print(json.dumps({'n':len(df),'mean_error':float((100*(df.occ_volume_est/df.mesh_volume-1)).mean())}),flush=True)
if __name__=='__main__':main()
