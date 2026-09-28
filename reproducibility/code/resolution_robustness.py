"""Post-audit sensitivity analysis: all grid phases, tie rules, and connectivity.

Reads original labels. Existing native-grid measurements are recomputed. These
additional analyses are exploratory and never alter the correction model.
"""
from pathlib import Path
import sys,json,zipfile,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.ndimage import label,generate_binary_structure
from skimage.measure import euler_number
sys.path.insert(0,str(Path(__file__).parent/'original_extracted'))
import mitoem_external_validation_v2 as mm

def regrid(mask,f,oy=0,ox=0,tie_foreground=True):
    core=mask[:,1:-1,1:-1]
    core=np.pad(core,((0,0),(oy,0),(ox,0)))
    z,y,x=core.shape
    core=np.pad(core,((0,0),(0,(-y)%f),(0,(-x)%f)))
    z,y,x=core.shape
    votes=core.reshape(z,y//f,f,x//f,f).sum(axis=(2,4))
    small=(2*votes>=f*f) if tie_foreground else (2*votes>f*f)
    return np.pad(small,((0,0),(1,1),(1,1)))

def topology(mask):
    return dict(components6=int(label(mask,generate_binary_structure(3,1))[1]),
                components26=int(label(mask,generate_binary_structure(3,3))[1]),
                euler26=int(euler_number(mask,connectivity=3)))

def one(item):
    iid,mask=item
    native=mm.mesh_measures(mask,mm.SPACING_UM);nt=topology(mask);out=[]
    for f in (2,3):
        for oy in range(f):
            for ox in range(f):
                for tie in ([True,False] if f==2 else [True]):
                    small=regrid(mask,f,oy,ox,tie)
                    r=dict(instance_id=iid,factor=f,phase_y=oy,phase_x=ox,ties_foreground=tie,
                           native_volume=native['volume'],native_surface=native['surface'],native_sphericity=native['sphericity'])
                    r.update({'native_'+k:v for k,v in nt.items()})
                    r.update(topology(small))
                    try:
                        m=mm.mesh_measures(small,(.030,.008*f,.008*f))
                        r.update(m)
                        for k in ('volume','surface','sphericity'):r[k+'_change_pct']=100*(m[k]/native[k]-1)
                        r['error']=''
                    except Exception as e:r['error']=str(e)
                    out.append(r)
    return out

def selection_flow(df):
    remaining=pd.Series(True,index=df.index);out=[]
    criteria=[('slab_boundary',~df.touch_z_boundary),('image_boundary',~df.touch_xy_boundary),
              ('fewer_than_2000_voxels',df.voxel_count>=2000),('fewer_than_3_z_planes',df.z1-df.z0>=3),
              ('roi_over_2000000',df.roi_voxels<=2000000)]
    for name,keep in criteria:
        excluded=int((remaining&~keep).sum());remaining &= keep
        out.append(dict(criterion=name,excluded_sequential=excluded,remaining=int(remaining.sum()),fails_regardless_of_other_criteria=int((~keep).sum())))
    assert remaining.equals(df.complete_for_morphometry)
    return out

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser()
    ap.add_argument('--label-zip',required=True)
    ap.add_argument('--selection-table',type=Path,default=root/'data/resolution/instance_selection.csv')
    ap.add_argument('--out',type=Path,default=root/'data/resolution')
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.selection_table)
    (out/'selection_flow.json').write_text(json.dumps(selection_flow(df),indent=2))
    df.to_csv(out/'instance_selection.csv',index=False)
    with zipfile.ZipFile(args.label_zip) as z:
        names=mm.catalog(z,mm.LABEL_RE)
        objects=mm.build_masks(z,names,df[df.complete_for_morphometry])
    print(f'Built {len(objects)} masks',flush=True)
    rows=[]
    with ProcessPoolExecutor(args.workers) as ex:
        for i,part in enumerate(ex.map(one,[(iid,mask) for iid,(_,mask) in objects.items()],chunksize=2),1):
            rows.extend(part)
            if i%25==0:print(f'{i}/{len(objects)} resolution checks',flush=True)
    table=pd.DataFrame(rows);table.to_csv(out/'phase_tie_topology.csv',index=False)
    summary=[]
    for key,g in table.groupby(['factor','phase_y','phase_x','ties_foreground']):
        r=dict(zip(['factor','phase_y','phase_x','ties_foreground'],key));r['n']=len(g)
        r['failures']=int(g.error.ne('').sum())
        for k in ('volume','surface','sphericity'):
            r[k+'_median']=float(g[k+'_change_pct'].median())
            r[k+'_p05']=float(g[k+'_change_pct'].quantile(.05));r[k+'_p95']=float(g[k+'_change_pct'].quantile(.95))
        for k in ('components6','components26','euler26'):r[k+'_changed']=int(g[k].ne(g['native_'+k]).sum())
        summary.append(r)
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
