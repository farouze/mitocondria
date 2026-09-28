#!/usr/bin/env python3
"""Quantify orientation and microscope-configuration sensitivity in 3DMSL renders."""
import argparse, os, re, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from hit_common import IMG_CONFIGS, find_instance_dirs, images_for_config, instance_id
from image_features import _view_features

FEATURES=('fg_fraction','fg_bbox_area_frac','mean','std','otsu')

def view_name(path):
    s=os.path.splitext(os.path.basename(path))[0]
    m=re.search(r'_(0|x90|x180|y90|y180|z90)$',s)
    return m.group(1) if m else s

def one_instance(d):
    rows=[]
    for c in IMG_CONFIGS:
        for f in images_for_config(d,c):
            z=_view_features(f);z.update(instance_id=instance_id(d),config=c,view=view_name(f),file=os.path.basename(f));rows.append(z)
    return rows

def safe(d):
    try:return one_instance(d)
    except Exception as e:return [dict(instance_id=instance_id(d),error=str(e))]

def icc21(M):
    M=np.asarray(M,float);n,k=M.shape
    if n<2 or k<2:return np.nan
    g=M.mean();rm=M.mean(1);cm=M.mean(0)
    MSR=k*((rm-g)**2).sum()/(n-1);MSC=n*((cm-g)**2).sum()/(k-1)
    MSE=((M-rm[:,None]-cm[None,:]+g)**2).sum()/((n-1)*(k-1))
    den=MSR+(k-1)*MSE+k*(MSC-MSE)/n
    return float((MSR-MSE)/den) if den else np.nan

def boot_icc(M,n=1000,seed=790):
    rng=np.random.default_rng(seed);vals=[]
    for _ in range(n):
        v=icc21(M[rng.integers(0,len(M),len(M))])
        if np.isfinite(v):vals.append(v)
    return tuple(np.percentile(vals,[2.5,97.5]))
def rankcorr(x,y):
    a=pd.Series(x).rank().to_numpy();b=pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(a,b)[0,1]) if a.std() and b.std() else np.nan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--outdir',required=True)
    ap.add_argument('--sample',type=int,default=400);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--n-boot',type=int,default=1000)
    a=ap.parse_args();os.makedirs(a.outdir,exist_ok=True)
    dirs=find_instance_dirs(a.root);dirs=dirs[:a.sample] if a.sample else dirs
    rows=[];t=time.time()
    if a.workers<=1:
        for d in dirs:rows.extend(safe(d))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            fs=[ex.submit(safe,d) for d in dirs]
            for i,f in enumerate(as_completed(fs),1):
                rows.extend(f.result())
                if i%100==0 or i==len(fs):print('%d/%d'%(i,len(fs)),flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(os.path.join(a.outdir,'imaging_view_features.csv'),index=False)
    ok=raw.dropna(subset=['config','view']).copy()
    agg=ok.groupby(['instance_id','config'])[list(FEATURES)].agg(['mean','std'])
    flat=pd.DataFrame(index=agg.index)
    for f in FEATURES:
        flat[f+'_mean']=agg[(f,'mean')];flat[f+'_orientation_cv']=agg[(f,'std')]/agg[(f,'mean')].abs().clip(lower=1e-12)
    flat=flat.reset_index();flat.to_csv(os.path.join(a.outdir,'imaging_condition_summary.csv'),index=False)
    L=['# Imaging-Condition Reliability','',
       'This stage measures two distinct effects: variation across the six orientations within a configuration, and agreement across Conf1, Conf2, and Epi1 after averaging orientations.','',
       '## Orientation sensitivity','', '| Configuration | feature | median CV | 95th-percentile CV |','|---|---|---:|---:|']
    for c in IMG_CONFIGS:
        d=flat[flat.config==c]
        for f in ('fg_fraction','fg_bbox_area_frac'):
            x=d[f+'_orientation_cv'].replace([np.inf,-np.inf],np.nan).dropna()
            L.append('| %s | %s | %.3f | %.3f |'%(c,f,x.median(),x.quantile(.95)))
    L += ['', '## Cross-configuration agreement','',
          '| Feature | complete n | ICC(2,1) | 95% bootstrap CI | lowest pairwise Spearman |','|---|---:|---:|---:|---:|']
    for j,f in enumerate(('fg_fraction','fg_bbox_area_frac')):
        w=flat.pivot(index='instance_id',columns='config',values=f+'_mean').dropna()
        cols=[c for c in IMG_CONFIGS if c in w];M=w[cols].to_numpy();ic=icc21(M);ci=boot_icc(M,a.n_boot,790+j)
        rs=[rankcorr(w[x],w[y]) for q,x in enumerate(cols) for y in cols[q+1:]]
        L.append('| %s | %d | %.3f | [%.3f, %.3f] | %.3f |'%(f,len(w),ic,ci[0],ci[1],min(rs)))
    L += ['', '## Interpretation guardrails','',
          '- These are measurement-reliability results. They do not establish mitochondrial function.',
          '- Per-image min-max intensity normalization means intensity means and thresholds should not be compared as calibrated physical intensities.',
          '- A constant or saturated Epi1 bounding-box feature is an assay failure for that descriptor, not evidence of invariance.',
          '- Run the sample first. If the results are stable and QC is acceptable, rerun with `--sample 0` for the full shard.','']
    with open(os.path.join(a.outdir,'imaging_condition_report.md'),'w') as f:f.write('\n'.join(L))
    print('\n'.join(L));print('Completed in %.1fs'%(time.time()-t))
if __name__=='__main__':main()
