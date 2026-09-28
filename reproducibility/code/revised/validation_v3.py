#!/usr/bin/env python3
"""Internal validation after exploratory development of occupancy-volume correction.

Fits correction parameters on training instances, selects a 95% residual bound on
calibration instances, and reports performance once on held-out test instances.
This supersedes in-sample correction rates from agreement_v2.py.
"""
import argparse, json, os
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt=None

REQ = ['instance_id','mesh_volume','mesh_area','occ_volume_est','occ_fraction_inside',
       'occ_extent_x','occ_extent_y','occ_extent_z']
CANDIDATE_MARGINS = (2.17, 5.0, 6.0, 10.0)

def stratified_split(df, seed=790):
    """60/20/20 split within mesh-volume deciles."""
    rng=np.random.default_rng(seed)
    bins=pd.qcut(df['mesh_volume'],10,labels=False,duplicates='drop')
    split=np.empty(len(df),dtype=object)
    for b in sorted(pd.unique(bins)):
        idx=np.flatnonzero(np.asarray(bins)==b); rng.shuffle(idx)
        n=len(idx); a=int(round(.60*n)); c=int(round(.20*n))
        split[idx[:a]]='train'; split[idx[a:a+c]]='calibration'; split[idx[a+c:]]='test'
    return split

def feature_matrix(df, means=None, scales=None):
    bbox=df['occ_extent_x']*df['occ_extent_y']*df['occ_extent_z']
    raw=np.column_stack([
        df['occ_volume_est']/bbox.clip(lower=1e-12),
        df['occ_fraction_inside'],
        np.log(df['occ_volume_est'].clip(lower=1e-12)),
        np.log(bbox.clip(lower=1e-12)),
        df[['occ_extent_x','occ_extent_y','occ_extent_z']].max(axis=1) /
          df[['occ_extent_x','occ_extent_y','occ_extent_z']].min(axis=1).clip(lower=1e-12),
    ]).astype(float)
    if means is None: means=np.nanmean(raw,axis=0)
    if scales is None: scales=np.nanstd(raw,axis=0,ddof=1)
    scales=np.where((scales==0)|~np.isfinite(scales),1.,scales)
    return np.column_stack([np.ones(len(raw)),(raw-means)/scales]),means,scales

def fit_ols(X,y): return np.linalg.lstsq(X,np.asarray(y,float),rcond=None)[0]
def predict_pct(X,b): return np.clip(X@b,-50,100)
def signed_error(df, corrected=None):
    value=df['occ_volume_est'].to_numpy() if corrected is None else np.asarray(corrected)
    return 100*(value/df['mesh_volume'].to_numpy()-1)
def corrected_volume(df,pred_pct): return df['occ_volume_est'].to_numpy()/(1+np.asarray(pred_pct)/100)
def conformal_q(abs_resid,coverage=.95):
    x=np.sort(np.asarray(abs_resid,float)); n=len(x)
    if n == 0 or not np.isfinite(x).all() or not 0 < coverage < 1:
        raise ValueError("Finite nonempty scores and coverage in (0,1) are required")
    k=int(np.ceil((n+1)*coverage))
    return float(x[k-1]) if k <= n else float("inf")
def boot_ci(x,fn,n_boot=1000,seed=790):
    x=np.asarray(x,float);rng=np.random.default_rng(seed)
    z=np.array([fn(x[rng.integers(0,len(x),len(x))]) for _ in range(n_boot)])
    return tuple(np.percentile(z,[2.5,97.5]))
def summarize(e,n_boot,seed):
    e=np.asarray(e,float)
    return dict(n=len(e),mean_bias=float(e.mean()),mean_bias_ci=boot_ci(e,np.mean,n_boot,seed),
                median_abs=float(np.median(abs(e))),median_abs_ci=boot_ci(e,lambda x:np.median(abs(x)),n_boot,seed+1),
                rmse=float(np.sqrt(np.mean(e**2))),loa_lo=float(e.mean()-1.96*e.std(ddof=1)),
                loa_hi=float(e.mean()+1.96*e.std(ddof=1)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--in',dest='inp',required=True);ap.add_argument('--outdir',required=True)
    ap.add_argument('--seed',type=int,default=790);ap.add_argument('--n-boot',type=int,default=1000)
    a=ap.parse_args();os.makedirs(a.outdir,exist_ok=True)
    df=pd.read_csv(a.inp);missing=[c for c in REQ if c not in df]
    if missing: raise ValueError('missing columns: '+', '.join(missing))
    df=df[REQ].replace([np.inf,-np.inf],np.nan).dropna().copy()
    df=df[(df[['mesh_volume','occ_volume_est','occ_extent_x','occ_extent_y','occ_extent_z']]>0).all(axis=1)].reset_index(drop=True)
    df['split']=stratified_split(df,a.seed)
    tr=df.split=='train';cal=df.split=='calibration';te=df.split=='test'
    y=signed_error(df)
    Xtr,mu,sd=feature_matrix(df[tr]);Xall,_,_=feature_matrix(df,mu,sd)
    beta=fit_ols(Xtr,y[tr]);global_bias=float(y[tr].mean())
    preds={'uncorrected':np.zeros(len(df)),'global_train_mean':np.full(len(df),global_bias),
           'occupancy_only_ols':predict_pct(Xall,beta)}
    errors={k:signed_error(df,corrected_volume(df,v)) if k!='uncorrected' else y for k,v in preds.items()}
    summaries={};bounds={}
    for j,(name,e) in enumerate(errors.items()):
        q=conformal_q(abs(e[cal]),.95);bounds[name]=q
        s=summarize(e[te],a.n_boot,a.seed+10*j);s['calibration_q95_abs_error']=q
        s['test_coverage_at_q95']=float(np.mean(abs(e[te])<=q))
        s['within_candidate_margins']={str(m):float(np.mean(abs(e[te])<=m)) for m in CANDIDATE_MARGINS}
        summaries[name]=s
    out=df[['instance_id','split','mesh_volume','occ_volume_est']].copy()
    for name in preds:
        out['predicted_bias_'+name]=preds[name];out['corrected_error_'+name]=errors[name]
    out.to_csv(os.path.join(a.outdir,'validation_v3_predictions.csv'),index=False)
    df[['instance_id','split']].to_csv(os.path.join(a.outdir,'validation_v3_splits.csv'),index=False)
    model={'calibration_score':'abs(100*(corrected_volume/mesh_volume-1))', 'calibration_index_1based':int(np.ceil((int(cal.sum())+1)*.95)), 'seed':a.seed,'features':['bbox_fill','occ_fraction_inside','log_occ_volume','log_bbox_volume','occ_extent_elongation'],
           'feature_means':mu.tolist(),'feature_scales':sd.tolist(),'coefficients':beta.tolist(),
           'global_train_mean_bias_pct':global_bias,'calibration_bounds_pct':bounds,'summaries':summaries}
    with open(os.path.join(a.outdir,'validation_v3_model.json'),'w') as f:json.dump(model,f,indent=2)
    # Test-only plot
    names=list(errors);vals=[errors[k][te] for k in names]
    if plt is not None:
        fig,ax=plt.subplots(figsize=(8,4.5));ax.boxplot(vals,showfliers=False); ax.set_xticks([1,2,3]); ax.set_xticklabels(['uncorrected','global mean','occupancy-only'])
        ax.axhline(0,color='black',lw=1);ax.set_ylabel('held-out occupancy volume error (%)')
        ax.set_title('Internal test after exploratory development');fig.tight_layout()
        fig.savefig(os.path.join(a.outdir,'validation_v3_test_errors.png'),dpi=180);plt.close(fig)
    L=['# Internal Validation Following Exploratory Development','',
       'The earlier correction percentages were computed on the same instances used to fit the correction. This analysis uses a stratified 60/20/20 train/calibration/test split. Model parameters are fitted on training rows; earlier full-shard exploration may have informed model design.','',
       '| Split | n |','|---|---:|']
    for z in ('train','calibration','test'):L.append('| %s | %d |'%(z,int((df.split==z).sum())))
    L += ['', '## Held-out test performance','',
          '| Method | mean bias (95% CI) | median absolute error (95% CI) | RMSE | 95% limits | calibration q95 | test coverage |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        s=summaries[name];L.append('| %s | %+.3f [%+.3f, %+.3f] | %.3f [%.3f, %.3f] | %.3f | %+.3f to %+.3f | %.3f | %.1f%% |'%
        (name,s['mean_bias'],*s['mean_bias_ci'],s['median_abs'],*s['median_abs_ci'],s['rmse'],s['loa_lo'],s['loa_hi'],s['calibration_q95_abs_error'],100*s['test_coverage_at_q95']))
    L += ['', '## Sensitivity to candidate error margins','', '| Method | <=2.17% | <=5% | <=6% | <=10% |','|---|---:|---:|---:|---:|']
    for name in names:
        w=summaries[name]['within_candidate_margins'];L.append('| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |'%(name,100*w['2.17'],100*w['5.0'],100*w['6.0'],100*w['10.0']))
    L += ['', '## Interpretation guardrails','',
          '- The calibrated q95 is an **in-domain measurement-error bound**, not a biological or functional threshold.',
          '- The 2.17% value is a Monte-Carlo repeatability coefficient estimated from stored query points; it is shown only as a sensitivity threshold.',
          '- The occupancy-only correction may be deployed without mesh measurements, but it still requires validation on another 3DMSL shard.',
          '- The proposed boundary-dilation mechanism remains a hypothesis until verified against the 3DMSL generation code.',
          '- MitoEM can test descriptor portability, but it cannot validate this occupancy correction because it lacks paired 3DMSL occupancy labels.','']
    with open(os.path.join(a.outdir,'validation_v3_report.md'),'w') as f:f.write('\n'.join(L))
    print('\n'.join(L));print('\nWrote validation_v3 outputs to',a.outdir)
if __name__=='__main__':main()
