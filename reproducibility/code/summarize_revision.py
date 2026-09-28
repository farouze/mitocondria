"""Reproduce reported results and generate auditable post-review diagnostics."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.stats import spearmanr
sys.path.insert(0,str(Path(__file__).parent/'original_extracted'))
import validation_v3 as v

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data';FIG=ROOT/'paper'/'figures'

def wilson(k,n):
    p=k/n;z=1.959963984540054;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [100*(mid-half),100*(mid+half)]

def evaluate(df,model):
    X,_,_=v.feature_matrix(df,np.array(model['feature_means']),np.array(model['feature_scales']))
    predictions={'uncorrected':np.zeros(len(df)),
                 'global_train_mean':np.full(len(df),model['global_train_mean_bias_pct']),
                 'occupancy_only_ols':v.predict_pct(X,np.array(model['coefficients']))}
    out=df[['instance_id','mesh_volume','occ_volume_est','occ_fraction_inside']].copy()
    for name,bias in predictions.items():
        out[name+'_volume']=v.corrected_volume(df,bias)
        out[name+'_error_pct']=v.signed_error(df,out[name+'_volume'])
    return out

def main():
    d=pd.read_csv(DATA/'development_geometry.csv');f=pd.read_csv(DATA/'second_batch_geometry.csv')
    model_path=DATA/'reconstructed_development'/'validation_v3_model.json'
    before=hashlib.sha256(model_path.read_bytes()).hexdigest();model=json.loads(model_path.read_text())
    dp=pd.read_csv(DATA/'reconstructed_development'/'validation_v3_predictions.csv')
    fp=evaluate(f,model);fp.to_csv(DATA/'reproduced_second_batch_predictions.csv',index=False)
    summary={};q=model['calibration_bounds_pct']['occupancy_only_ols']
    rng=np.random.default_rng(790)
    for name in model['calibration_bounds_pct']:
        e=fp[name+'_error_pct'].to_numpy();s=v.summarize(e,1000,790)
        s['bound']=model['calibration_bounds_pct'][name];s['covered']=int((abs(e)<=s['bound']).sum())
        bs=np.array([[np.mean(z),np.median(abs(z)),np.mean(abs(z)<=s['bound'])]
                     for z in (e[rng.integers(0,len(e),len(e))] for _ in range(1000))])
        ci=np.percentile(bs,[2.5,97.5],axis=0)
        s['mean_bias_ci']=ci[:,0].tolist();s['median_abs_ci']=ci[:,1].tolist();s['coverage_ci']=ci[:,2].tolist()
        s['coverage_pct']=100*s['covered']/len(e);summary[name]=s
    # Assert the advertised central performance reproduces before using derived plots.
    assert round(summary['occupancy_only_ols']['median_abs'],3)==.664
    assert round(summary['occupancy_only_ols']['rmse'],3)==1.743
    assert summary['occupancy_only_ols']['covered']==2512
    it=dp[dp.split.eq('test')].merge(d[['instance_id','occ_fraction_inside']],on='instance_id',validate='one_to_one')
    groups=[]
    for dataset,df,ecol in [('internal',it,'corrected_error_occupancy_only_ols'),('second_batch',fp,'occupancy_only_ols_error_pct')]:
        for name,mask in [('all',np.ones(len(df),bool)),('low',df.occ_fraction_inside.lt(.02)),('other',df.occ_fraction_inside.ge(.02))]:
            e=df.loc[mask,ecol].to_numpy();k=int((abs(e)<=q).sum());n=len(e)
            groups.append(dict(dataset=dataset,group=name,n=n,covered=k,failures=n-k,coverage_pct=100*k/n,
                               wilson_low=wilson(k,n)[0],wilson_high=wilson(k,n)[1],median_abs=float(np.median(abs(e)))))
    g=pd.DataFrame(groups);g.to_csv(DATA/'coverage_groups.csv',index=False)
    internal=g[g.dataset.eq('internal')].set_index('group');new=g[g.dataset.eq('second_batch')].set_index('group')
    counterfactual=sum(new.loc[k,'n']*internal.loc[k,'coverage_pct']/100 for k in ('low','other'))/len(fp)
    summary['coverage_decomposition']={'internal_pct':float(internal.loc['all','coverage_pct']),
        'mixture_only_counterfactual_pct':100*counterfactual,'second_batch_pct':float(new.loc['all','coverage_pct']),
        'unflagged_share_of_failures_pct':100*float(new.loc['other','failures']/new.loc['all','failures'])}
    a=d.mesh_volume.to_numpy();b=d.occ_volume_est.to_numpy();delta=b-a;means=(a+b)/2
    summary['agreement']={'bias':float(delta.mean()),'mean_mesh':float(a.mean()),'mean_pair':float(means.mean()),
        'loa':(delta.mean()+np.array([-1,1])*1.96*delta.std(ddof=1)).tolist(),
        'pe_pair_pct':float(100*1.96*delta.std(ddof=1)/means.mean()),
        'pe_mesh_pct':float(100*1.96*delta.std(ddof=1)/a.mean()),
        'rho_difference_pairmean':float(np.corrcoef(delta,means)[0,1])}
    mechanism=[]
    for df in (d,f):
        df['error_pct']=100*(df.occ_volume_est/df.mesh_volume-1)
        df['predictor']=df.mesh_area*df.occ_scale/df.mesh_volume
    x=d.predictor.to_numpy();y=d.error_pct.to_numpy();slope=float(x@y/(x@x))
    for name,df in [('development',d),('second_batch',f)]:
        pred=slope*df.predictor;err=df.error_pct-pred
        mechanism.append(dict(dataset=name,slope_pct=slope,implied_tnorm=slope/100,
                              mean_residual_pp=float(err.mean()),rmse_pp=float(np.sqrt(np.mean(err**2))),
                              r2_fixed_model=float(1-np.sum(err**2)/np.sum((df.error_pct-df.error_pct.mean())**2))))
    pd.DataFrame(mechanism).to_csv(DATA/'mechanistic_baseline.csv',index=False)
    summary['known_query_box_sensitivity']={name:float(np.median(100*(df.occ_volume_known_box/df.occ_volume_est-1))) for name,df in [('development',d),('second_batch',f)]}
    summary['exact_mesh_file_hash_overlap']=len(set(d.mesh_sha256)&set(f.mesh_sha256))
    # Figures expose every tail observation; no truncation of data.
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.5,2.7))
    for name,label,color in [('uncorrected','Uncorrected','#929292'),('global_train_mean','Global','#c27635'),('occupancy_only_ols','Occupancy-only','#286284')]:
        vals=np.sort(np.abs(fp[name+'_error_pct']))
        ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=label,color=color,lw=1)
    ax.axvline(q,color='#333333',ls='--',lw=.8,label='2.661% bound')
    ax.axhline(.95,color='#777777',ls=':',lw=.8)
    ax.set(xscale='symlog',xlim=(0,None),ylim=(0,1.02),xlabel='Absolute volume error (%)',ylabel='Fraction within error')
    ax.legend(fontsize=7,loc='lower right');fig.tight_layout();fig.savefig(FIG/'frozen_ecdf.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.1,2.6))
    axs[0].scatter(means,delta,s=3,alpha=.35,rasterized=True)
    for y0 in [delta.mean(),*summary['agreement']['loa']]:axs[0].axhline(y0,color='#333333',ls='--',lw=.7)
    axs[0].set(xlabel='Mean volume (cubic mesh units)',ylabel='Occupancy - mesh volume')
    axs[0].xaxis.set_major_locator(MaxNLocator(4))
    axs[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)
    rel=100*(b/a-1);axs[1].scatter(a,rel,s=3,alpha=.35,rasterized=True)
    axs[1].axhline(0,color='#333333',lw=.7)
    axs[1].set(xscale='log',xlabel='Mesh volume (cubic mesh units; log scale)',ylabel='Relative error (%)')
    fig.tight_layout();fig.savefig(FIG/'agreement_diagnostics.pdf');plt.close(fig)
    bins=pd.cut(f.occ_fraction_inside,[0,.005,.01,.02,.05,.1,.2,1],include_lowest=True)
    f['sampling_sd_pct']=100*np.sqrt((1-f.occ_fraction_inside)/(f.occ_fraction_inside*f.occ_n_points))
    f.groupby(bins,observed=True).agg(n=('instance_id','size'),median_sampling_sd_pct=('sampling_sd_pct','median')).to_csv(DATA/'sampling_by_occupancy.csv')
    assert before==hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary['model_sha256']=before
    summary['provenance']='Local deterministic reconstruction with notebook seed 2026; matches reported primary metrics. Original frozen model file was not available for byte comparison.'
    (DATA/'verified_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print(g.to_string(index=False));print(pd.DataFrame(mechanism).to_string(index=False))

if __name__=='__main__':main()
