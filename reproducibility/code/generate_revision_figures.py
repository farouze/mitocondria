from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import pearsonr,spearmanr
import json
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'paper'/'figures'
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
all_fits=pd.read_csv(ROOT/'data/boundary/replayed_fits.csv')
df=all_fits[all_fits.original_accepted];rejected=all_fits[~all_fits.original_accepted]
reduced=df.drop(df.t_vol.idxmax())
influence={'omitted_id':int(df.loc[df.t_vol.idxmax(),'id']),'primary_n':len(df),'sensitivity_n':len(reduced),'pearson_primary':float(pearsonr(df.t_vol,df.exact_t50).statistic),'pearson_sensitivity':float(pearsonr(reduced.t_vol,reduced.exact_t50).statistic),'spearman_primary':float(spearmanr(df.t_vol,df.exact_t50).statistic),'spearman_sensitivity':float(spearmanr(reduced.t_vol,reduced.exact_t50).statistic),'note':'Exploratory influence diagnostic only; no primary observation excluded.'}
(ROOT/'data/boundary/influence_check.json').write_text(json.dumps(influence,indent=2))
assert len(df)==57
x=np.linspace(-.004,.009,500)
fig,ax=plt.subplots(1,2,figsize=(7.1,2.7))
for r in df.itertuples():ax[0].plot(x,expit(-(x-r.exact_t50)/r.exact_w),color='#286284',alpha=.14,lw=.6)
ax[0].plot(x,expit(-(x-df.exact_t50.median())/df.exact_w.median()),color='#183e59',lw=1.7)
ax[0].axvline(0,color='#333333',lw=.8);ax[0].axhline(.5,color='#999999',ls=':',lw=.6)
ax[0].set(xlabel='Signed distance from raw mesh (normalized)',ylabel='Fitted probability of inside label',ylim=(-.02,1.02))
ax[1].scatter(df.t_vol,df.exact_t50,s=12,color='#286284',label='Originally accepted (57)')
ax[1].scatter(rejected.t_vol,rejected.exact_t50,s=25,facecolors='none',edgecolors='#b06a30',label='Originally rejected (3)')
ax[1].legend(loc='upper left',fontsize=6.5,frameon=False)
lims=[.001,.008]
ax[1].plot(lims,lims,ls='--',color='#999999',lw=.8)
ax[1].set(xlim=lims,ylim=lims,xlabel='Volume-implied normalized displacement',ylabel='Label-transition midpoint')
fig.tight_layout();fig.savefig(FIG/'boundary_verified.pdf');fig.savefig(FIG/'boundary_verified.png',dpi=200);plt.close(fig)

rs=pd.read_csv(ROOT/'data/resolution/summary.csv')
primary=rs[(rs.phase_y==0)&(rs.phase_x==0)&rs.ties_foreground].sort_values('factor')
assert len(primary)==2
fig,axs=plt.subplots(1,3,figsize=(7.1,2.55),layout='constrained')
for ax,key,label in zip(axs,['volume','surface','sphericity'],['(a) Volume','(b) Surface area','(c) Sphericity']):
    med=primary[key+'_median'].to_numpy()
    lo=primary[key+'_p05'].to_numpy();hi=primary[key+'_p95'].to_numpy()
    ax.errorbar([16,24],med,yerr=[med-lo,hi-med],fmt='o',capsize=4,color='#28648a',lw=1.3)
    ax.axhline(0,color='#777777',ls='--',lw=.7)
    ax.set_xticks([16,24]);ax.set_xlabel('In-plane spacing (nm)');ax.set_title(label,fontsize=9)
    ax.grid(axis='y',alpha=.15);ax.set_xlim(12,31)
    for xx,yy in zip([16,24],med):
        ax.annotate(f'{yy:+.2f}%',(xx,yy),xytext=(5,5),textcoords='offset points',fontsize=7)
axs[0].set_ylabel('Change from native mesh (%)')
fig.savefig(FIG/'resolution_sensitivity.pdf',bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(6.4,3))
for f,tie,label,color in [(2,True,'16 nm; foreground ties','#286284'),(2,False,'16 nm; background ties','#b06a30'),(3,True,'24 nm; no ties','#667a61')]:
    g=rs[(rs.factor==f)&(rs.ties_foreground==tie)]
    xs=np.arange(len(g));ax.plot(xs,g.volume_median,'o-',label=label,color=color,lw=1,ms=4)
ax.axhline(0,color='#555555',ls=':',lw=.8);ax.set(xlabel='Grid phase index (row-major order)',ylabel='Median volume change (%)')
ax.legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/'data/resolution/phase_sensitivity.pdf');fig.savefig(FIG/'phase_sensitivity.pdf');plt.close(fig)
print('Regenerated accepted-fit boundary figure and phase diagnostic')
