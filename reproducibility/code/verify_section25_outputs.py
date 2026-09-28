"""Reaggregate saved Section 25 outputs; does not rerun rendering or containment."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--package',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args();P=a.package
src=P/'data/section25_original';out=P/'audit';out.mkdir(exist_ok=True)
d=pd.read_csv(src/'mesh_check/recreation_per_condition.csv',dtype={'id':str});proto=json.loads((src/'mesh_check/protocol.json').read_text());ids=proto['ids']
expected=json.loads((P/'data/section24_original/recreation_full/protocol.json').read_text())['ids'][5:15]
assert ids==expected and len(set(ids))==10
assert len(d)==40 and not d.duplicated(['id','convention','offset_vox']).any()
assert set(zip(d.id,d.convention,d.offset_vox))=={(i,c,o) for i in ids for c in ['centred','literal'] for o in [0.,1.5]}
c=d[d.convention=='centred'].copy();assert np.isfinite(c[['mesh_occ_err_pct','occ_err_pct','mesh_band_points','mesh_vs_trilinear_agree_near','mesh_agree_near_0.01']].to_numpy()).all()
assert d[d.convention=='literal'].mesh_occ_err_pct.isna().all()
c['diff']=c.mesh_occ_err_pct-c.occ_err_pct
S={str(o):{'n':len(g),'median_vol_diff_pp_mesh_minus_trilinear':float(g['diff'].median()),'max_abs_vol_diff_pp':float(g['diff'].abs().max()),'median_label_agreement_near_surface':float(g.mesh_vs_trilinear_agree_near.median()),'median_agree_released_trilinear':float(g['agree_near_0.01'].median()),'median_agree_released_mesh':float(g['mesh_agree_near_0.01'].median())} for o,g in c.groupby('offset_vox')}
x=c[c.offset_vox==1.5].set_index('id').loc[ids];y=c[c.offset_vox==0].set_index('id').loc[ids]
S['paired_change_1.5_minus_0']={'trilinear_mean':float((x.occ_err_pct-y.occ_err_pct).mean()),'mesh_mean':float((x.mesh_occ_err_pct-y.mesh_occ_err_pct).mean())}
def diff(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b);return max(diff(a[k],b[k]) for k in a)
 return abs(float(a)-float(b))
saved=json.loads((src/'mesh_check/mesh_check_summary.json').read_text());notebook=json.loads((P/'data/section25_recorded/recorded_containment_summary.json').read_text())
assert diff(S,saved)<1e-10 and diff(S,notebook)<1e-10
old=pd.read_csv(P/'data/section24_original/recreation_full/recreation_per_condition.csv',dtype={'id':str});join=d.merge(old,on=['id','convention','offset_vox'],suffixes=('','_old'),validate='one_to_one');assert len(join)==40
interp_delta=float(abs(join.occ_err_pct-join.occ_err_pct_old).max());assert interp_delta<1e-10
mt=pd.read_csv(src/'model_sensitivity_v2/model_sensitivity_table.csv');local=pd.read_csv(P/'data/section25_model_mse_recheck/model_sensitivity_table.csv');j=mt.merge(local,on=['model','evaluation'],suffixes=('','_local'),validate='one_to_one');assert len(j)==10
cols=['bias','mdape','rmse','bound','coverage','coverage_low_occ','coverage_high_occ'];model_delta={k:float(abs(j[k]-j[k+'_local']).max()) for k in cols};assert max(model_delta.values())<1e-9
verify={'n_objects':10,'n_condition_records':40,'n_hybrid_condition_records':20,'expected_nonpilot_ids_match':True,'max_summary_difference':diff(S,saved),'max_notebook_summary_difference':diff(S,notebook),'max_section24_interpolation_difference_pp':interp_delta,'model_comparison_max_differences':model_delta,'hybrid_positive_offset_pairs':int(((x.mesh_occ_err_pct-y.mesh_occ_err_pct)>0).sum()),'limitation':'Independent reaggregation of supplied outputs, not a fresh rendering/fusion or full containment rerun.'}
(out/'section25_verified_summary.json').write_text(json.dumps(S,indent=2));(out/'section25_verification.json').write_text(json.dumps(verify,indent=2));(out/'section25_input_sha256.json').write_text(json.dumps({str(f.relative_to(src)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(src.rglob('*')) if f.is_file()},indent=2));print(json.dumps(verify,indent=2));print(json.dumps(S,indent=2))
