"""Verify archived file hashes and replay the supplied model without refitting."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
r=args.run;args.out.mkdir(parents=True,exist_ok=True)
proto=json.loads((r/'protocol.json').read_text())
checks={name:hashlib.sha256((r/'frozen_inputs'/name).read_bytes()).hexdigest()==h for name,h in proto['source_sha256'].items()}
assert all(checks.values()),checks
m=json.loads((r/'frozen_inputs/validation_v3_model.json').read_text());a=pd.read_csv(r/'audit.csv');p=pd.read_csv(r/'frozen_predictions.csv')
assert a.instance_id.is_unique and p.instance_id.is_unique and len(a)==len(p)==2728
assert set(a.instance_id)==set(p.instance_id)
d=a.merge(p,on='instance_id',suffixes=('','_stored'),validate='one_to_one').sort_values('instance_id')
E=d[['occ_extent_x','occ_extent_y','occ_extent_z']].to_numpy();V=d.occ_volume_est.to_numpy();B=E.prod(1)
X=np.column_stack([V/B,d.occ_fraction_inside,np.log(V),np.log(B),E.max(1)/E.min(1)])
b=np.array(m['coefficients']);eh=np.clip(b[0]+((X-np.array(m['feature_means']))/np.array(m['feature_scales']))@b[1:],-50,100)
result={'n':len(d),'recorded_input_hashes_match':checks,'model_sha256':hashlib.sha256((r/'frozen_inputs/validation_v3_model.json').read_bytes()).hexdigest(),'models':{}}
for kind,e in [('uncorrected',np.zeros(len(V))),('global_train_mean',np.full(len(V),m['global_train_mean_bias_pct'])),('occupancy_only_ols',eh)]:
 cv=V/(1+e/100);err=100*(cv/d.mesh_volume.to_numpy()-1);delta=float(np.max(np.abs(err-d[kind+'_error_pct'].to_numpy())))
 assert delta<1e-9,(kind,delta)
 q=m['calibration_bounds_pct'][kind];low=d.occ_fraction_inside.to_numpy()<.02
 result['models'][kind]={'max_abs_error_difference_pp':delta,'mdape':float(np.median(abs(err))),'rmse':float(np.sqrt(np.mean(err**2))),'bound_pct':q,'covered':int((abs(err)<=q).sum()),'low_covered':int((abs(err[low])<=q).sum()),'low_n':int(low.sum()),'high_covered':int((abs(err[~low])<=q).sum())}
result['limitation']='Matching archived hashes and predictions verifies artifact consistency, not every historical analytical decision or independent biological provenance.'
(args.out/'original_frozen_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
