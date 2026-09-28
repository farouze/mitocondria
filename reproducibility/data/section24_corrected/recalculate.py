import csv,json,hashlib
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
SRC=args.input
OUT=args.out
OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:list(csv.DictReader(p.open()))
rows=read(SRC/'recreation_full/recreation_per_condition.csv')
pilot=read(SRC/'recreation_pilot/recreation_per_condition.csv')
ids=json.loads((SRC/'recreation_full/protocol.json').read_text())['ids']
pids=json.loads((SRC/'recreation_pilot/protocol.json').read_text())['ids']
key=lambda r:(r['id'],r['convention'],float(r['offset_vox']))
D={key(r):r for r in rows}
assert len(D)==len(rows)==360
assert set(D)=={(i,c,o) for i in ids for c in ['centred','literal'] for o in [0.,.75,1.5]}
assert len(set(ids))==60 and len(set(pids))==5 and set(pids)<=set(ids)
assert all(np.isfinite(float(r[k])) for r in rows for k in ['occ_err_pct','fused_mesh_err_pct','t50','w','agree_near_0.01'])
assert all(D[key(r)]==r for r in pilot), 'pilot/full differences'
def paired(ii,c,col,accept=False):
 if accept:ii=[i for i in ii if all(D[i,c,o]['t50_converged']=='True' for o in [0.,1.5])]
 d=np.array([float(D[i,c,1.5][col])-float(D[i,c,0.][col]) for i in ii])
 if not len(d):return {'n':0,'mean':None,'ci95_mean':None}
 rng=np.random.default_rng(0)
 b=[np.mean(rng.choice(d,len(d))) for _ in range(5000)]
 return dict(n=len(d),mean=float(d.mean()),median=float(np.median(d)),ci95_mean=np.percentile(b,[2.5,97.5]).tolist(),n_positive=int((d>0).sum()))
S={'status':'Retrospective corrected summary; pilot excluded after all-60 outcomes inspected. No new independent test set.', 'bootstrap':'5000 paired object resamples, percentile interval, seed 0; conditional on sampled objects.', 'pilot_ids':pids,'integrity':{'complete_unique_condition_rows':len(rows),'pilot_full_records_identical':True},'groups':{}}
for name,ii in [('nonpilot55',[i for i in ids if i not in pids]),('all60',ids)]:
 S['groups'][name]={}
 for c in ['centred','literal']:
  C={'volume_change_pp':paired(ii,c,'occ_err_pct'),'fused_mesh_change_pp':paired(ii,c,'fused_mesh_err_pct'),'t50_change_accepted_pairs':paired(ii,c,'t50',True),'conditions':{}}
  for o in [0.,.75,1.5]:
   rr=[D[i,c,o] for i in ii]; ok=[r for r in rr if r['t50_converged']=='True']
   C['conditions'][str(o)]={'n':len(rr),'median_error_pct':float(np.median([float(r['occ_err_pct']) for r in rr])),'t50_accepted':len(ok),'t50_failed':len(rr)-len(ok),'t50_accepted_median':float(np.median([float(r['t50']) for r in ok])) if ok else None,'near_agreement_median':float(np.median([float(r['agree_near_0.01']) for r in rr]))}
  C['strictly_increasing_objects']=sum(all(float(D[i,c,a]['occ_err_pct'])<float(D[i,c,b]['occ_err_pct']) for a,b in [(0.,.75),(.75,1.5)]) for i in ii)
  S['groups'][name][c]=C
with (OUT/'nonpilot55_per_condition.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(r for r in rows if r['id'] not in pids)
(OUT/'corrected_summary.json').write_text(json.dumps(S,indent=2))
(OUT/'input_sha256.json').write_text(json.dumps({str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SRC.rglob('*')) if p.is_file()},indent=2))
print(json.dumps(S,indent=2))
