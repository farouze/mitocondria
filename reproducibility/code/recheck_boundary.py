"""Replay the same 60 IDs, retaining the archived acceptance decisions separately."""
from pathlib import Path
import sys, json, re, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent/'original_extracted'))
import e5_boundary_refit as e5

def one(item):
    j,iid,geometry_root=item
    r=e5.process(str(Path(geometry_root)/iid),.02,3000,j,32)
    r['id']=iid
    r['original_accepted']=iid not in {'25111','26173','26662'}
    return r

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser()
    ap.add_argument('--geometry-root',type=Path,required=True)
    ap.add_argument('--ids',type=Path,default=root/'data/original_boundary_ids.json')
    ap.add_argument('--out',type=Path,default=root/'data/boundary')
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    ids=json.loads(args.ids.read_text())
    rows=[]
    with ProcessPoolExecutor(args.workers) as ex:
        fs=[ex.submit(one,(j,iid,args.geometry_root)) for j,iid in enumerate(ids)]
        for f in as_completed(fs):
            rows.append(f.result());pd.DataFrame(rows).to_csv(out/'replayed_fits.csv',index=False)
            print(f'{len(rows)}/60 boundary fits replayed',flush=True)
    df=pd.DataFrame(rows).sort_values('id');df.to_csv(out/'replayed_fits.csv',index=False)
    results={}
    for name,sub in [('original_accepted',df[df.original_accepted]),('local_accepted',df[df.exact_converged])]:
        results[name]=e5.summarize(sub,'exact_t50','exact_w')
    results['local_acceptance_differs_from_archived_ids']=df.loc[df.original_accepted!=df.exact_converged,'id'].tolist()
    (out/'summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
