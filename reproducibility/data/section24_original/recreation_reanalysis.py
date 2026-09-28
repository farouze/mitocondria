#!/usr/bin/env python3
"""Reanalysis of the E8 label-pipeline recreation (no new rendering or fusion).

Implements the corrections from the external audit of Section 24:
  * the 5 pilot objects are excluded from the PRIMARY analysis (55 objects);
    all 60 are reported as a sensitivity analysis. This exclusion is applied
    after the all-60 results were inspected, and is disclosed as such.
  * pilot and full-run rows for the 5 pilot objects must be identical
    (a determinism check);
  * both label conventions are reported;
  * t50 summaries use only pairs whose fits passed the acceptance checks in both
    conditions, with all-fit values as a sensitivity analysis;
  * every object must have all 6 condition rows and no error rows.
Bootstrap intervals are object-level (5,000 resamples, seed 0) and are
conditional on the sampled objects.
"""
import argparse, json
import numpy as np
import pandas as pd


def paired(df, col, a=1.5, b=0.0, accepted=False, n_boot=5000, seed=0):
    x = df[df.offset_vox == a].set_index("id"); y = df[df.offset_vox == b].set_index("id")
    ids = x.index.intersection(y.index)
    if accepted:
        ids = [i for i in ids if bool(x.loc[i, "t50_converged"]) and bool(y.loc[i, "t50_converged"])]
    d = (x.loc[ids, col] - y.loc[ids, col]).to_numpy(float)
    rng = np.random.default_rng(seed)
    bm = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return {"n": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
            "ci95_mean": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))], "n_positive": int((d > 0).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", required=True, help="recreation_full folder")
    ap.add_argument("--pilot", required=True, help="recreation_pilot folder")
    ap.add_argument("--out", required=True, help="output JSON")
    a = ap.parse_args()
    F = pd.read_csv(f"{a.full}/recreation_per_condition.csv", dtype={"id": str})
    P = pd.read_csv(f"{a.pilot}/recreation_per_condition.csv", dtype={"id": str})
    R = pd.read_csv(f"{a.full}/released_per_object.csv", dtype={"id": str}).set_index("id")

    checks = {"n_ids": int(F.id.nunique()), "rows_per_id": F.groupby("id").size().value_counts().to_dict(),
              "error_rows": int(F["error"].notna().sum()) if "error" in F else 0}
    assert set(checks["rows_per_id"]) == {6} and checks["error_rows"] == 0, checks
    pilot_ids = sorted(P.id.unique())
    m = P.merge(F, on=["id", "offset_vox", "convention"], suffixes=("_p", "_f"))
    checks["pilot_vs_full_max_abs_diff"] = {c: float(np.max(np.abs(m[c + "_p"] - m[c + "_f"])))
                                           for c in ("occ_err_pct", "fused_mesh_err_pct", "agree_near_0.01", "t50")}
    checks["pilot_ids"] = pilot_ids

    out = {"checks": checks}
    for name, ids in (("primary_55", [i for i in F.id.unique() if i not in pilot_ids]), ("sensitivity_all_60", list(F.id.unique()))):
        for conv in ("centred", "literal"):
            df = F[(F.convention == conv) & F.id.isin(ids)]
            r15 = df[df.offset_vox == 1.5].set_index("id")
            o = {"occ_err_1.5_minus_0": paired(df, "occ_err_pct"),
                 "fused_mesh_err_1.5_minus_0": paired(df, "fused_mesh_err_pct"),
                 "t50_1.5_minus_0_accepted_pairs": paired(df, "t50", accepted=True),
                 "t50_1.5_minus_0_all_fits": paired(df, "t50"),
                 "by_offset": {str(k): {"occ_err_median": float(g.occ_err_pct.median()),
                                        "t50_median_accepted": float(g[g.t50_converged == True].t50.median()),
                                        "n_t50_accepted": int((g.t50_converged == True).sum()), "n": int(len(g)),
                                        "agree_near_median": float(g["agree_near_0.01"].median())}
                               for k, g in df.groupby("offset_vox")},
                 "recreated_1.5_vs_released": {
                     "released_median": float(R.loc[ids, "occ_err_pct"].median()),
                     "recreated_median": float(r15.loc[ids, "occ_err_pct"].median()),
                     "median_abs_diff_pp": float((r15.loc[ids, "occ_err_pct"] - R.loc[ids, "occ_err_pct"]).abs().median()),
                     "pearson_r": float(np.corrcoef(r15.loc[ids, "occ_err_pct"], R.loc[ids, "occ_err_pct"])[0, 1])}}
            o["offset_share_of_recreated_mean_error"] = o["occ_err_1.5_minus_0"]["mean"] / float(r15.loc[ids, "occ_err_pct"].mean())
            out[f"{name}_{conv}"] = o
    json.dump(out, open(a.out, "w"), indent=1)
    p = out["primary_55_centred"]
    print(json.dumps(checks, indent=1))
    print(f"PRIMARY (55, centred): occupancy error change {p['occ_err_1.5_minus_0']['mean']:+.3f} pp "
          f"[{p['occ_err_1.5_minus_0']['ci95_mean'][0]:.3f}, {p['occ_err_1.5_minus_0']['ci95_mean'][1]:.3f}], "
          f"positive {p['occ_err_1.5_minus_0']['n_positive']}/{p['occ_err_1.5_minus_0']['n']}")


if __name__ == "__main__":
    main()
