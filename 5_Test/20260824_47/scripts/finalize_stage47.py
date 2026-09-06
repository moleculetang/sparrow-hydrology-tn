"""Aggregate completed Stage47 workers and apply registered spatial gates."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test/20260824_47"; WORK = RUN / "work"; OUT = RUN / "outputs"; REPORTS = RUN / "reports"; LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"; MANIFEST = RUN / "program_manifest.json"; PARENT_LOCK = ROOT / "5_Test/20260824_46/locks/stage46_lock.json"
MODELS = ["PARENT", "REG3_OBSERVATION", "DYN_HYDRO_DELIVERY", "MINERAL_LIFETIME", "AQ_SIZE"]; CANDIDATES = MODELS[1:]; MARGIN = 0.005; REPLICATES = 10000; SEED = 260847


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic_json(payload, path): path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_suffix(path.suffix+".part"); tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); os.replace(tmp,path)
def atomic_parquet(frame,path): path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".part"); frame.to_parquet(tmp,index=False); os.replace(tmp,path)


def comparison(candidate: pd.DataFrame, parent: pd.DataFrame, holdout_type: str):
    block_key = {"REACH":"reach_id","TREE":"terminal_tree_id","FIRST_OBSERVED_2021":"station_key"}[holdout_type]
    keys=["fold_id","holdout_type","holdout_id","station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]
    joined=candidate.loc[candidate.holdout_type.eq(holdout_type),keys+["pred_tn_mg_l","baseline_pred_tn_mg_l"]].merge(parent.loc[parent.holdout_type.eq(holdout_type),keys+["pred_tn_mg_l"]],on=keys,suffixes=("_candidate","_parent"),validate="one_to_one")
    rows=[]
    for block,group in joined.groupby(block_key):
        obs=np.log1p(group.tn_mg_l.to_numpy(float)); cand=np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)); par=np.log1p(group.pred_tn_mg_l_parent.to_numpy(float)); base=np.log1p(group.baseline_pred_tn_mg_l.to_numpy(float))
        rows.append({"block":str(block),"candidate_rmse":float(np.sqrt(np.mean((cand-obs)**2))),"parent_rmse":float(np.sqrt(np.mean((par-obs)**2))),"candidate_sse":float(np.sum((cand-obs)**2)),"baseline_sse":float(np.sum((base-obs)**2))})
    blocks=pd.DataFrame(rows); diff=blocks.candidate_rmse.to_numpy()-blocks.parent_rmse.to_numpy(); rng=np.random.default_rng(SEED+sum(map(ord,holdout_type))); delta=[]; skill=[]
    for _ in range(REPLICATES):
        take=rng.integers(0,len(blocks),len(blocks)); sample=blocks.iloc[take]; delta.append(float(diff[take].mean())); skill.append(float(1.0-sample.candidate_sse.sum()/sample.baseline_sse.sum()))
    dl,du=np.quantile(delta,[.025,.975]); sl,su=np.quantile(skill,[.025,.975]); point_skill=float(1.0-blocks.candidate_sse.sum()/blocks.baseline_sse.sum())
    return {"holdout_type":holdout_type,"blocks":len(blocks),"delta_candidate_minus_parent":float(diff.mean()),"delta_ci95_lower":float(dl),"delta_ci95_upper":float(du),"noninferior":bool(du<MARGIN),"improved":bool(du<0),"station_blind_skill_log":point_skill,"skill_ci95_lower":float(sl),"skill_ci95_upper":float(su),"positive_skill":bool(sl>0)}


def tree_sign_flip(candidate,parent):
    keys=["fold_id","holdout_type","holdout_id","station_key","reach_id","terminal_tree_id","year","month","tn_mg_l"]
    joined=candidate.loc[candidate.holdout_type.eq("TREE"),keys+["pred_tn_mg_l"]].merge(parent.loc[parent.holdout_type.eq("TREE"),keys+["pred_tn_mg_l"]],on=keys,suffixes=("_candidate","_parent"),validate="one_to_one"); values=[]
    for _,g in joined.groupby("terminal_tree_id"):
        obs=np.log1p(g.tn_mg_l); a=np.log1p(g.pred_tn_mg_l_candidate); b=np.log1p(g.pred_tn_mg_l_parent); values.append(float(np.sqrt(np.mean((a-obs)**2))-np.sqrt(np.mean((b-obs)**2))))
    values=np.asarray(values); perms=np.asarray([np.mean(values*np.asarray(signs)) for signs in itertools.product((-1.,1.),repeat=len(values))]); point=float(values.mean()); return {"trees":len(values),"permutations":len(perms),"observed_mean_delta":point,"two_sided_p":float(np.mean(np.abs(perms)>=abs(point)-1e-15))}


def main():
    for path in [OUT,REPORTS,LOCKS]: path.mkdir(parents=True,exist_ok=True)
    predictions=[]; parameters=[]
    for model in MODELS:
        pp=WORK/f"{model.lower()}_nested_predictions.parquet"; qp=WORK/f"{model.lower()}_nested_parameters.parquet"
        if not pp.exists() or not qp.exists(): raise RuntimeError(f"Missing worker outputs for {model}")
        p=pd.read_parquet(pp); q=pd.read_parquet(qp)
        if q.fold_id.nunique()!=331: raise RuntimeError(f"{model} has {q.fold_id.nunique()}/331 folds")
        predictions.append(p); parameters.append(q)
    prediction=pd.concat(predictions,ignore_index=True); parameter=pd.concat(parameters,ignore_index=True); parent=prediction.loc[prediction.candidate.eq("PARENT") & prediction.layer.eq("population_transferable")]
    comparisons={}; eligibility={}; sign_flips={}
    for candidate in CANDIDATES:
        selected=prediction.loc[prediction.candidate.eq(candidate)&prediction.layer.eq("population_transferable")]; rows=[comparison(selected,parent,kind) for kind in ["REACH","TREE","FIRST_OBSERVED_2021"]]; comparisons[candidate]=rows; by={r["holdout_type"]:r for r in rows}
        par=parameter.loc[parameter.candidate.eq(candidate)]; extra=[c for c in par.columns if c in {"gamma_reg_1","gamma_reg_2","gamma_reg_3","beta_D","log_tau_mineral_days","beta_aq_size"}]; boundary_ok=True
        # Bounds were already enforced; count exact-near-bound values using registered numerical ranges.
        bounds={"gamma_reg_1":(-.5,.5),"gamma_reg_2":(-.5,.5),"gamma_reg_3":(-.5,.5),"beta_D":(-1.,1.),"log_tau_mineral_days":(np.log(182.625),np.log(3652.5)),"beta_aq_size":(-.75,.75)}
        for holdout in ["REACH","TREE"]:
            group=par.loc[par.holdout_type.eq(holdout)]
            for name in extra:
                lo,hi=bounds[name]; rate=float(((group[name]<=lo+1e-5*(hi-lo))|(group[name]>=hi-1e-5*(hi-lo))).mean()); boundary_ok &= rate<=.05
        eligibility[candidate]=bool(by["REACH"]["noninferior"] and by["TREE"]["noninferior"] and by["REACH"]["positive_skill"] and by["TREE"]["positive_skill"] and boundary_ok)
        sign_flips[candidate]=tree_sign_flip(selected,parent)
    eligible=[c for c in CANDIDATES if eligibility[c]]; counts=parameter.groupby(["candidate","holdout_type"]).size().to_dict(); expected={"REACH":309,"TREE":21,"FIRST_OBSERVED_2021":1}; checks={"all_worker_counts_exact":all(all(counts.get((m,k),0)==v for k,v in expected.items()) for m in MODELS),"predictions_finite":bool(np.isfinite(prediction.loc[prediction.layer.eq("population_transferable"),"pred_tn_mg_l"]).all()),"heldout_conditional_unavailable":bool(prediction.loc[prediction.layer.eq("gauged_conditional"),"pred_tn_mg_l"].isna().all()),"tree163_absent":bool((prediction.terminal_tree_id!=163).all()),"keys_unique":not prediction.duplicated(["candidate","layer","fold_id","station_key","reach_id","terminal_tree_id","year","month"]).any()}; status="PASS_STAGE47_SPATIAL_VALIDATION" if all(checks.values()) else "FAIL_STAGE47_ENGINEERING"
    paths={"predictions":OUT/"stage47_nested_spatial_predictions.parquet","parameters":OUT/"stage47_nested_spatial_parameters.parquet","comparisons":OUT/"stage47_spatial_comparisons.parquet"}; atomic_parquet(prediction,paths["predictions"]); atomic_parquet(parameter,paths["parameters"]); atomic_parquet(pd.DataFrame([{"candidate":c,**r} for c,rows in comparisons.items() for r in rows]),paths["comparisons"])
    part=[str(p) for p in RUN.rglob("*.part")]; checks["no_part_files_after_write"]=len(part)==0
    if part: status="FAIL_STAGE47_ENGINEERING"; eligible=[]
    decision={"stage":"20260824_47","status":status,"eligible_candidates":eligible,"eligibility":eligibility,"comparisons":comparisons,"tree_sign_flip":sign_flips,"checks":checks,"input_hashes":{str(p):sha256(p) for p in [CONTRACT,MANIFEST,PARENT_LOCK]},"output_hashes":{k:sha256(p) for k,p in paths.items()},"authorized_successor":"20260824_48" if status.startswith("PASS") else None}; atomic_json(decision,REPORTS/"stage47_decision.json"); atomic_json({"stage":"20260824_47","status":status,"eligible_candidates":eligible,"decision_sha256":sha256(REPORTS/"stage47_decision.json"),"authorized_successor":decision["authorized_successor"]},LOCKS/"stage47_lock.json")
    lines=["# `20260824_47` nested spatial validation","",f"Status: `{status}`.","",f"Spatially eligible: `{eligible}`.","","| candidate | holdout | delta vs parent | CI95 | noninferior | skill | skill CI95 |","|---|---|---:|---:|---|---:|---:|"]
    for c,rows in comparisons.items():
        for r in rows: lines.append(f"| {c} | {r['holdout_type']} | {r['delta_candidate_minus_parent']:.5f} | {r['delta_ci95_lower']:.5f}–{r['delta_ci95_upper']:.5f} | {r['noninferior']} | {r['station_blind_skill_log']:.3f} | {r['skill_ci95_lower']:.3f}–{r['skill_ci95_upper']:.3f} |")
    lines += ["", "All held-out conditional predictions are unavailable by contract; selection uses population-transferable predictions only. Tree 163 remains outside the formal river gate."]
    (REPORTS/"technical_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps(decision,ensure_ascii=False,indent=2,default=str))
    if not status.startswith("PASS"): raise RuntimeError(status)


if __name__=="__main__": main()
