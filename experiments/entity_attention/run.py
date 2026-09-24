"""Portable 01–10 launcher. Scientific interventions remain in the maintained main scripts."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
REVISION="49e3418fbbbca6ecbdf9608b4d22e5a407081db4"
def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(x) for x in p.read_text().splitlines()]
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n")
def tasks(config,out,model,device="cuda"):
    main=REPO/"scripts/inspect/qwen_entity_attention_ablation.py"
    profile=REPO/"scripts/inspect/qwen_prompt_attention_profile.py"
    n=config["experiment"]
    result=[]
    for g in config["templates"]:
        common=["--model",model,"--device",device,"--entities-json",str(HERE/"data/entities100.json"),
            "--template-ids",str(g["template_id"]),"--entity-ids",",".join(map(str,g["entity_ids"])),"--max-new-tokens","12"]
        if n in ["02","04"]:
            outdir=out/"runs"/g["template_name"]
            result.append(dict(phase=0,group=g,condition="profiles",expected=2*len(g["entity_ids"]),out=str(outdir),
                command=[sys.executable,str(profile)]+common+["--display-name-slots",str(config["display_name_slots"]),
                "--reference-dir",str(out/"reference_controls"),"--out-dir",str(outdir)]))
            continue
        desc=config.get("description")
        if desc:common+=["--entity-descriptions-json",str(HERE/"data"/f"descriptions_{desc}.json")]
        modes=[("ordinary",["--ordinary-baseline-only"],0,1)]
        if n in ["01","03","05","10"]:
            modes.append(("blocked",["--baseline-only","--block-intermediate-prompt-entity"],0,1))
        else:
            modes.append(("entity_blocked",["--widths","36","--start-layers","0","--skip-baselines"],1,1))
        if n in ["03","05"]:
            modes.append(("signed_restored",["--block-intermediate-prompt-entity","--restore-entity-attention-start-layer","0",
                "--restore-entity-attention-policy","signed"],1,1))
        if n=="10":
            modes.append(("sweep",["--block-intermediate-prompt-entity","--widths","4,6","--skip-baselines"],1,64))
        for cond,flags,phase,mult in modes:
            outdir=out/"runs"/f'{g["template_id"]:02d}_{g["template_name"]}'/cond
            result.append(dict(phase=phase,group=g,condition=cond,expected=mult*len(g["entity_ids"]),out=str(outdir),
                command=[sys.executable,str(main)]+common+flags+["--out-dir",str(outdir)]))
    return sorted(result,key=lambda x:x["phase"])
def audit_task(task,reference=None):
    rr=rows(Path(task["out"])/"generations.jsonl")
    assert len(rr)==task["expected"],(task["out"],len(rr),task["expected"])
    assert {r["entity_id"] for r in rr}==set(task["group"]["entity_ids"])
    cond=task["condition"]
    keys=set()
    for r in rr:
        key=(r["template_id"],r["entity_id"],r["condition"],r.get("width"),r.get("start_layer"))
        assert key not in keys,key
        keys.add(key)
        if cond=="profiles":continue
        ordinary=cond=="ordinary"
        intermediate=cond in ["blocked","signed_restored","sweep"]
        width=r["width"] if cond=="sweep" else (36 if cond=="entity_blocked" else 0)
        assert r["readout_attention_ablation_applications"]==width
        assert r["generated_attention_ablation_applications"]==(0 if ordinary else 36*max(len(r["generated_token_ids"])-1,0))
        assert r["block_intermediate_prompt_entity"]==intermediate
        assert (r["intermediate_attention_ablation_applications"]==0) if not intermediate else (r["intermediate_attention_ablation_applications"]>0)
        if cond=="sweep":assert width in [4,6] and r["disabled_layers"]==list(range(r["start_layer"],r["start_layer"]+width))
        if cond=="signed_restored":
            assert r["restore_policy"]=="signed" and r["restore_layers"]==list(range(36))
        if reference and cond in ["ordinary","blocked"]:
            g=task["group"]
            old={x["entity_id"]:x for x in rows(reference/"runs"/f'{g["template_id"]:02d}_{g["template_name"]}'/cond/"generations.jsonl")}
            assert all(r[k]==old[r["entity_id"]][k] for k in ["prompt","prompt_token_ids","entity_positions","completion","generated_token_ids","stop_reason"]),("control mismatch",key)
    return len(rr)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment",choices=[f"{i:02}" for i in range(1,11)])
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--results",type=Path,help="Private results repo, needed for 02–05 and 10 control parity")
    p.add_argument("--model",help="Local model snapshot; otherwise download the pinned revision")
    p.add_argument("--device",default="cuda")
    p.add_argument("--template-ids",help="Optional subset for splitting templates across independently launched jobs")
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args();config=read(HERE/"configs"/f"{a.experiment}.json")
    if a.template_ids:
        ids={int(x) for x in a.template_ids.split(",")}
        available={g["template_id"] for g in config["templates"]}
        if not ids<=available:p.error("Template is absent from this experiment cohort")
        config["templates"]=[g for g in config["templates"] if g["template_id"] in ids]
    out=a.out.resolve()
    if a.dry_run:
        plan=tasks(config,out,a.model or f"Qwen/Qwen3-8B-Base@{REVISION}",a.device)
        print(json.dumps({"expected_rows":sum(t["expected"] for t in plan),"tasks":plan},indent=2));return
    if out.exists():p.error("--out must be a fresh directory; no historical results are overwritten")
    reference=None
    if a.experiment in ["02","03","04","05","10"]:
        if not a.results:p.error("--results is required for saved-control verification")
        reference=a.results.resolve()/read(HERE/"configs/01.json")["result_directory"]
        if not (reference/"annotations.jsonl").exists():p.error("01 reference annotations are missing")
        expected="92e8b20cdca62533aa657a55e7c2cf012b89c707b944590ce1aec0331118b022"
        assert hashlib.sha256((reference/"annotations.jsonl").read_bytes()).hexdigest()==expected
    if a.model:model=str(Path(a.model).resolve()) if Path(a.model).exists() else a.model
    else:
        from huggingface_hub import snapshot_download
        model=snapshot_download("Qwen/Qwen3-8B-Base",revision=REVISION)
    out.mkdir(parents=True)
    if a.experiment in ["02","04"]:
        # Relocate model metadata in a derived reference view; retain original payloads and provenance.
        for g in config["templates"]:
            for cond in ["ordinary","blocked"]:
                rel=Path("runs")/f'{g["template_id"]:02d}_{g["template_name"]}'/cond
                src=reference/rel;dst=out/"reference_controls"/rel;dst.mkdir(parents=True)
                meta=read(src/"metadata.json")
                meta["reference_original_model"]=meta["model"];meta["model"]=model
                meta["reference_source_metadata_sha256"]=hashlib.sha256((src/"metadata.json").read_bytes()).hexdigest()
                dump(dst/"metadata.json",meta)
                shutil.copy2(src/"generations.jsonl",dst/"generations.jsonl")
    plan=tasks(config,out,model,a.device)
    dump(out/"run_plan.json",{"config":config,"model":model,"intended_revision":REVISION,
        "model_override":bool(a.model),"tasks":plan,"annotation_status":"not_reviewed"})
    (out/"logs").mkdir()
    for i,t in enumerate(plan):
        print(f'{i+1}/{len(plan)} {t["group"]["template_name"]} {t["condition"]}',flush=True)
        with (out/"logs"/f"{i:02}.log").open("w") as f:
            subprocess.run(t["command"],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True)
        audit_task(t,reference)
    with (out/"generations.jsonl").open("w") as f:
        for t in plan:
            for row in rows(Path(t["out"])/"generations.jsonl"):
                row["portable_run_condition"]=t["condition"];f.write(json.dumps(row,ensure_ascii=False)+"\n")
    dump(out/"audit.json",{"status":"passed","rows":sum(t["expected"] for t in plan),
        "saved_controls_checked":bool(reference),"annotation_status":"not_reviewed",
        "note":"New output categories require complete-continuation review; no automatic semantic scoring."})
if __name__=="__main__": main()
