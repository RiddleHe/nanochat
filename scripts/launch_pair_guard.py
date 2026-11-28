"""Hold cooperative GPU locks and enforce leave-one-idle before starting a job."""
import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time

def snapshot():
    def query(args):
        return list(csv.reader(io.StringIO(subprocess.check_output(["nvidia-smi",*args,"--format=csv,noheader,nounits"],text=True))))
    rows=query(["--query-gpu=index,uuid,memory.used,utilization.gpu"])
    apps=query(["--query-compute-apps=gpu_uuid,pid,process_name"])
    occupied={r[0].strip() for r in apps}
    cards=[dict(index=int(r[0]),uuid=r[1].strip(),memory_mib=int(r[2]),util=int(r[3])) for r in rows]
    idle={r["index"] for r in cards if r["uuid"] not in occupied and r["memory_mib"]<32 and r["util"]==0}
    return dict(time=time.time(),cards=cards,processes=apps,idle=sorted(idle))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--gpus",required=True);ap.add_argument("--evidence",required=True)
    ap.add_argument("command",nargs=argparse.REMAINDER);a=ap.parse_args()
    gpus=[int(x) for x in a.gpus.split(",")];assert len(gpus)==len(set(gpus))
    before=snapshot();idle=set(before["idle"])
    if not set(gpus)<=idle or (len(idle)>1 and len(gpus)>=len(idle)):
        raise RuntimeError(f"Allocation unavailable or would take all idle GPUs: selected{gpus},idle{sorted(idle)}")
    handles=[]
    for gpu in sorted(gpus):
        path=f"/tmp/nanochat-looping-gpu{gpu}.lock"
        try: fd=os.open(path,os.O_RDONLY)
        except FileNotFoundError:
            old=os.umask(0)
            try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o666)
            finally:os.umask(old)
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(fd)
    after=snapshot();idle=set(after["idle"])
    if not set(gpus)<=idle or (len(idle)>1 and len(gpus)>=len(idle)):raise RuntimeError("GPU state changed while taking locks")
    evidence=Path(a.evidence);evidence.parent.mkdir(parents=True,exist_ok=True)
    evidence.write_text(json.dumps(dict(before=before,after=after,selected=gpus,pid=os.getpid()),indent=2)+"\n")
    env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=",".join(map(str,gpus))
    command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command:raise RuntimeError("No worker command")
    result=subprocess.run(command,env=env)
    for fd in handles:os.close(fd)
    raise SystemExit(result.returncode)

if __name__=="__main__":main()
