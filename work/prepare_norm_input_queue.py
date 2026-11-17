"""Prepare, and optionally start, the normalized single-head availability queue."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,pwd,shutil,subprocess,time

REPO=Path(__file__).resolve().parents[1]
PYTHON='/hdd/mh3897/nanochat/.venv/bin/python'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',action='store_true')
    args=parser.parse_args()
    target=Path(json.loads((REPO/'work/norm-input-prepared-runs.json').read_text())['single_qk_norm_input'])
    manifest=json.loads((target/'manifest.json').read_text())
    assert manifest['model_config']['norm_value_input'] and not manifest['model_config']['sum_head_attention']
    assert manifest['validation']['status']=='passed'
    ids={}
    for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines():
        index,uid=line.split(',');ids[int(index)]=uid.strip()
    gpus=[{'index':i,'uuid':ids[i]} for i in (1,2)]
    occupants=[]
    for line in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines():
        uid,pid=[v.strip() for v in line.split(',')]
        if uid in {g['uuid'] for g in gpus}:
            try:owner=pwd.getpwuid(Path('/proc',pid).stat().st_uid).pw_name
            except FileNotFoundError:continue
            occupants.append({'gpu_uuid':uid,'pid':int(pid),'owner':owner})
    queue=Path('/local-ssd/mh3897/run_queues')/datetime.now(timezone.utc).strftime('single_norm_when_idle_1_2_%Y%m%dT%H%M%SZ')
    queue.mkdir()
    shutil.copy2(REPO/'work/queue_norm_input.py',queue/'queue.py')
    shutil.copy2(Path(__file__),queue/'prepare_queue.py')
    config={'target_run':str(target),'gpus':gpus,'occupants_at_queue_creation':occupants,'poll_seconds':60,
            'deadline_epoch':time.time()+7*86400,
            'target_hashes':{name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in
                             ('manifest.json','launch.py','launch.sh','launcher_core.py')}}
    (queue/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    subprocess.run([PYTHON,str(queue/'queue.py'),'--config',str(queue/'config.json'),'--check'],check=True)
    (REPO/'work/norm-input-queue.txt').write_text(str(queue)+'\n')
    if args.start:
        with (queue/'queue.log').open('x') as log:
            child=subprocess.Popen([PYTHON,'-u',str(queue/'queue.py'),'--config',str(queue/'config.json')],cwd=queue,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (queue/'pid').write_text(str(child.pid)+'\n')
        print(json.dumps({'queue':str(queue),'pid':child.pid,'target':str(target),'gpus':[1,2],'occupants':occupants}),flush=True)
    else:print(queue)

if __name__=='__main__':main()
