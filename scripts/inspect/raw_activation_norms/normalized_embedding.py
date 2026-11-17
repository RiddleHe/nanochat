"""Measure actual post-RMSNorm embedding norms, weighted by full-validation counts."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .common import read_json, sha256, verify_run, write_json


def compute(run, output):
    import numpy as np
    import torch
    m = verify_run(run)
    assert m['config'].get('model_type','gpt_base') == 'gpt_base'
    sys.path.insert(0,str(run/'code'))
    from nanochat.checkpoint_manager import build_model
    from nanochat.model.gpt_base import norm
    import nanochat
    assert Path(nanochat.__file__).resolve().parent == run/'code/nanochat'
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    model,_,_=build_model(str(run/'inputs/checkpoint'),m['config']['checkpoint_step'],torch.device('cuda:0'),'eval')
    model.requires_grad_(False)
    counts=np.load(run/'inputs/token_frequencies.npy')
    assert counts.sum()==m['tokenization']['tokens']
    with torch.inference_mode():
        table=model.transformer.wte.weight.to(torch.bfloat16)
        values=norm(table).float().norm(dim=-1)
        ids=np.memmap(run/'inputs/token_ids.u32',mode='r',dtype='<u4')
        checks=[]
        for start in (0,(len(ids)//2048-1)*2048):
            tokens=torch.from_numpy(np.array(ids[start:start+2048],dtype=np.int64)).unsqueeze(0).cuda()
            captured=[]
            handle=model.transformer.h[0].register_forward_pre_hook(lambda module,args: captured.append(args[0].float().norm(dim=-1)))
            try: model(tokens)
            finally: handle.remove()
            assert len(captured)==1 and torch.equal(captured[0],values[tokens])
            checks.append(dict(start=start,tokens=2048,exact_first_block_input_norm_match=True))
        per_vocab=values.cpu().double().numpy()[:len(counts)]
    mean=float(counts@per_vocab/counts.sum())
    variance=float(counts@np.square(per_vocab-mean)/counts.sum())
    assert np.isfinite(per_vocab).all() and (per_vocab>0).all()
    np.savez(output/'per_vocabulary.npz',norm=per_vocab,frequency=counts)
    report=dict(mean=mean,variance=variance,std=float(np.sqrt(variance)),count=int(counts.sum()),
                boundary='actual first block input after initial RMSNorm',forward_dtype='bfloat16',norm_dtype='float32',
                aggregation='frequency-weighted mean and population variance in float64',passed=True,checks=checks,
                epsilon=float(torch.finfo(torch.bfloat16).eps),torch=torch.__version__,cuda=torch.version.cuda,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],baseline_run=str(run),
                manifest_sha256=sha256(run/'manifest.json'),checkpoint_sha256=m['inputs_hashes']['checkpoint/model_007809.pt'],
                frequencies_sha256=m['inputs_hashes']['token_frequencies.npy'],per_vocabulary_sha256=sha256(output/'per_vocabulary.npz'))
    write_json(output/'statistics.json',report)
    print(f'Mean={mean:.9f}; SD={report["std"]:.9f}; count={report["count"]}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--worker-output',type=Path)
    a=p.parse_args();run=a.run_dir.resolve()
    if a.worker_output:
        output=a.worker_output.resolve();job=read_json(output/'job.json')
        assert all(sha256(output/'code/scripts/inspect/raw_activation_norms'/name)==digest for name,digest in job['code_hashes'].items())
        assert os.environ['CUDA_VISIBLE_DEVICES']==job['gpu_uuid']
        compute(run,output);return
    description=subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    fields=[s.strip() for s in description.split(',')]
    assert int(fields[2])<1000 and int(fields[3])<=5,description
    output=run/'derived'/('normalized_embedding_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    code=output/'code/scripts/inspect/raw_activation_norms';code.mkdir(parents=True,exist_ok=False)
    for source in (Path(__file__),Path(__file__).with_name('common.py')):shutil.copy2(source,code/source.name)
    env=dict(os.environ,**read_json(run/'manifest.json')['environment_overrides']);env['CUDA_VISIBLE_DEVICES']=fields[1]
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','PYTHONPATH'):env.pop(key,None)
    write_json(output/'job.json',dict(baseline_run=str(run),gpu_uuid=fields[1],environment_overrides={k:v for k,v in env.items() if k in read_json(run/'manifest.json')['environment_overrides']},code_hashes={p.name:sha256(p) for p in code.iterdir()}))
    subprocess.run([sys.executable,'-m','scripts.inspect.raw_activation_norms.normalized_embedding','--run-dir',str(run),'--worker-output',str(output)],cwd=output/'code',env=env,check=True)
    print(output)


if __name__=='__main__':main()
