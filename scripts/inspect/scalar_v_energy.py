"""Freeze and collect all-layer ScalarRes V energies on the saved validation tokens."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def prepare(parent,gpu,output):
    m=read(parent/'manifest.json')
    assert m['config']['model_type']=='gpt_base_scalar_res_qkv_no_embed_norm'
    assert read(parent/'validation/summary.json')['passed']
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('code','inputs','tokens','derived','validation'): (output/folder).mkdir()
    for rel,digest in m['source_hashes'].items():
        src=parent/'code'/rel
        assert sha(src)==digest
        dst=output/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    script=output/'code/collect_scalar_v_energy.py';shutil.copy2(__file__,script)
    for name in ('tokens.npy','source_index.json','checkpoint_metadata.json'):
        shutil.copy2(parent/'inputs'/name,output/'inputs'/name)
    assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
    assert sha(output/'inputs/tokens.npy')==m['tokens_sha256']
    env=dict(m['environment_overrides'],CUDA_VISIBLE_DEVICES=gpu)
    manifest=dict(parent=str(parent),parent_manifest_sha256=sha(parent/'manifest.json'),
        parent_results_sha256=sha(parent/'results.json'),checkpoint_path=m['checkpoint_path'],
        checkpoint_sha256=m['checkpoint_sha256'],checkpoint_step=m['checkpoint_step'],
        tokens_sha256=m['tokens_sha256'],tokens_shape=m['tokens_shape'],model_type=m['config']['model_type'],
        source_hashes={**m['source_hashes'],'collect_scalar_v_energy.py':sha(script)},
        input_hashes={p.name:sha(p) for p in (output/'inputs').iterdir()},
        gpu_uuid=gpu,environment_overrides=env,python_executable=m['python_executable'],
        layers=list(range(24)),heads=12,output_dimensions=128,storage='per-token raw squared V-output norms, FP32',
        metric='q_s=e_s/mean_all_available_sources(e), per token/head before averaging',
        modes=['raw','scalar_weighted'],weighted='raw energy times actual BF16 coefficient squared; common RMSNorm cancels',
        denominator='Emb plus every prior attention and MLP source, including zero energies',ddof=0)
    write(output/'manifest.json',manifest)
    command=[m['python_executable'],str(script),'--collect','--output',str(output)]
    write(output/'launch.json',dict(command=command,cwd=str(output/'code'),environment_overrides=env))
    with (output/'collection.log').open('w') as log:
        p=subprocess.Popen(command,cwd=output/'code',env=dict(os.environ,**env),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(output/'process.json',dict(pid=p.pid))
    print(json.dumps(dict(run=str(output),pid=p.pid)),flush=True)


def collect(run):
    import numpy as np
    import torch
    m=read(run/'manifest.json');parent=Path(m['parent']);pm=read(parent/'manifest.json');pr=read(parent/'results.json')
    assert Path(__file__).resolve()==run/'code/collect_scalar_v_energy.py'
    for rel,digest in m['source_hashes'].items():assert sha(run/'code'/rel)==digest,rel
    for name,digest in m['input_hashes'].items():assert sha(run/'inputs'/name)==digest
    assert sha(parent/'manifest.json')==m['parent_manifest_sha256'] and sha(parent/'results.json')==m['parent_results_sha256']
    assert sha(m['checkpoint_path'])==m['checkpoint_sha256']
    for key,value in m['environment_overrides'].items():assert os.environ.get(key)==value,key
    for path,digest in pm['tokenizer_sha256'].items():assert sha(path)==digest
    for path,digest in pm['kernel_cache_sha256'].items():assert sha(Path(pm['kernel_cache_path'])/path)==digest
    assert torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(42);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from nanochat.checkpoint_manager import build_model
    from scripts.inspect.qkv_input_alignment.scalar_res import capture_sources
    import nanochat
    assert Path(nanochat.__file__).resolve().parent==run/'code/nanochat'
    write(run/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),gpu_uuid=m['gpu_uuid'],tf32=False))
    model,_,_=build_model(str(Path(m['checkpoint_path']).parent),m['checkpoint_step'],torch.device('cuda:0'),'eval')
    model.requires_grad_(False)
    tokens=np.load(run/'inputs/tokens.npy');assert list(tokens.shape)==[128,2048]
    coefficient_schema=read(parent/'summaries/scalar_coefficients.json')
    assert sha(parent/'summaries/scalar_coefficients.npz')==coefficient_schema['statistics_sha256']
    coefficients={}
    with np.load(parent/'summaries/scalar_coefficients.npz') as z:
        for row in coefficient_schema['receivers']:
            if row['kind']=='V':coefficients[row['layer']]=z['coefficients_bf16_as_fp32'][row['start']:row['stop']].astype(np.float64)
    np.savez(run/'inputs/v_coefficients.npz',**{f'V_{l:02d}':v for l,v in coefficients.items()})
    weights=[b.attn.c_v.weight.detach().to(torch.bfloat16).float() for b in model.transformer.h]
    moments={};n_total=0;chunk_hashes={};source_error=0.;fp64_error=0.;identity_error=0.;start=time.monotonic()
    def merge(key,a,old_n):
        mu=a.mean(0);m2=((a-mu)**2).sum(0);n=a.shape[0]
        if key not in moments:moments[key]=[mu,m2]
        else:
            old_mu,old_m2=moments[key];delta=mu-old_mu;total=old_n+n
            moments[key]=[old_mu+delta*n/total,old_m2+m2+delta**2*old_n*n/total]
    with torch.inference_mode():
        gates=model._sqrt_gate(model.scalar_res_logits_flat[:model.scalar_res_active])
        for l in range(24):
            lo,hi=model.scalar_res_offsets[4*l+2:4*l+4]
            assert np.array_equal(gates[lo:hi].to(torch.bfloat16).float().cpu().numpy(),coefficients[l])
        for i,sequence in enumerate(tokens):
            ids=torch.from_numpy(sequence.copy()).unsqueeze(0).to('cuda:0')
            baseline=model(ids)
            sources,logits=capture_sources(model,ids)
            assert torch.equal(baseline,logits) and torch.isfinite(logits).all()
            del baseline,logits
            assert torch.equal(sources[0],model.transformer.wte(ids).to(torch.bfloat16).reshape(-1,1536))
            ref=parent/'tokens'/f'chunk_{i:05d}.npz'
            assert sha(ref)==pr['chunk_sha256'][ref.name]
            raw=torch.stack([s.float().norm(dim=-1) for s in sources],-1).cpu().numpy()
            with np.load(ref) as z:
                np.testing.assert_allclose(raw,z['source_norm'],rtol=2e-5,atol=1e-6)
                source_error=max(source_error,float(np.max(np.abs(raw-z['source_norm'])/np.maximum(z['source_norm'],1e-6))))
            chunk={'source_norm':raw}
            for l in range(24):
                count=2*l+1;pieces=[]
                for s in range(0,count,4):
                    x=torch.stack([v.float() for v in sources[s:min(s+4,count)]])
                    projection=(x@weights[l].T).reshape(x.shape[0],2048,12,128)
                    e=projection.square().sum(-1).permute(1,2,0)
                    if i==0:
                        # Independent FP64 per-head projection on four real tokens, every source/layer.
                        reference=torch.einsum('std,hkd->thsk',x[:,:4].double(),weights[l].reshape(12,128,1536).double()).square().sum(-1)
                        error=float(((e[:4].double()-reference).abs()/reference.clamp_min(1e-18)).max())
                        fp64_error=max(fp64_error,error);assert error<2e-5
                        torch.testing.assert_close(projection.square().sum((-1,-2)),e.permute(2,0,1).sum(-1),rtol=2e-6,atol=1e-5)
                    pieces.append(e.cpu().numpy())
                energy=np.concatenate(pieces,axis=2)
                assert energy.shape==(2048,12,count) and np.isfinite(energy).all() and (energy>=0).all()
                chunk[f'raw_energy_V_{l:02d}']=energy
                energy=energy.astype(np.float64)
                merge(f'raw_energy_V_{l:02d}',energy,n_total)
                for mode,energies in [('raw',energy),('scalar_weighted',energy*coefficients[l][None,None,:]**2)]:
                    denominator=energies.mean(-1,keepdims=True);assert (denominator>0).all()
                    q=energies/denominator
                    err=float(np.abs(q.mean(-1)-1).max());identity_error=max(identity_error,err);assert err<1e-12
                    merge(f'{mode}_relative_energy_V_{l:02d}',q,n_total)
            path=run/'tokens'/f'chunk_{i:05d}.npz';np.savez(path,**chunk);chunk_hashes[path.name]=sha(path)
            n_total+=len(sequence)
            write(run/'status.json',dict(state='collecting',completed=i+1,total=128,tokens=n_total,elapsed_seconds=time.monotonic()-start))
            print(f'{i+1}/128 sequences; {time.monotonic()-start:.1f}s; source error {source_error:.3g}',flush=True)
    arrays={}
    for key,(mu,m2) in moments.items():
        variance=m2/n_total
        arrays.update({key+'_mean':mu,key+'_variance':variance,key+'_std':np.sqrt(variance),key+'_valid_count':np.full(mu.shape,n_total,dtype=np.int64)})
    assert n_total==262144
    np.savez_compressed(run/'derived/statistics.npz',**arrays)
    write(run/'derived/schema.json',dict(layers=list(range(24)),heads=12,tokens=n_total,source_axis='Emb,A0,M0,...',modes=m['modes'],formula=m['metric'],denominator=m['denominator'],ddof=0,square_root=False,top_k=False,output_coordinates=128,normalization_order='per token/head, then token mean; curves equal head mean',weighted=m['weighted']))
    write(run/'validation/summary.json',dict(passed=True,exact_uninstrumented_logits_all_sequences=True,source_bank_identity=True,scalar_receiver_order=True,embedding_identity=True,matched_saved_source_norms=True,source_norm_max_relative_error=source_error,fp64_projection_max_relative_error=fp64_error,fp64_validation_tokens=4,fp64_validation_layers=24,source_mean_ratio_identity_error=identity_error,positive_denominators_all_tokens_heads_layers_modes=True,tokens=n_total))
    write(run/'results.json',dict(state='complete',manifest_sha256=sha(run/'manifest.json'),tokens=n_total,chunk_sha256=chunk_hashes,output_sha256={name:sha(run/name) for name in ('derived/statistics.npz','derived/schema.json','validation/summary.json','inputs/v_coefficients.npz','environment.json')}))
    write(run/'status.json',dict(state='complete',tokens=n_total,elapsed_seconds=time.monotonic()-start))
    print('Validated all layers and heads.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path);p.add_argument('--gpu-uuid');p.add_argument('--output',type=Path);p.add_argument('--collect',action='store_true');a=p.parse_args()
    if a.collect:collect(a.output.resolve())
    else:
        out=a.output or Path('/local-ssd/mh3897/analysis_runs/d24_scalar_res_v_energy')/('step7773_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        prepare(a.parent.resolve(),a.gpu_uuid,out.resolve())
