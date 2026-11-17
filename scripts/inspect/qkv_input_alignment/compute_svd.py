"""Compute head SVDs once from forward-cast weights; default CPU, no model replay."""
import argparse
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from .common import matrix_specs, now, read_json, require, sha256, stage_entry, verify_run, verify_svd, write_json, write_npz


def decompose(weight, thresholds, tie_rtol=1e-8, validation=None):
    """Compact SVD of [output,input]. Returned Vh rows are input directions."""
    a = weight.to(torch.float64)
    require(a.ndim == 2 and bool(torch.isfinite(a).all()), 'Invalid matrix')
    u,s,vh = torch.linalg.svd(a, full_matrices=False)
    norm = float(torch.linalg.vector_norm(a))
    reconstruction = float(torch.linalg.vector_norm((u*s)@vh-a)) / max(norm,np.finfo(float).tiny)
    # Only nonzero singular directions have meaning for selecting a strong subspace.
    numerical_rank = int((s > (max(a.shape)*torch.finfo(a.dtype).eps*s[0])).sum()) if len(s) else 0
    spectrum = s.cpu().numpy()
    cumulative = np.cumsum(spectrum**2)
    total = float(cumulative[-1])
    ks, captured = [], []
    for threshold in thresholds:
        if total == 0:
            ks.append(0); captured.append(0.0)
            continue
        k = min(int(np.searchsorted(cumulative/total, threshold, side='left'))+1, numerical_rank)
        cutoff = spectrum[k-1]
        while k < numerical_rank and abs(spectrum[k]-cutoff) <= tie_rtol*cutoff:
            k += 1
        ks.append(k); captured.append(float(cumulative[k-1]/total))
    stored = vh.to(torch.float32)
    orth = float((stored.double()@stored.double().T-torch.eye(len(s),device=a.device,dtype=a.dtype)).abs().max())
    if validation:
        require(reconstruction <= validation['svd_relative_error'], 'SVD reconstruction failed')
        require(orth <= validation['stored_basis_orthogonality_error'], 'Stored basis lost orthogonality')
    arrays = dict(sigma=spectrum, vh=stored.cpu().numpy(), k=np.array(ks,dtype=np.int64),
        captured_squared_fraction=np.array(captured), matrix_valid=np.array(total>0), rank=np.array(numerical_rank))
    return arrays, dict(reconstruction_relative_error=reconstruction, stored_basis_orthogonality_error=orth)


def compute(run, device='cpu'):
    run=Path(run)
    manifest, meta=verify_run(run,checkpoint=True)
    config=manifest['config']
    if (run/'svd/index.json').exists():
        verify_svd(run,manifest)
        print('Verified existing SVD bank; nothing recomputed.')
        return
    require(not any((run/'tokens').glob('chunk_*.json')), 'Cannot replace SVDs used by committed chunks')
    require(device in ('cpu','cuda'), 'Use cpu or the single selected cuda device')
    if device=='cuda':
        require(torch.cuda.device_count()==1,'Exactly one explicitly selected GPU required')
    torch.set_num_threads(config['cpu_threads'])
    state=torch.load(manifest['checkpoint_path'],map_location='cpu',weights_only=True,mmap=True)
    state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
    environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
        device=device,device_name=torch.cuda.get_device_name(0) if device=='cuda' else platform.processor(),svd_dtype='float64')
    path=run/'svd/environment.json'
    if path.exists(): require(read_json(path)==environment,'SVD resume environment changed')
    else: write_json(path,environment)
    files={}
    for layer,kind,heads,head_dim,width in matrix_specs(meta, config):
        name=f'{kind}_{layer:02d}'
        path=run/'svd'/f'{name}.npz'
        marker=path.with_suffix('.json')
        if marker.exists():
            rec=read_json(marker)
            require(rec['passed'] and rec['manifest_sha256']==sha256(run/'manifest.json') and rec['sha256']==sha256(path),'SVD resume mismatch')
            files[path.name]=rec['sha256']; continue
        key=f'transformer.h.{layer}.attn.c_{kind.lower()}.weight'
        weight=state[key].to(torch.bfloat16).to(torch.float64).to(device)
        require(tuple(weight.shape)==(heads*head_dim,width),f'Bad weight shape: {key}')
        banks=[];checks=[]
        for head in range(heads):
            arrays,check=decompose(weight[head*head_dim:(head+1)*head_dim],config['squared_energy_thresholds'],config['tie_rtol'],config['validation'])
            banks.append(arrays);checks.append(dict(head=head,**check))
        write_npz(path,{key:np.stack([b[key] for b in banks]) for key in banks[0]})
        files[path.name]=sha256(path)
        write_json(marker,dict(passed=True,manifest_sha256=sha256(run/'manifest.json'),sha256=files[path.name],checks=checks,
            weight_key=key,weight_cast='bfloat16_then_float64',layer=layer,kind=kind))
        print(f'SVD {name}: {heads} heads',flush=True)
    write_json(run/'svd/index.json',dict(manifest_sha256=sha256(run/'manifest.json'),files=files,
        validation_sha256={Path(name).with_suffix('.json').name:sha256((run/'svd'/name).with_suffix('.json')) for name in files},
        environment_sha256=sha256(run/'svd/environment.json'),thresholds=config['squared_energy_thresholds'],
        threshold_definition='squared_singular_values',created_utc=now()))
    write_json(run/'status.json',dict(state='svd_complete',completed_chunks=0))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    run=stage_entry(a.run_dir,'compute_svd')
    compute(run,a.device)


if __name__=='__main__':
    main()
