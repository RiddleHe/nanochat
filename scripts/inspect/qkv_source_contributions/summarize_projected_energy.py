"""Square saved per-token per-head projection norms, then average tokens on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .prepare import sha256, write_json


def derive(run, layers, kinds='QKV'):
    run=Path(run).resolve()
    manifest=json.loads((run/'manifest.json').read_text());results=json.loads((run/'results.json').read_text())
    assert manifest['config']['experiment']=='d24_qkv_separate_sources'
    assert results['state']=='analysis_complete' and json.loads((run/'validation/summary.json').read_text())['passed']
    assert sha256(run/'summaries/statistics.npz')==results['statistics_sha256']
    assert sha256(run/'inputs/tokens.npy')==manifest['tokens_sha256']
    meta=json.loads((run/'inputs/checkpoint_metadata.json').read_text())['model_config']
    assert sha256(run/'inputs/checkpoint_metadata.json')==manifest['metadata_sha256']
    assert layers and len(set(layers))==len(layers) and all(0<=l<meta['n_layer'] for l in layers)
    sources=json.loads((run/'inputs/source_index.json').read_text());assert sources==manifest['source_index']
    mh=sha256(run/'manifest.json');nseq=manifest['config']['num_sequences'];length=manifest['config']['sequence_length']
    sums={};squares={};hashes={};count=0
    for i in range(nseq):
        path=run/'tokens'/f'chunk_{i:05d}.npz';rec=json.loads(path.with_suffix('.json').read_text())
        assert rec['sequence_id']==i and rec['tokens']==length and rec['manifest_sha256']==mh and rec['tokens_sha256']==manifest['tokens_sha256']
        assert sha256(path)==rec['sha256'];hashes[path.name]=rec['sha256']
        assert json.loads((run/'validation'/path.with_suffix('.json').name).read_text())['passed']
        with np.load(path) as z:
            for layer in layers:
                for kind in kinds:
                    key=f'abs_{kind}_{layer:02d}';a=z[key].astype(np.float64)
                    heads=meta['n_head'] if kind=='Q' else meta['n_kv_head']
                    assert a.shape==(length,heads,2*layer+1) and np.isfinite(a).all() and (a>=0).all()
                    energy=np.square(a)  # Square each token/head/source independently BEFORE averaging.
                    sums[key]=sums.get(key,0)+energy.sum(axis=0)
                    squares[key]=squares.get(key,0)+np.square(energy).sum(axis=0)
        count+=length
        if (i+1)%16==0:print(f'Squared and accumulated {i+1}/{nseq} sequences',flush=True)
    assert count==results['tokens']==nseq*length
    arrays={};maximum_check_error=0.
    with np.load(run/'summaries/statistics.npz') as previous:
        for key,total in sums.items():
            mean=total/count
            std=np.sqrt(np.maximum(squares[key]/count-mean*mean,0))
            assert np.isfinite(mean).all() and np.isfinite(std).all()
            # Independent identity using the previous UNSQUARED norm statistics.
            oracle=previous[key+'_token_mean']**2+previous[key+'_token_std']**2
            np.testing.assert_allclose(mean,oracle,rtol=1e-10,atol=1e-10)
            maximum_check_error=max(maximum_check_error,float(np.max(np.abs(mean-oracle))))
            arrays[key+'_squared_token_mean']=mean;arrays[key+'_squared_token_std']=std
    inputs=dict(run_dir=str(run),manifest_sha256=mh,statistics_sha256=results['statistics_sha256'],tokens_sha256=manifest['tokens_sha256'],
                chunk_sha256=hashes,layers=layers,kinds=kinds,code_sha256=sha256(Path(__file__)),prepare_sha256=sha256(Path(__file__).with_name('prepare.py')))
    fp=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:12]
    output=run/'derived'/f'projected_energy_{"_".join(map(str,layers))}_{kinds}_{fp}'
    if output.exists():
        old=json.loads((output/'manifest.json').read_text());assert old['inputs']==inputs and sha256(output/'statistics.npz')==old['statistics_sha256']
        return output
    output.mkdir(parents=True)
    np.savez(output/'statistics.npz',**arrays)
    for p in [Path(__file__),Path(__file__).with_name('prepare.py')]:shutil.copy2(p,output/p.name)
    write_json(output/'manifest.json',dict(inputs=inputs,tokens=count,source_index=sources,model_config=meta,
        statistics_sha256=sha256(output/'statistics.npz'),quantity='mean_token(norm(W_head * actual_source_component)**2)',
        input='source_output / actual_complete_residual_RMSNorm_denominator',projection='FP32 with BF16-forward-cast weights; before RoPE and QK normalization',
        axes=['head','available_source'],aggregation='square per token then mean; no head averaging or concatenation',
        singular_directions='all directions; no top-k truncation',std_ddof=0))
    write_json(output/'validation.json',dict(passed=True,chunks=nseq,tokens=count,chunk_hashes_verified=True,
        second_moment_identity_verified=True,max_absolute_identity_error=maximum_check_error))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--layers',type=int,nargs='+',required=True)
    p.add_argument('--kinds',default='QKV',choices=['Q','K','V','QKV'])
    a=p.parse_args();print(derive(a.run_dir,a.layers,a.kinds))


if __name__=='__main__':main()
