"""Derive squared V-output norms relative to each token/head's average source.

Reuse validated per-token energy-share moments: q = (2L+1) * share.
L0 is verified from matching saved input alignments before adding its unit ratio.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def derive(run, alignment, output):
    m, result = read(run/'manifest.json'), read(run/'results.json')
    am, ar = read(alignment/'manifest.json'), read(alignment/'results.json')
    assert result['state'] == 'complete' and ar['state'] == 'analysis_complete'
    assert result['manifest_sha256'] == sha(run/'manifest.json')
    assert result['statistics_sha256'] == sha(run/'derived/statistics.npz')
    assert result['validation_sha256'] == sha(run/'validation/summary.json')
    assert read(run/'validation/summary.json')['passed']
    assert m['config']['model_type'] == am['config']['model_type'] == 'gpt_base_no_embed_norm'
    assert m['config']['layers'] == list(range(1,24))
    assert m['inputs_hashes']['checkpoint/model_007809.pt'] == am['checkpoint_sha256']
    assert m['tokens_sha256'] == am['tokens_sha256'] == result['tokens_sha256']
    assert ar['manifest_sha256'] == sha(alignment/'manifest.json')
    assert ar['validation_sha256'] == sha(alignment/'validation/summary.json')
    assert read(alignment/'validation/summary.json')['passed']
    assert ar['svd_index_sha256'] == sha(alignment/'svd/index.json')
    svd_index = read(alignment/'svd/index.json')
    # The SVD file hash is recorded by the frozen SVD stage.
    svd_hash = sha(alignment/'svd/V_00.npz')
    assert svd_hash in json.dumps(svd_index)
    with np.load(alignment/'svd/V_00.npz') as svd:
        assert svd['matrix_valid'].all()
        assert all(np.all(svd['sigma'][h, :int(svd['k'][h,0])] > 0) for h in range(12))
    # Positive top-subspace energy proves a nonzero full V projection. L0 has
    # only Emb, so the resulting source-average ratio is exactly one.
    minimum = np.full(12, np.inf)
    token_count = 0
    alignment_chunks = {}
    for i in range(am['tokens_shape'][0]):
        p = alignment/'tokens'/f'chunk_{i:05d}.npz'
        digest = sha(p)
        assert digest == ar['chunk_sha256'][p.name]
        alignment_chunks[p.name] = digest
        with np.load(p) as z:
            a = z['alignment_V_00'][:,:,0,0]
            assert a.shape == (2048,12) and np.isfinite(a).all() and (a > 1e-8).all()
            minimum = np.minimum(minimum, a.min(0))
            token_count += a.shape[0]
        if (i+1)%32 == 0:
            print(f'Checked L0 nonzero V projection: {i+1}/128 sequences', flush=True)
    assert token_count == result['tokens'] == 262144
    arrays = dict(relative_energy_V_00_mean=np.ones((12,1)),
                  relative_energy_V_00_std=np.zeros((12,1)),
                  relative_energy_V_00_variance=np.zeros((12,1)),
                  relative_energy_V_00_valid_count=np.full((12,1), token_count, dtype=np.int64))
    identity_error = 0.
    with np.load(run/'derived/statistics.npz') as z:
        for layer in range(1,24):
            n = 2*layer+1
            key = f'energy_share_V_{layer:02d}'
            mean, std, var = z[key+'_mean'], z[key+'_std'], z[key+'_variance']
            assert mean.shape == (12,n)
            assert np.isfinite(mean).all() and np.isfinite(std).all()
            assert ((mean>=0)&(mean<=1)).all() and (std>=0).all()
            np.testing.assert_allclose(std**2,var,rtol=1e-12,atol=1e-15)
            identity_error=max(identity_error,float(np.max(np.abs(mean.sum(-1)-1))))
            assert identity_error < 1e-12
            prefix=f'relative_energy_V_{layer:02d}'
            arrays.update({prefix+'_mean':mean*n,prefix+'_std':std*n,
                           prefix+'_variance':var*n*n,
                           prefix+'_valid_count':np.full(mean.shape,token_count,dtype=np.int64)})
    # Independently verify the conversion against direct per-token division on
    # one actual sequence, all 23 layers, all heads, and every available source.
    chunk=run/'tokens/chunk_00000.npz'
    assert sha(chunk)==result['chunk_hashes'][chunk.name]
    direct_error=0.
    with np.load(chunk) as z:
        for layer in range(1,24):
            e=z[f'energy_V_{layer:02d}'].astype(np.float64)
            assert (e>=0).all() and np.isfinite(e).all() and (e.sum(-1)>0).all()
            q=e/e.mean(-1,keepdims=True)
            scaled=e/e.sum(-1,keepdims=True)*(2*layer+1)
            direct_error=max(direct_error,float(np.max(np.abs(q-scaled))))
            np.testing.assert_allclose(q,scaled,rtol=5e-15,atol=1e-14)
    np.savez_compressed(output/'statistics.npz',**arrays)
    write(output/'schema.json',dict(formula='q_s = squared_output_norm_s / mean_all_available_sources(squared_output_norm)',
        denominator_sources='Emb plus all prior attention and MLP outputs, including zeros',
        normalization_order='divide per token/head, then average across tokens; curves equally average heads',
        square_root=False,top_k=False,source_axis='Emb,A0,M0,A1,M1,...',
        directions='all 128 V-head output coordinates',layers=list(range(24)),tokens=token_count,
        receiver_normalization='common receiving RMSNorm factor cancels in source ratios',
        zero_source='zero energy included in the average',ddof=0,reference=1,
        layer0='unit ratio derived after verifying positive projection for every sampled token/head'))
    write(output/'validation.json',dict(passed=True,source_statistics_hash_verified=True,
        source_validation_passed=True,matched_checkpoint_and_tokens=True,
        layer0_tokens=token_count,layer0_min_squared_alignment_by_head=minimum.tolist(),
        layer0_svd_sha256=svd_hash,layer0_chunk_sha256=alignment_chunks,
        per_head_mean_source_identity_error=identity_error,
        direct_per_token_check_sequences=1,direct_per_token_check_layers=23,
        direct_vs_scaled_share_max_abs_error=direct_error))
    write(output/'results.json',dict(state='complete',run=str(run),alignment_run=str(alignment),
        manifest_sha256=sha(output/'manifest.json'),
        output_sha256={n:sha(output/n) for n in ['statistics.npz','schema.json','validation.json']}))
    print(f'Validated L0–L23; direct ratio check error {direct_error:.3g}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--alignment-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--frozen',action='store_true')
    a=p.parse_args();run=a.run_dir.resolve();alignment=a.alignment_run.resolve()
    output=(a.output_dir or run/'derived/source_mean_relative_energy').resolve()
    if not a.frozen:
        output.mkdir(parents=True,exist_ok=False)
        (output/'code').mkdir();script=output/'code/relative_energy.py';shutil.copy2(__file__,script)
        files=[run/'manifest.json',run/'results.json',run/'derived/statistics.npz',run/'validation/summary.json',alignment/'manifest.json',alignment/'results.json']
        write(output/'manifest.json',dict(run=str(run),alignment_run=str(alignment),python=sys.version,numpy=np.__version__,
              code_sha256=sha(script),inputs={str(f):sha(f) for f in files}))
        command=[sys.executable,str(script),'--run-dir',str(run),'--alignment-run',str(alignment),'--output-dir',str(output),'--frozen']
        with (output/'analysis.log').open('w') as log:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            for line in child.stdout:print(line,end='',flush=True);log.write(line)
            assert child.wait()==0,'Frozen derivation failed; see analysis.log'
        return
    m=read(output/'manifest.json')
    assert Path(__file__).resolve()==output/'code/relative_energy.py'
    assert sha(__file__)==m['code_sha256'] and m['python']==sys.version and m['numpy']==np.__version__
    assert m['run']==str(run) and m['alignment_run']==str(alignment)
    assert all(sha(Path(path))==digest for path,digest in m['inputs'].items())
    derive(run,alignment,output)

if __name__=='__main__':main()
