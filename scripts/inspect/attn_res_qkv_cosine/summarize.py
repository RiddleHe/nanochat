"""Validate chunks and aggregate per-token means, population stds and counts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .prepare import sha256, write_json


def summarize(run):
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    config = manifest['config']
    assert sha256(run/'inputs/tokens.npy') == manifest['tokens_sha256']
    manifest_hash = sha256(run/'manifest.json')
    source_index = json.loads((run/'inputs/source_index.json').read_text())
    assert source_index == manifest['source_index']
    depth = len(source_index)//2
    sums, squares, counts, query_norms, maxima = {}, {}, {}, {}, {}
    assert len(list((run/'tokens').glob('chunk_*.json'))) == config['num_sequences']
    for index in range(config['num_sequences']):
        chunk = run/'tokens'/f'chunk_{index:05d}.npz'
        record = json.loads(chunk.with_suffix('.json').read_text())
        vp = run/'validation'/chunk.with_suffix('.json').name
        assert record['sha256'] == sha256(chunk)
        assert record['manifest_sha256'] == manifest_hash
        assert record['tokens_sha256'] == manifest['tokens_sha256']
        assert record['validation_sha256'] == sha256(vp)
        validation = json.loads(vp.read_text())
        assert validation['passed'] and validation['exact_forward_replay']
        assert len(validation['checks']) == depth
        for check in validation['checks']:
            assert check['exact_reference_weights'] and check['exact_bf16_mixture_replay'] and check['source_identity_passed']
            for key in ('score_identity_normalized_error','bf16_score_normalized_error','key_source_cosine_difference','rms_key_norm_relative_error'):
                maxima[key] = max(maxima.get(key,0),check[key])
        with np.load(chunk) as data:
            assert len(data.files) == depth*6
            for layer in range(depth):
                suffix=f'{layer:02d}'
                q = data['query_norm_'+suffix]
                assert q.shape == (3,) and np.isfinite(q).all() and (q>=0).all()
                if layer in query_norms:
                    np.testing.assert_array_equal(q,query_norms[layer])
                query_norms[layer]=q.copy()
                sn = data['source_norm_'+suffix]
                kn = data['key_norm_'+suffix]
                assert sn.shape == kn.shape == (config['sequence_length'],2*layer+1)
                valid=(q[None,:,None] > 0) & (sn[:,None,:] > 0)
                for name in ('cosine','key_cosine','scores','source_norm','key_norm'):
                    key=name+'_'+suffix
                    values=data[key]
                    assert values.dtype == np.float32
                    assert values.shape == ((config['sequence_length'],3,2*layer+1) if name in ('cosine','key_cosine','scores') else sn.shape)
                    if name in ('cosine','key_cosine'):
                        np.testing.assert_array_equal(np.isfinite(values),valid)
                        assert np.all(np.abs(values[valid])<=1+config['validation']['cosine_bound_tolerance'])
                    else:
                        assert np.isfinite(values).all()
                    finite=np.isfinite(values)
                    v=np.where(finite,values,0).astype(np.float64)
                    sums[key]=sums.get(key,0)+v.sum(0)
                    squares[key]=squares.get(key,0)+np.square(v).sum(0)
                    counts[key]=counts.get(key,0)+finite.sum(0)
    stats={}
    for key in sums:
        count=counts[key]
        mean=np.divide(sums[key],count,out=np.full_like(sums[key],np.nan),where=count>0)
        second=np.divide(squares[key],count,out=np.full_like(sums[key],np.nan),where=count>0)
        stats[key+'_token_mean']=mean
        stats[key+'_token_std']=np.sqrt(np.maximum(second-mean*mean,0))
        stats[key+'_valid_count']=count
    for layer,q in query_norms.items():
        stats[f'query_norm_{layer:02d}']=q
    np.savez(run/'summaries/statistics.npz',**stats)
    write_json(run/'summaries/schema.json',dict(query_types=['Q','K','V'],source_index=source_index,
        cosine_shape='[3,2*receiving_layer+1]',norm_shape='[2*receiving_layer+1]',std_ddof=0,
        measurement=config['measurement'],undefined='NaN when query or source L2 norm is zero',
        score_measurement='actual BF16 pre-softmax dot product, losslessly stored as FP32',
        query='forward-cast depth-routing query, not token Q projection',
        aggregation='per-token measurement then FP64 aggregation; token occurrences include BOS',
        key_cosine='cosine with actual BF16 RMS-normalized key; separate from raw-source cosine'))
    result=dict(passed=True,chunks_verified=config['num_sequences'],tokens=config['num_sequences']*config['sequence_length'],
        all_sequences_exact_forward_replay=True,all_layers_exact_reference_weights=True,
        all_layers_exact_bf16_mixture_replay=True,maxima=maxima,
        zero_query_receivers=[dict(layer=l,query=k) for l,q in query_norms.items() for j,k in enumerate(('Q','K','V')) if q[j]==0])
    write_json(run/'validation/summary.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    summarize(parser.parse_args().run_dir)
