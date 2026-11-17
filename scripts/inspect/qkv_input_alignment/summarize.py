"""Verify alignment chunks and compute per-token statistics on CPU, without plotting."""
import argparse
from pathlib import Path
import numpy as np
from .common import matrix_specs, now, read_json, require, sha256, stage_entry, verify_chunk, verify_run, verify_svd, write_json, write_npz


def summarize(run):
    run=Path(run)
    manifest,meta=verify_run(run)
    verify_svd(run,manifest)
    mh,sh=sha256(run/'manifest.json'),sha256(run/'svd/index.json')
    collection=read_json(run/'collection.json')
    nseq,length=manifest['tokens_shape']
    require(collection['state']=='collection_complete' and collection['manifest_sha256']==mh and collection['svd_index_sha256']==sh,'Collection incomplete or mismatched')
    require(collection['chunks']==nseq and collection['tokens']==nseq*length,'Collection count mismatch')
    require(len(list((run/'tokens').glob('chunk_*.json')))==nseq,'Incomplete chunk set')
    thresholds=manifest['config']['squared_energy_thresholds']
    full=manifest['config'].get('measurement','sources')=='full_input'
    energy_kinds=manifest['config'].get('raw_projected_energy_kinds',[])
    specs=list(matrix_specs(meta, manifest['config']));banks={}
    for layer,kind,heads,hd,width in specs:
        with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as z:
            banks[layer,kind]={k:z[k].copy() for k in ('k','matrix_valid','captured_squared_fraction')}
    sums={};squares={};counts={};chunk_hashes={}
    scalar = manifest['config'].get('model_type') == 'gpt_base_scalar_res_qkv_no_embed_norm'
    references=None if scalar else read_json(run/'inputs/reference_chunks.json')
    for i in range(nseq):
        path=run/'tokens'/f'chunk_{i:05d}.npz'
        rec=verify_chunk(run,path,mh,sh)
        require(rec['sequence_id']==i and rec['tokens']==length and rec['tokens_sha256']==manifest['tokens_sha256'],'Chunk identity mismatch')
        require(rec['reference_chunk_sha256']==(None if scalar else references[i]['sha256']),'Chunk reference mismatch')
        checks=read_json(run/'validation'/path.with_suffix('.json').name)
        if scalar:
            require(all(checks.get(k) for k in ('independent_module_norms_match',
                    'unweighted_bank_identity_verified', 'scalar_receiver_order_verified',
                    'raw_embedding_identity_verified')), 'Scalar bank validation missing')
        require(checks.get('exact_uninstrumented_logits') and checks.get('reference_source_norms_match'), 'Replay checks missing')
        if manifest['config'].get('model_type') == 'gpt_base_no_embed_norm':
            require(checks.get('raw_embedding_identity_verified'), 'Raw embedding identity check missing')
        chunk_hashes[path.name]=rec['sha256']
        with np.load(path) as z:
            expected={'source_norm'}|{f'alignment_{kind}_{layer:02d}' for layer,kind,*_ in specs}
            expected|={f'raw_energy_{kind}_{layer:02d}' for layer,kind,*_ in specs if kind in energy_kinds}
            if full: expected.add('input_norm')
            require(set(z.files)==expected,'Unexpected chunk arrays')
            raw=z['source_norm']
            require(raw.shape==(length,2*meta['n_layer']+1) and np.isfinite(raw).all() and (raw>=0).all(),'Invalid source norms')
            if full:
                inputs=z['input_norm']
                require(inputs.shape==(length,meta['n_layer']) and np.isfinite(inputs).all() and (inputs>=0).all(),'Invalid full input norms')
                checks=read_json(run/'validation'/path.with_suffix('.json').name)
                require(checks.get('actual_projection_inputs_captured') and checks.get('qkv_inputs_exactly_equal') and checks.get('exact_uninstrumented_logits'), 'Full input checks missing')
            for layer,kind,heads,hd,width in specs:
                key=f'alignment_{kind}_{layer:02d}';values=z[key].astype(np.float64)
                bank=banks[layer,kind]
                require(values.shape==(length,heads,1 if full else 2*layer+1,len(thresholds)),'Wrong alignment dimensions')
                norms=inputs[:,layer:layer+1] if full else raw[:,:2*layer+1]
                valid=(norms[:,None,:,None]>0)&bank['matrix_valid'][None,:,None,None]&(bank['k'][None,:,None,:]>0)
                require(np.array_equal(np.isfinite(values),valid),'Invalid missing-value mask')
                require(np.isnan(values[~valid]).all(),'Undefined entries must be NaN')
                require(((values[valid]>=0)&(values[valid]<=1)).all(),'Invalid alignment range')
                if len(thresholds)>1:
                    delta=np.diff(values,axis=-1)
                    require((delta[np.isfinite(delta)]>=-manifest['config']['validation']['alignment_bound_tolerance']).all(),'Nonmonotonic threshold alignment')
                safe=np.where(valid,values,0.)
                sums[key]=sums.get(key,0)+safe.sum(0)
                squares[key]=squares.get(key,0)+np.square(safe).sum(0)
                counts[key]=counts.get(key,0)+valid.sum(0,dtype=np.int64)
                if kind in energy_kinds:
                    energy_key=f'raw_energy_{kind}_{layer:02d}'
                    energy=z[energy_key].astype(np.float64)
                    require(energy.shape==(length,heads,2*layer+1),'Wrong raw projected energy dimensions')
                    require(np.isfinite(energy).all() and (energy>=0).all(),'Invalid raw projected energy')
                    require((energy[np.broadcast_to(norms[:,None,:]==0,energy.shape)]==0).all(),'Zero source must project to zero')
                    sums[energy_key]=sums.get(energy_key,0)+energy.sum(0)
                    squares[energy_key]=squares.get(energy_key,0)+np.square(energy).sum(0)
                    counts[energy_key]=counts.get(energy_key,0)+np.full(energy.shape[1:],length,dtype=np.int64)
            if energy_kinds:
                checks=read_json(run/'validation'/path.with_suffix('.json').name)
                require(checks.get('raw_projected_energy_verified') and checks['svd_energy_max_relative_error']<=checks['svd_energy_relative_tolerance'], 'Energy identity check missing or failed')
    arrays={}
    for key,total in sums.items():
        count=counts[key]
        mean=np.divide(total,count,out=np.full_like(total,np.nan),where=count>0)
        second=np.divide(squares[key],count,out=np.full_like(total,np.nan),where=count>0)
        arrays[key+'_token_mean']=mean
        arrays[key+'_token_std']=np.sqrt(np.maximum(second-mean*mean,0))
        arrays[key+'_valid_count']=count
    path=run/'summaries/statistics.npz';write_npz(path,arrays)
    write_json(run/'summaries/schema.json',dict(source_index=([dict(index=0,source_family='full_projection_input')] if full else read_json(run/'inputs/source_index.json')),
        measurement='full_input' if full else 'sources',
        input_boundary='actual_linear_projection_pre_hook_after_receiving_RMSNorm' if full else 'separate_source_outputs',
        threshold_definition='cumulative_squared_singular_values_with_cutoff_ties',thresholds=thresholds,
        input_alignment='sum_top_k_squared_dot_products_divided_by_full_input_squared_norm',
        chunk_axes=['token','actual_head','available_source','threshold'],summary_axes=['actual_head','available_source','threshold'],
        availability='one_actual_input_per_receiving_layer' if full else 'source_index[:2*receiving_layer+1]',undefined='NaN; excluded from means/std; count retained',
        isotropic_reference='k / input_dimension',std_ddof=0,head_aggregation=False))
    if energy_kinds:
        schema=read_json(run/'summaries/schema.json')
        schema['raw_projected_energy']=dict(kinds=energy_kinds,definition='mean_over_tokens_of_squared_L2_of_W_head_times_raw_unweighted_source',
            receiver_normalization=False,routing_weighting=False,singular_directions='all',
            chunk_axes=['token','actual_head','available_source'],summary_axes=['actual_head','available_source'],zero_vectors='zero, included')
        write_json(run/'summaries/schema.json',schema)
    write_json(run/'validation/summary.json',dict(passed=True,tokens=nseq*length,chunks=nseq,
        hashes_verified=True,masks_verified=True,bounds_verified=True,head_aggregation=False))
    write_json(run/'results.json',dict(state='analysis_complete',tokens=nseq*length,
        manifest_sha256=mh,svd_index_sha256=sh,statistics_sha256=sha256(path),
        schema_sha256=sha256(run/'summaries/schema.json'),validation_sha256=sha256(run/'validation/summary.json'),
        chunk_sha256=chunk_hashes,visualization_status='not_rendered'))
    write_json(run/'status.json',dict(state='analysis_complete',completed_chunks=nseq,updated_utc=now()))
    print(f'Validated and summarized {nseq} sequences; no plots generated.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    a=p.parse_args();summarize(stage_entry(a.run_dir,'summarize'))


if __name__=='__main__': main()
