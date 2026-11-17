"""Replay a checkpoint with observation hooks, then measure source alignment."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
import torch
from .common import matrix_specs, now, read_json, require, sha256, stage_entry, verify_chunk, verify_run, verify_svd, write_json, write_npz


def alignment(sources, basis, ks, matrix_valid, tolerance=3e-6):
    """sources [source,token,input], basis [head,direction,input], result [token,head,source,threshold]."""
    require(sources.ndim==3 and basis.ndim==3 and sources.shape[-1]==basis.shape[-1], 'Alignment shape mismatch')
    require(bool(torch.isfinite(sources).all()) and bool(torch.isfinite(basis).all()), 'Nonfinite inputs')
    require(ks.ndim==2 and ks.shape[0]==basis.shape[0] and bool(((ks>=0)&(ks<=basis.shape[1])).all()),'Invalid k')
    # The complete input norm includes energy in directions that the head cannot see.
    denominator=sources.square().sum(-1).T  # [token,source]
    maximum=int(ks.max())
    if maximum:
        coordinates=torch.einsum('std,hkd->tshk',sources,basis[:,:maximum])
        cumulative=coordinates.square().cumsum(-1)
        gather=(ks-1).clamp_min(0)[None,None].expand(cumulative.shape[0],cumulative.shape[1],-1,-1)
        numerator=cumulative.gather(-1,gather).permute(0,2,1,3)
    else:
        numerator=torch.zeros((sources.shape[1],basis.shape[0],sources.shape[0],ks.shape[1]),device=sources.device)
    valid=(denominator[:,None,:,None]>0)&matrix_valid[None,:,None,None]&(ks[None,:,None,:]>0)
    values=numerator/denominator[:,None,:,None].clamp_min(torch.finfo(sources.dtype).tiny)
    chosen=values[valid.expand_as(values)]
    require(bool(torch.isfinite(chosen).all()),'Nonfinite alignment')
    require(not chosen.numel() or (float(chosen.min())>=-tolerance and float(chosen.max())<=1+tolerance),'Alignment outside [0,1]')
    values=torch.where(valid,values.clamp(0,1),torch.full_like(values,float('nan')))
    return values


def capture_attn_res_sources(model, ids):
    """Observe the unweighted bank without altering routing or model arithmetic."""
    captured={};seen={};handles=[];receiving_layers=0
    original=model._attn_res
    had_override='_attn_res' in model.__dict__
    for layer,block in enumerate(model.transformer.h):
        for offset,module in ((1,block.attn),(2,block.mlp)):
            def capture(module,args,out,index=2*layer+offset):
                require(index not in captured, 'Source module called more than once')
                seen[index]=out
                captured[index]=out.detach().clone()
            handles.append(module.register_forward_hook(capture))
    def observe(queries,sources):
        nonlocal receiving_layers
        if queries.shape[0]==3:
            layer=receiving_layers
            require(len(sources)==2*layer+1, 'Unexpected AttnRes source bank')
            require(torch.equal(queries,model.attn_res_queries[4*layer:4*layer+3]), 'Unexpected routing query order')
            if layer==0:
                seen[0]=sources[0]
                captured[0]=sources[0].detach().clone()
            require(all(source is seen[i] for i,source in enumerate(sources)), 'Source-bank identity mismatch')
            receiving_layers+=1
        return original(queries,sources)
    model._attn_res=observe
    try:
        logits=model(ids)
    finally:
        if had_override: model._attn_res=original
        else: del model._attn_res
        for handle in handles: handle.remove()
    require(receiving_layers==len(model.transformer.h), 'Missing AttnRes receiving layers')
    require(len(captured)==2*len(model.transformer.h)+1, 'Missing AttnRes sources')
    return [captured[i].reshape(-1,captured[i].shape[-1]) for i in range(len(captured))],logits


def capture_sources(model, ids, full_inputs=None, model_type='gpt_base'):
    """Observe the same source boundaries as the existing separate-source collector."""
    if model_type == 'gpt_base_scalar_res_qkv_no_embed_norm':
        require(full_inputs is None, 'ScalarRes supports per-source alignment only')
        from .scalar_res import capture_sources as capture_scalar_sources
        return capture_scalar_sources(model, ids)
    if model_type=='gpt_base_attn_res_qkv':
        require(full_inputs is None, 'AttnRes full-input capture is not implemented')
        return capture_attn_res_sources(model,ids)
    require(model_type in ('gpt_base','gpt_base_no_embed_norm'), 'Unsupported source adapter')
    captured={};handles=[]
    def emb(module,args): captured[0]=args[0].detach().clone()
    handles.append(model.transformer.h[0].register_forward_pre_hook(emb))
    for layer,block in enumerate(model.transformer.h):
        if full_inputs is not None:
            for kind in 'QKV':
                def projection_input(module,args,layer=layer,kind=kind):
                    require((layer,kind) not in full_inputs, 'Projection called more than once')
                    full_inputs[layer,kind]=args[0].detach().clone()
                handles.append(getattr(block.attn,'c_'+kind.lower()).register_forward_pre_hook(projection_input))
        def attn(module,args,out,layer=layer):
            captured[1+2*layer]=(out[0] if isinstance(out,tuple) else out).detach().clone()
        def mlp(module,args,out,layer=layer): captured[2+2*layer]=out.detach().clone()
        handles.extend([block.attn.register_forward_hook(attn),block.mlp.register_forward_hook(mlp)])
    try:
        logits=model(ids)
    finally:
        for handle in handles: handle.remove()
    require(len(captured)==2*len(model.transformer.h)+1,'Missing source hooks')
    if full_inputs is not None:
        require(len(full_inputs)==3*len(model.transformer.h), 'Missing projection input hooks')
        for layer in range(len(model.transformer.h)):
            require(torch.equal(full_inputs[layer,'Q'],full_inputs[layer,'K']) and
                    torch.equal(full_inputs[layer,'Q'],full_inputs[layer,'V']), 'QKV full inputs differ')
    return [captured[i].reshape(-1,captured[i].shape[-1]) for i in range(len(captured))],logits


@torch.inference_mode()
def measure(model, ids, banks, meta, config, reference_raw):
    scalar = config.get('model_type') == 'gpt_base_scalar_res_qkv_no_embed_norm'
    if scalar:
        from .scalar_res import independent_norms
        baseline = model(ids)
        independent_logits, reference_raw = independent_norms(model, ids)
        require(torch.equal(baseline, independent_logits), 'Independent norm hooks changed logits')
        del independent_logits
    else:
        baseline=model(ids)
    full=config.get('measurement','sources')=='full_input'
    full_inputs={} if full else None
    model_type=config.get('model_type','gpt_base')
    sources,logits=capture_sources(model,ids,full_inputs,model_type)
    require(bool(torch.isfinite(logits).all()) and torch.equal(baseline,logits),'Instrumented logits differ')
    del baseline,logits
    if model_type == 'gpt_base_no_embed_norm' or scalar:
        expected = model.transformer.wte(ids).to(torch.bfloat16).reshape(-1, meta['n_embd'])
        require(torch.equal(sources[0], expected), 'Raw embedding boundary mismatch')
    raw=torch.stack([s.float().norm(dim=-1) for s in sources],-1).cpu().numpy()
    np.testing.assert_allclose(raw,reference_raw,rtol=config['validation']['source_norm_rtol'],atol=config['validation']['source_norm_atol'])
    arrays={'source_norm':raw}
    if full:
        arrays['input_norm']=torch.stack([full_inputs[l,'Q'].float().reshape(-1,meta['n_embd']).norm(dim=-1)
                                         for l in range(meta['n_layer'])],-1).cpu().numpy()
    energy_error=0.
    for layer,kind,heads,head_dim,width in matrix_specs(meta, config):
        bank=banks[layer,kind]
        basis=torch.from_numpy(bank['vh']).to(ids.device)
        ks=torch.from_numpy(bank['k']).to(ids.device)
        valid=torch.from_numpy(bank['matrix_valid']).to(ids.device)
        if full:
            x=full_inputs[layer,kind].float().reshape(1,-1,width)
            arrays[f'alignment_{kind}_{layer:02d}']=alignment(x,basis,ks,valid,config['validation']['alignment_bound_tolerance']).cpu().numpy()
            continue
        pieces=[];energy_pieces=[]
        energy=kind in config.get('raw_projected_energy_kinds',[])
        if energy:
            weight=getattr(model.transformer.h[layer].attn,'c_'+kind.lower()).weight.to(torch.bfloat16).float()
            sigma=torch.from_numpy(bank['sigma']).to(ids.device,dtype=torch.float32)
        for start in range(0,2*layer+1,config['source_batch_size']):
            group=torch.stack([s.float() for s in sources[start:min(start+config['source_batch_size'],2*layer+1)]])
            values=alignment(group,basis,ks,valid,config['validation']['alignment_bound_tolerance'])
            pieces.append(values.cpu().numpy())
            if energy:
                projected=(group@weight.T).reshape(group.shape[0],group.shape[1],heads,head_dim)
                squared=projected.square().sum(-1).permute(1,2,0)
                require(bool(torch.isfinite(squared).all()) and bool((squared>=0).all()), 'Invalid projected energy')
                # Independent SVD-coordinate identity on each source/head, first 8 tokens.
                coordinates=torch.einsum('std,hkd->tshk',group[:,:8],basis)
                spectral=(coordinates.square()*sigma[None,None].square()).sum(-1).permute(0,2,1)
                direct=squared[:8]
                error=float(((spectral-direct).abs()/direct.clamp_min(1e-12)).max())
                energy_error=max(energy_error,error)
                require(error<=2e-5, 'Projected norm and squared SVD sum disagree')
                energy_pieces.append(squared.cpu().numpy())
        arrays[f'alignment_{kind}_{layer:02d}']=np.concatenate(pieces,axis=2)
        if energy: arrays[f'raw_energy_{kind}_{layer:02d}']=np.concatenate(energy_pieces,axis=2)
    checks=dict(passed=True,exact_uninstrumented_logits=True,reference_source_norms_match=True,
        zero_source_vectors=int((raw==0).sum()),all_alignment_values_bounded=True)
    if scalar:
        checks.update(independent_module_norms_match=True, unweighted_bank_identity_verified=True,
                      scalar_receiver_order_verified=True, raw_embedding_identity_verified=True,
                      source_capture_adapter=model_type)
    if model_type == 'gpt_base_no_embed_norm':
        checks.update(raw_embedding_identity_verified=True, source_capture_adapter=model_type)
    if model_type=='gpt_base_attn_res_qkv':
        checks.update(unweighted_bank_identity_verified=True,source_capture_adapter=model_type)
    if config.get('raw_projected_energy_kinds'):
        checks.update(raw_projected_energy_verified=True,svd_energy_max_relative_error=energy_error,
                      svd_energy_validation_tokens_per_source_batch=8,svd_energy_relative_tolerance=2e-5)
    if full:
        checks.update(actual_projection_inputs_captured=True,qkv_inputs_exactly_equal=True,
                      zero_full_input_vectors=int((arrays['input_norm']==0).sum()))
    return arrays,checks


def collect(run):
    run=Path(run)
    manifest,meta=verify_run(run,checkpoint=True)
    config=manifest['config']
    svd=verify_svd(run,manifest)
    require(torch.cuda.device_count()==1,'Exactly one selected GPU is required for collection')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==manifest['gpu_uuid'],'Wrong selected GPU')
    for path,digest in manifest['tokenizer_sha256'].items(): require(sha256(path)==digest,'Tokenizer changed')
    for path,digest in manifest['kernel_cache_sha256'].items(): require(sha256(Path(manifest['kernel_cache_path'])/path)==digest,'Kernel cache changed')
    torch.manual_seed(config['seed']);torch.set_num_threads(config['cpu_threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Imports occur only after CLI routing into frozen code and environment.
    from nanochat.checkpoint_manager import build_model
    import nanochat
    require(Path(nanochat.__file__).resolve().parent==run/'code/nanochat','Model import escaped frozen code')
    environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
        platform=platform.platform(),gpu=torch.cuda.get_device_name(0),gpu_uuid=manifest['gpu_uuid'],tf32=False)
    if (run/'environment.json').exists(): require(read_json(run/'environment.json')==environment,'Replay environment changed')
    else: write_json(run/'environment.json',environment)
    packages='\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n'
    package_path=run/'environment-packages.txt'
    if package_path.exists(): require(package_path.read_text()==packages,'Installed packages changed')
    else: package_path.write_text(packages)
    checkpoint_dir=Path(manifest['checkpoint_path']).parent
    live_metadata=checkpoint_dir/f'meta_{manifest["checkpoint_step"]:06d}.json'
    require(sha256(live_metadata)==manifest['input_hashes']['inputs/checkpoint_metadata.json'],'Checkpoint metadata changed')
    model,_,_=build_model(str(checkpoint_dir),manifest['checkpoint_step'],torch.device('cuda:0'),'eval')
    model.requires_grad_(False)
    scalar = config.get('model_type') == 'gpt_base_scalar_res_qkv_no_embed_norm'
    if scalar:
        from .scalar_res import extract_coefficients
        extract_coefficients(model, run)
    banks={}
    for layer,kind,heads,hd,width in matrix_specs(meta, config):
        with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as z:
            bank={k:z[k].copy() for k in z.files}
        require(bank['vh'].shape==(heads,hd,width) and bank['k'].shape==(heads,len(config['squared_energy_thresholds'])),'SVD bank shape mismatch')
        banks[layer,kind]=bank
    tokens=np.load(run/'inputs/tokens.npy')
    require(list(tokens.shape)==manifest['tokens_shape'],'Wrong tokens shape')
    references=None if scalar else read_json(run/'inputs/reference_chunks.json')
    mh,sh=sha256(run/'manifest.json'),sha256(run/'svd/index.json')
    start_time=time.monotonic()
    for i,sequence in enumerate(tokens):
        path=run/'tokens'/f'chunk_{i:05d}.npz'
        if path.with_suffix('.json').exists():
            rec=verify_chunk(run,path,mh,sh)
            require(rec['sequence_id']==i and rec['tokens']==len(sequence),'Resume sequence mismatch')
            continue
        ref, raw = {'sha256': None}, None
        if references is not None:
            ref=references[i]
            require(ref['sequence_id']==i and sha256(ref['path'])==ref['sha256'],'Reference source chunk changed')
            with np.load(ref['path']) as z: raw=z['source_norm']
        ids=torch.from_numpy(sequence.copy()).unsqueeze(0).to('cuda:0')
        arrays,checks=measure(model,ids,banks,meta,config,raw)
        write_npz(path,arrays)
        validation=run/'validation'/path.with_suffix('.json').name
        write_json(validation,checks)
        write_json(path.with_suffix('.json'),dict(sequence_id=i,tokens=len(sequence),sha256=sha256(path),
            manifest_sha256=mh,svd_index_sha256=sh,tokens_sha256=manifest['tokens_sha256'],
            reference_chunk_sha256=ref['sha256'],validation_sha256=sha256(validation)))
        write_json(run/'status.json',dict(state='collecting',completed_chunks=i+1,updated_utc=now()))
        print(f'Alignment {i+1}/{len(tokens)} sequences; {time.monotonic()-start_time:.1f}s',flush=True)
    write_json(run/'collection.json',dict(state='collection_complete',chunks=len(tokens),tokens=int(tokens.size),
        manifest_sha256=mh,svd_index_sha256=sh,environment_sha256=sha256(run/'environment.json'),
        duration_seconds=time.monotonic()-start_time,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30))
    write_json(run/'status.json',dict(state='collection_complete',completed_chunks=len(tokens),updated_utc=now()))
    print('Collection complete. Run summarize separately; no plots generated.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();run=stage_entry(args.run_dir,'collect')
    try: collect(run)
    except Exception as error:
        write_json(run/'status.json',dict(state='failed',stage='collect',error=repr(error),traceback=traceback.format_exc(),updated_utc=now()))
        raise


if __name__=='__main__': main()
