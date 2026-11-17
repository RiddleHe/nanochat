"""CPU tests with known geometry and a tiny synthetic model; no real checkpoint replay."""
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .common import matrix_specs, read_json, sha256, source_index, verify_svd, write_json, write_npz
from .compute_svd import compute, decompose
from .collect import alignment, capture_sources, measure
from .summarize import summarize
from .plot import plot
from .plot_raw_energy import plot as plot_raw_energy
from .prepare import prepare
from .reuse_svd import reuse, cutoffs


def config():
    return read_json(Path(__file__).with_name('config.json'))


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for kind in 'qkv': setattr(self,'c_'+kind,torch.nn.Linear(4,4,bias=False))
    def forward(self,x):
        return self.c_q(x)+self.c_k(x)+self.c_v(x),None


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.attn=Attention();self.mlp=torch.nn.Linear(4,4,bias=False)
    def forward(self,x):
        x=x+self.attn(x)[0]
        return x+self.mlp(x)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.transformer=torch.nn.Module()
        self.transformer.h=torch.nn.ModuleList([Block(),Block()])
        self.emb=torch.nn.Embedding(4,4)
        with torch.no_grad(): self.emb.weight[0].zero_()
    def forward(self,ids):
        x=self.emb(ids)
        for block in self.transformer.h: x=block(x)
        return x


class RoutedAttention(Attention):
    def forward(self,q,k,v):
        return self.c_q(q)+self.c_k(k)+self.c_v(v)


class TinyAttnRes(TinyModel):
    def __init__(self):
        super().__init__()
        for block in self.transformer.h: block.attn=RoutedAttention()
        self.attn_res_queries=torch.nn.Parameter(torch.randn(9,4))

    @staticmethod
    def _attn_res(queries,sources):
        values=torch.stack(sources)
        weights=torch.einsum('rd,nbtd->rnbt',queries,values).softmax(dim=1)
        return torch.einsum('rnbt,nbtd->rbtd',weights,values)

    def forward(self,ids):
        x=self.emb(ids)
        sources=[x/(x.square().mean(-1,keepdim=True)+1e-5).sqrt()]
        for l,block in enumerate(self.transformer.h):
            q,k,v=self._attn_res(self.attn_res_queries[4*l:4*l+3],sources)
            sources.append(block.attn(q,k,v))
            sources.append(block.mlp(self._attn_res(self.attn_res_queries[4*l+3:4*l+4],sources)[0]))
        self.oracle_sources=[s.detach().clone() for s in sources]
        return self._attn_res(self.attn_res_queries[-1:],sources)[0]


def fixture(root):
    torch.manual_seed(17)
    model=TinyModel().eval()
    # Exactly representable weights establish identical forward-cast and fixture matrices.
    with torch.no_grad():
        for p in model.parameters(): p.copy_(p.bfloat16().float())
    c=config();c['squared_energy_thresholds']=[.5,.9,1.0]
    run=root/'run';run.mkdir()
    for name in ('code','inputs','svd','tokens','summaries','validation','visualization','logs'):(run/name).mkdir()
    meta=dict(n_layer=2,n_head=2,n_kv_head=2,n_embd=4,sequence_len=4)
    metadata=dict(model_type='gpt_base',model_config=meta,step=1)
    write_json(run/'inputs/checkpoint_metadata.json',metadata)
    write_json(run/'inputs/source_index.json',source_index(2))
    tokens=np.array([[0,1,2,3],[3,2,1,0]],dtype=np.int64)
    np.save(run/'inputs/tokens.npy',tokens)
    cp=root/'model_000001.pt';torch.save(model.state_dict(),cp)
    refs=[]
    for i,seq in enumerate(tokens):
        with torch.inference_mode(): sources,_=capture_sources(model,torch.from_numpy(seq).unsqueeze(0))
        raw=torch.stack([s.norm(dim=-1) for s in sources],-1).numpy()
        path=root/f'reference_{i}.npz';write_npz(path,dict(source_norm=raw))
        refs.append(dict(sequence_id=i,path=str(path),sha256=sha256(path)))
    write_json(run/'inputs/reference_chunks.json',refs)
    m=dict(config=c,run_id='synthetic_test',tokens_shape=[2,4],tokens_sha256=sha256(run/'inputs/tokens.npy'),
        checkpoint_path=str(cp),checkpoint_sha256=sha256(cp),source_hashes={},
        input_hashes={str(p.relative_to(run)):sha256(p) for p in (run/'inputs').iterdir()})
    write_json(run/'manifest.json',m)
    return run,m,meta,model,tokens,refs


class AlignmentTests(unittest.TestCase):
    def test_no_embed_norm_selected_v_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            run,m,meta,model,tokens,refs=fixture(Path(temp))
            model.transformer.wte=model.emb
            m['config'].update(model_type='gpt_base_no_embed_norm', layers=[1], kinds=['V'], squared_energy_thresholds=[.5])
            metadata=read_json(run/'inputs/checkpoint_metadata.json')
            metadata['model_type']='gpt_base_no_embed_norm'
            write_json(run/'inputs/checkpoint_metadata.json',metadata)
            m['input_hashes']['inputs/checkpoint_metadata.json']=sha256(run/'inputs/checkpoint_metadata.json')
            write_json(run/'manifest.json',m)
            compute(run)
            self.assertEqual(set(verify_svd(run,m)['files']), {'V_01.npz'})
            with np.load(run/'svd/V_01.npz') as z: bank={k:z[k].copy() for k in z.files}
            ids=torch.from_numpy(tokens[0]).unsqueeze(0)
            with torch.inference_mode(): sources,logits=capture_sources(model,ids,model_type='gpt_base_no_embed_norm')
            torch.testing.assert_close(sources[0],model.emb(ids).reshape(-1,4),rtol=0,atol=0)
            with np.load(refs[0]['path']) as z: raw=z['source_norm']
            arrays,checks=measure(model,ids,{(1,'V'):bank},meta,m['config'],raw)
            self.assertTrue(checks['raw_embedding_identity_verified'])
            self.assertEqual(set(arrays), {'source_norm','alignment_V_01'})
            for h in range(2):
                for si in range(3):
                    x=sources[si].double().numpy();b=bank['vh'][h,:bank['k'][h,0]].astype('float64')
                    den=(x*x).sum(-1);num=((x@b.T)**2).sum(-1)
                    oracle=np.divide(num,den,out=np.full(4,np.nan),where=den>0)
                    np.testing.assert_allclose(arrays['alignment_V_01'][:,h,si,0],oracle,rtol=2e-6,atol=2e-7,equal_nan=True)
            with self.assertRaises(ValueError): list(matrix_specs(meta,dict(layers=[2],kinds=['V'])))
            with self.assertRaises(ValueError): list(matrix_specs(meta,dict(layers=[1],kinds=['invalid'])))

    def test_attn_res_source_adapter_and_alignment(self):
        torch.manual_seed(21)
        model=TinyAttnRes().eval();ids=torch.tensor([[0,1,2,3]])
        with torch.inference_mode():
            baseline=model(ids);expected=model.oracle_sources
            captured,logits=capture_sources(model,ids,model_type='gpt_base_attn_res_qkv')
        torch.testing.assert_close(baseline,logits,rtol=0,atol=0)
        self.assertNotIn('_attn_res',model.__dict__)
        for got,want in zip(captured,expected):
            torch.testing.assert_close(got,want.reshape(-1,4),rtol=0,atol=0)
        self.assertFalse(torch.equal(captured[0],model.emb(ids).reshape(-1,4)))
        meta=dict(n_layer=2,n_head=2,n_kv_head=2,n_embd=4)
        c=config();c.update(model_type='gpt_base_attn_res_qkv',squared_energy_thresholds=[.5],raw_projected_energy_kinds=['V'])
        banks={}
        for l,block in enumerate(model.transformer.h):
            for kind in 'QKV':
                weight=getattr(block.attn,'c_'+kind.lower()).weight.detach().bfloat16().float()
                parts=[decompose(weight[h*2:h*2+2],[.5])[0] for h in range(2)]
                banks[l,kind]={key:np.stack([p[key] for p in parts]) for key in parts[0]}
        raw=torch.stack([s.norm(dim=-1) for s in captured],-1).numpy()
        arrays,checks=measure(model,ids,banks,meta,c,raw)
        self.assertTrue(checks['unweighted_bank_identity_verified'])
        self.assertTrue(checks['raw_projected_energy_verified'])
        for l in range(2):
            for kind in 'QKV':
                bank=banks[l,kind]
                for h in range(2):
                    for si in range(2*l+1):
                        x=expected[si].reshape(-1,4).double().numpy()
                        b=bank['vh'][h,:bank['k'][h,0]].astype('float64')
                        den=(x*x).sum(-1);num=((x@b.T)**2).sum(-1)
                        oracle=np.divide(num,den,out=np.full(4,np.nan),where=den>0)
                        np.testing.assert_allclose(arrays[f'alignment_{kind}_{l:02d}'][:,h,si,0],oracle,rtol=2e-6,atol=2e-7,equal_nan=True)
                        if kind=='V':
                            w=model.transformer.h[l].attn.c_v.weight.detach().bfloat16().double().numpy()[2*h:2*h+2]
                            expected_energy=((x@w.T)**2).sum(-1)
                            np.testing.assert_allclose(arrays[f'raw_energy_V_{l:02d}'][:,h,si],expected_energy,rtol=2e-6,atol=2e-7)
        # Cleanup must also restore routing when an observed forward fails.
        with patch.object(model.transformer.h[0].mlp,'forward',side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError,'test failure'):
                capture_sources(model,ids,model_type='gpt_base_attn_res_qkv')
        self.assertNotIn('_attn_res',model.__dict__)
        self.assertFalse(model.transformer.h[0].attn._forward_hooks)

    def test_reuse_svd_new_thresholds_without_decomposition(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source,m,meta,model,tokens,refs=fixture(root)
            compute(source)
            run=root/'reused'
            shutil.copytree(source,run)
            shutil.rmtree(run/'svd');(run/'svd').mkdir()
            m=read_json(run/'manifest.json')
            m['config']['squared_energy_thresholds']=[.25,.5,.9]
            write_json(run/'inputs/svd_reference.json',dict(run=str(source),manifest_sha256=sha256(source/'manifest.json'),svd_index_sha256=sha256(source/'svd/index.json')))
            m['input_hashes']['inputs/svd_reference.json']=sha256(run/'inputs/svd_reference.json')
            write_json(run/'manifest.json',m)
            with patch('torch.linalg.svd',side_effect=AssertionError('SVD must not run')):
                reuse(run)
            for kind in 'QKV':
                with np.load(source/'svd'/f'{kind}_01.npz') as old, np.load(run/'svd'/f'{kind}_01.npz') as new:
                    for key in ('sigma','vh','rank','matrix_valid'):
                        self.assertEqual(new[key].tobytes(),old[key].tobytes())
                    np.testing.assert_array_equal(new['k'][:,1:],old['k'][:,:2])
                    x=torch.randn(3,4,4)
                    a=alignment(x,torch.from_numpy(old['vh']),torch.from_numpy(old['k']),torch.from_numpy(old['matrix_valid']))
                    b=alignment(x,torch.from_numpy(new['vh']),torch.from_numpy(new['k']),torch.from_numpy(new['matrix_valid']))
                    torch.testing.assert_close(a[...,:2],b[...,1:],rtol=0,atol=0)
            k,_=cutoffs(np.array([3.,1.,1.]),3,[.5,.9],1e-8)
            self.assertEqual(k.tolist(),[1,3])
            (source/'svd/V_01.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SVD changed'):
                reuse(run)

    def test_undivided_input_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            run,m,meta,model,tokens,refs=fixture(Path(temp))
            m['config']['measurement']='full_input'
            write_json(run/'manifest.json',m)
            compute(run)
            banks={}
            for layer in range(2):
                for kind in 'QKV':
                    with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as z:
                        banks[layer,kind]={k:z[k].copy() for k in z.files}
            mh,sh=sha256(run/'manifest.json'),sha256(run/'svd/index.json')
            chunks=[]
            for i,seq in enumerate(tokens):
                ids=torch.from_numpy(seq).unsqueeze(0)
                # Independent explicit forward: complete residual input, not source averages.
                with torch.inference_mode():
                    x=model.emb(ids);inputs=[]
                    for block in model.transformer.h:
                        inputs.append(x.reshape(-1,4).double().numpy().copy())
                        x=x+block.attn(x)[0];x=x+block.mlp(x)
                with np.load(refs[i]['path']) as z:raw=z['source_norm']
                arrays,checks=measure(model,ids,banks,meta,m['config'],raw)
                self.assertTrue(checks['actual_projection_inputs_captured'])
                for layer in range(2):
                    np.testing.assert_allclose(arrays['input_norm'][:,layer],np.linalg.norm(inputs[layer],axis=-1),rtol=1e-6)
                    for kind in 'QKV':
                        values=arrays[f'alignment_{kind}_{layer:02d}'];bank=banks[layer,kind]
                        self.assertEqual(values.shape,(4,2,1,3))
                        for h in range(2):
                            for ti,k in enumerate(bank['k'][h]):
                                x=inputs[layer];denom=(x*x).sum(-1)
                                coords=x@bank['vh'][h,:k].astype('float64').T
                                oracle=np.divide((coords*coords).sum(-1),denom,out=np.full(4,np.nan),where=denom>0)
                                np.testing.assert_allclose(values[:,h,0,ti],oracle,rtol=2e-6,atol=2e-7,equal_nan=True)
                chunks.append(arrays)
                path=run/'tokens'/f'chunk_{i:05d}.npz';write_npz(path,arrays)
                vp=run/'validation'/path.with_suffix('.json').name;write_json(vp,checks)
                write_json(path.with_suffix('.json'),dict(sequence_id=i,tokens=4,sha256=sha256(path),manifest_sha256=mh,svd_index_sha256=sh,
                    validation_sha256=sha256(vp),tokens_sha256=m['tokens_sha256'],reference_chunk_sha256=refs[i]['sha256']))
            write_json(run/'collection.json',dict(state='collection_complete',chunks=2,tokens=8,manifest_sha256=mh,svd_index_sha256=sh))
            summarize(run)
            self.assertEqual(read_json(run/'summaries/schema.json')['measurement'],'full_input')
            with np.load(run/'summaries/statistics.npz') as z:
                values=np.concatenate([c['alignment_Q_01'] for c in chunks]).astype('float64')
                np.testing.assert_allclose(z['alignment_Q_01_token_mean'],np.nanmean(values,axis=0))
                np.testing.assert_allclose(z['alignment_Q_01_token_std'],np.nanstd(values,axis=0),atol=1e-14)
                np.testing.assert_array_equal(z['alignment_Q_01_valid_count'],6)
            with self.assertRaisesRegex(ValueError,'per-source'):plot(run,Path(temp)/'bad.html',[1],[0])

    def test_prepare_and_frozen_cpu_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            parent,m,meta,_,tokens,refs=fixture(root)
            model_code=parent/'code/nanochat'
            model_code.mkdir();(model_code/'__init__.py').write_text('')
            tokenizer=root/'tokenizer.bin';tokenizer.write_bytes(b'synthetic')
            cache=root/'hub/models--varunneal--flash-attention-3';cache.mkdir(parents=True)
            (cache/'kernel').write_bytes(b'synthetic')
            write_json(parent/'inputs/provenance.json',dict(tokens_sha256=m['tokens_sha256'],tokenizer_sha256={str(tokenizer):sha256(tokenizer)}))
            m.update(config=dict(experiment='d24_qkv_separate_sources',num_sequences=2,sequence_length=4),
                metadata_sha256=sha256(parent/'inputs/checkpoint_metadata.json'),base_dir=str(root),
                source_hashes={'nanochat/__init__.py':sha256(model_code/'__init__.py')},
                kernel_cache_path=str(cache),kernel_cache_sha256={'kernel':sha256(cache/'kernel')})
            write_json(parent/'manifest.json',m)
            for i,ref in enumerate(refs):
                path=parent/'tokens'/f'chunk_{i:05d}.npz'
                path.write_bytes(Path(ref['path']).read_bytes())
                write_json(path.with_suffix('.json'),dict(sequence_id=i,tokens_sha256=m['tokens_sha256'],
                    manifest_sha256=sha256(parent/'manifest.json'),sha256=sha256(path)))
            write_npz(parent/'summaries/statistics.npz',{'dummy':np.array([1])})
            write_json(parent/'results.json',dict(state='analysis_complete',statistics_sha256=sha256(parent/'summaries/statistics.npz')))
            write_json(parent/'validation/summary.json',dict(passed=True))
            c=config();c['reference_run']=str(parent)
            actual_check_output=subprocess.check_output
            def commands(command,**kwargs):
                if command[0]=='nvidia-smi': return '0, GPU-SYNTHETIC, Synthetic device\n'
                return actual_check_output(command,**kwargs)
            with patch('subprocess.check_output',side_effect=commands):
                run=prepare(c,root/'prepared',0,'test')
            self.assertEqual(read_json(run/'status.json')['state'],'prepared')
            self.assertFalse((run/'svd/index.json').exists())
            proc=subprocess.run([sys.executable,'-m','scripts.inspect.qkv_input_alignment.compute_svd',
                '--run-dir',str(run)],capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            self.assertTrue((run/'svd/index.json').exists())
            # Tampered frozen source must be rejected on reuse.
            (run/'code/nanochat/__init__.py').write_text('# changed')
            proc=subprocess.run([sys.executable,'-m','scripts.inspect.qkv_input_alignment.compute_svd',
                '--run-dir',str(run)],capture_output=True,text=True)
            self.assertNotEqual(proc.returncode,0)
            self.assertIn('Frozen code changed',proc.stderr)

    def test_squared_selection_and_ties(self):
        c=config()
        a=torch.tensor([[3.,0.,0.,0.],[0.,1.,0.,0.]])
        b,_=decompose(a,[.9,1.],validation=c['validation'])
        self.assertEqual(b['k'].tolist(),[1,2])  # linear singular mass would need 2 at .9
        tie,_=decompose(torch.diag(torch.tensor([3.,1.,1.])),[.9])
        self.assertEqual(tie['k'].tolist(),[3])
        zero,_=decompose(torch.zeros(2,4),[.9])
        self.assertFalse(zero['matrix_valid']);self.assertEqual(zero['k'][0],0)

    def test_geometry_nullspace_scaling_and_zero(self):
        basis=torch.tensor([[[1.,0.,0.,0.],[0.,1.,0.,0.]],[[0.,1.,0.,0.],[0.,0.,1.,0.]]])
        x=torch.tensor([[[3.,4.,0.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.]]])
        ks=torch.tensor([[1,2],[1,2]])
        out=alignment(x,basis,ks,torch.tensor([True,True]))
        np.testing.assert_allclose(out[0,:,0].numpy(),[[9/25,1],[16/25,16/25]])
        np.testing.assert_array_equal(out[1,:,0].numpy(),np.zeros((2,2)))
        self.assertTrue(torch.isnan(out[2]).all())
        torch.testing.assert_close(out,alignment(-7*x,basis,ks,torch.tensor([True,True])),equal_nan=True)
        self.assertTrue(torch.isnan(alignment(x,basis,ks,torch.tensor([False,False]))).all())
        with self.assertRaises(ValueError): alignment(x,basis*10,ks,torch.tensor([True,True]))

    def test_degenerate_subspace_rotation(self):
        x=torch.tensor([[[2.,3.,4.]]])
        basis=torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
        rotation=torch.tensor([[.6,-.8],[.8,.6]])
        before=alignment(x,basis,torch.tensor([[2]]),torch.tensor([True]))
        after=alignment(x,rotation@basis,torch.tensor([[2]]),torch.tensor([True]))
        torch.testing.assert_close(before,after)

    def test_full_pipeline_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            run,m,meta,model,tokens,refs=fixture(Path(temp))
            m['config']['raw_projected_energy_kinds']=['V']
            write_json(run/'manifest.json',m)
            compute(run)
            before=sha256(run/'svd/index.json');compute(run)
            self.assertEqual(before,sha256(run/'svd/index.json'))
            banks={}
            for layer in range(2):
                for kind in 'QKV':
                    with np.load(run/'svd'/f'{kind}_{layer:02d}.npz') as z:banks[layer,kind]={k:z[k].copy() for k in z.files}
            mh,sh=sha256(run/'manifest.json'),sha256(run/'svd/index.json')
            expected=[]
            for i,seq in enumerate(tokens):
                with np.load(refs[i]['path']) as z:raw=z['source_norm']
                arrays,checks=measure(model,torch.from_numpy(seq).unsqueeze(0),banks,meta,m['config'],raw)
                expected.append(arrays)
                # Independent numpy projection for every available source/head/threshold.
                with torch.inference_mode():sources,_=capture_sources(model,torch.from_numpy(seq).unsqueeze(0))
                for layer in range(2):
                    for kind in 'QKV':
                        bank=banks[layer,kind]
                        for h in range(2):
                            for si in range(2*layer+1):
                                x=sources[si].double().numpy();denom=(x*x).sum(-1)
                                for ti,k in enumerate(bank['k'][h]):
                                    projected=x@bank['vh'][h,:k].astype('float64').T
                                    oracle=np.divide((projected*projected).sum(-1),denom,out=np.full(4,np.nan),where=denom>0)
                                    np.testing.assert_allclose(arrays[f'alignment_{kind}_{layer:02d}'][:,h,si,ti],oracle,rtol=2e-6,atol=2e-7,equal_nan=True)
                path=run/'tokens'/f'chunk_{i:05d}.npz';write_npz(path,arrays)
                vp=run/'validation'/path.with_suffix('.json').name;write_json(vp,checks)
                write_json(path.with_suffix('.json'),dict(sequence_id=i,tokens=4,sha256=sha256(path),manifest_sha256=mh,svd_index_sha256=sh,
                    validation_sha256=sha256(vp),tokens_sha256=m['tokens_sha256'],reference_chunk_sha256=refs[i]['sha256']))
            write_json(run/'collection.json',dict(state='collection_complete',chunks=2,tokens=8,manifest_sha256=mh,svd_index_sha256=sh))
            summarize(run)
            with np.load(run/'summaries/statistics.npz') as z:
                values=np.concatenate([e['alignment_Q_01'] for e in expected])
                np.testing.assert_allclose(z['alignment_Q_01_token_mean'],np.nanmean(values.astype('float64'),axis=0))
                np.testing.assert_allclose(z['alignment_Q_01_token_std'],np.nanstd(values.astype('float64'),axis=0),atol=1e-14)
                np.testing.assert_array_equal(z['alignment_Q_01_valid_count'],6)
                energies=np.concatenate([e['raw_energy_V_01'] for e in expected]).astype('float64')
                np.testing.assert_allclose(z['raw_energy_V_01_token_mean'],energies.mean(0))
                np.testing.assert_allclose(z['raw_energy_V_01_token_std'],energies.std(0),atol=1e-14)
                np.testing.assert_array_equal(z['raw_energy_V_01_valid_count'],8)
            energy_archive=plot_raw_energy(run,Path(temp)/'energy.html',[1])
            energy_data=read_json(energy_archive/'data.json')
            np.testing.assert_allclose(energy_data['panels'][0]['mean'],energies.mean(0)[0,[0,1]])
            self.assertTrue(energy_data['raw_sources'])
            output=Path(temp)/'plot.html';archive=plot(run,output,[1],[0,1],threshold=.9)
            d=read_json(archive/'data.json');self.assertEqual(len(d['panels']),6)
            self.assertEqual(d['panels'][0]['labels'],['Emb','A0'])
            self.assertNotIn('__ALIGNMENT_DATA__',output.read_text())
            # Preserve only an explicitly labeled synthetic preview for browser checking.
            preview=output.read_text().replace('GPTBase · input alignment','SYNTHETIC TEST · input alignment')
            Path('/tmp/qkv-alignment-synthetic.html').write_text(preview)
            plot(run,output,[1],[0,1],threshold=.9)  # identical regeneration
            empty=Path(temp)/'empty.html';plot(run,empty,[0],[0,1],family='mlp',threshold=.9)
            self.assertIn('"labels":[]',empty.read_text())
            path=run/'tokens/chunk_00000.npz'
            with path.open('ab') as f:f.write(b'corrupted')
            with self.assertRaisesRegex(ValueError,'checksum'):summarize(run)


if __name__=='__main__': unittest.main()
