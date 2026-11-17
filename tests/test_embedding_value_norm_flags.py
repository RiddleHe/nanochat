"""Value-input normalization, pooling, legacy loading, and frozen recipe flags."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import subprocess

import pytest
import torch

import nanochat.flash_attention as flash
import nanochat.model.gpt_base as base
import nanochat.model.gpt_base_no_embed_norm as no_norm
from nanochat.engine import KVCache
from nanochat.model_registry import get_model

KEYS = [
    'gpt_base_pooled_qk_embed_no_embed_norm',
    'gpt_base_single_qk_embed_no_embed_norm',
    'gpt_base_split_embed_head_out_no_embed_norm',
    'gpt_base_single_qk_value_proj_no_embed_norm',
]


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(flash, 'USE_FA3', False)
    monkeypatch.setattr(base, 'COMPUTE_DTYPE', torch.float32)
    monkeypatch.setattr(no_norm, 'COMPUTE_DTYPE', torch.float32)
    torch.manual_seed(42)
    torch.set_num_threads(1)


def build(key, **flags):
    Config, Model = get_model(key)
    cfg = Config(sequence_len=16, vocab_size=31, n_layer=3, n_head=2,
                 n_kv_head=2, n_embd=16, window_pattern='L', attention_chunk_size=2, **flags)
    model = Model(cfg)
    model.init_weights()
    with torch.no_grad():
        for b in model.transformer.h:
            b.attn.c_proj.weight.normal_(std=.1)
            b.mlp.c_proj.weight.normal_(std=.1)
    return model


@pytest.mark.parametrize('key', KEYS)
@pytest.mark.parametrize('window', [(-1, 0), (2, 0)])
def test_normalizes_only_value_branch_once_with_gradients_and_cache(key, window):
    model = build(key, norm_value_input=True, sum_head_attention=key==KEYS[0])
    reference = copy.deepcopy(model)
    reference.config.norm_value_input = False
    for m in (model, reference):
        for b in m.transformer.h:
            b.attn.window_size = window
    # Independent reference: normalize at each target attention boundary.
    for i in reference.config.pooled_qk_target_layers:
        def normalized_value(module, args):
            x, emb, ve, cs, cache = args
            return x, torch.nn.functional.rms_norm(emb, (emb.shape[-1],)), ve, cs, cache
        reference.transformer.h[i].attn.register_forward_pre_hook(normalized_value)
    tokens = torch.randint(31, (2, 9))
    observed = []
    handles = [model.transformer.h[0].register_forward_pre_hook(lambda m,a: observed.append(a[0].detach().clone()))]
    value_ids = []
    for i in model.config.pooled_qk_target_layers:
        handles.append(model.transformer.h[i].attn.register_forward_pre_hook(lambda m,a: value_ids.append(id(a[1]))))
    actual = model(tokens, tokens.roll(-1, 1))
    expected = reference(tokens, tokens.roll(-1, 1))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(observed[0], model.transformer.wte(tokens))
    assert len(set(value_ids)) == 1  # Same normalized tensor reused across targets.
    for h in handles: h.remove()
    actual.backward(); expected.backward()
    for (n,p),(_,r) in zip(model.named_parameters(),reference.named_parameters()):
        torch.testing.assert_close(p.grad,r.grad,atol=2e-6,rtol=2e-4,msg=n)
    assert model.transformer.wte.weight.grad.abs().sum()>0
    model.eval()
    with torch.no_grad():
        full = model(tokens)
        cache = KVCache(2,2,16,8,3,'cpu',torch.float32)
        chunks = [model(tokens[:,:3],kv_cache=cache),model(tokens[:,3:7],kv_cache=cache),model(tokens[:,7:],kv_cache=cache)]
        torch.testing.assert_close(torch.cat(chunks,1),full,atol=2e-6,rtol=2e-4)


@pytest.mark.parametrize('key', KEYS)
def test_old_config_loading_defaults_and_new_roundtrip(key, tmp_path):
    Config, Model = get_model(key)
    old = build(key)
    legacy = asdict(old.config)
    legacy.pop('norm_value_input'); legacy.pop('sum_head_attention')
    restored = Model(Config(**legacy));restored.load_state_dict(old.state_dict(),strict=True)
    tokens = torch.randint(31,(1,7))
    torch.testing.assert_close(old(tokens),restored(tokens),atol=0,rtol=0)
    new = build(key,norm_value_input=True,sum_head_attention=key==KEYS[0])
    path=tmp_path/'checkpoint.pt'
    torch.save({'config':asdict(new.config),'model':new.state_dict()},path)
    saved=torch.load(path,weights_only=True)
    loaded=Model(Config(**saved['config']));loaded.load_state_dict(saved['model'],strict=True)
    torch.testing.assert_close(new(tokens),loaded(tokens),atol=0,rtol=0)
    assert new.state_dict().keys()==old.state_dict().keys()


def test_sum_probabilities_and_gradients():
    mean = build(KEYS[0]).transformer.h[-1].attn
    summed = copy.deepcopy(mean);summed.sum_head_attention=True
    q,k,emb = [torch.randn(*shape,requires_grad=True) for shape in [(2,2,4,8),(2,2,4,8),(2,4,16)]]
    qr,kr,er = [t.detach().clone().requires_grad_() for t in (q,k,emb)]
    pos=torch.arange(4)
    actual=summed._chunk(q,k,emb,pos,pos)
    expected=2*mean._chunk(qr,kr,er,pos,pos)
    torch.testing.assert_close(actual,expected)
    grad=torch.randn_like(actual);actual.backward(grad);expected.backward(grad)
    for x,y in [(q,qr),(k,kr),(emb,er)]:torch.testing.assert_close(x.grad,y.grad)


@pytest.mark.parametrize('key', KEYS[1:]+['gpt_base_head_scaled_embed_no_embed_norm','gpt_base_split_embed_shared_out_no_embed_norm'])
def test_reject_sum_for_other_architectures(key):
    Config,_=get_model(key)
    with pytest.raises(ValueError,match='only by pooled'):
        Config(sum_head_attention=True)


@pytest.mark.parametrize('key,flops,steps',zip(KEYS,[4619614208,4247586816,4619501568,4031318016],[8072,8779,8072,9250]))
def test_d24_budget(key,flops,steps):
    Config,Model=get_model(key)
    with torch.device('meta'):
        model=Model(Config(sequence_len=2048,vocab_size=32768,n_layer=24,n_head=12,n_kv_head=12,n_embd=1536,
                           norm_value_input=True,sum_head_attention=key==KEYS[0]))
    assert model.estimate_flops()==flops
    assert round(3.91e19/(1048576*flops))==steps
