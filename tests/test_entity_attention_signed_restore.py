"""Signed restoration must replace only entity coefficients on the live trajectory."""
from types import SimpleNamespace
import sys
import pytest
import torch
from scripts.inspect import qwen_entity_attention_ablation as core


def fixture():
    generator=torch.Generator().manual_seed(20260922)
    q=torch.randn(1,4,6,3,generator=generator)
    k=torch.randn(1,2,6,3,generator=generator)
    v=torch.randn(1,2,6,3,generator=generator)
    mask=torch.full((1,1,6,6),float('-inf')).triu(1)
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,is_causal=True)
    return module,q,k,v,mask


@pytest.mark.parametrize('positions',[(1,),(1,2)])
@pytest.mark.parametrize('policy',['signed','positive-only'])
@pytest.mark.parametrize('block_intermediate',[False,True])
def test_restoration_matches_independent_weight_replacement(positions,policy,block_intermediate):
    module,q,k,v,mask=fixture()
    keys=k.repeat_interleave(2,dim=1);values=v.repeat_interleave(2,dim=1)
    weights=(q@keys.transpose(-2,-1)*3**-.5+mask).softmax(-1)
    live=weights[:,:,5,list(positions)]
    target=live.clone()
    target[:,::2]*=.5  # Negative corrections must survive in signed mode.
    target[:,1::2]+=.04
    expected_weights=weights.clone()
    expected_weights[:,:,5,list(positions)]=target if policy=='signed' else torch.maximum(live,target)
    if block_intermediate:
        for position in range(positions[-1]+1,5):expected_weights[:,:,position,list(positions)]=0
    expected=(expected_weights@values).transpose(1,2)
    with core.disable_entity_attention(range(0),1,5,6,block_intermediate,
        restore_layers=range(1),ordinary_entity_attention={0:target},
        entity_positions=positions,restore_policy=policy) as state:
        actual,_=core.entity_zero_attention_forward(module,q,k,v,mask,3**-.5)
        trace=state.restore_trace[0]
        delta=torch.tensor(trace['signed_entity_attention_correction_by_token'])
        assert delta.max()>0
        assert (delta.min()<0) if policy=='signed' else (delta.min()==0)
        torch.testing.assert_close(torch.tensor(trace['effective_entity_attention_by_token']),
                                   (target if policy=='signed' else torch.maximum(live,target))[0])
        assert state.restore_applications==1
        assert trace['restore_policy']==policy
    torch.testing.assert_close(actual,expected,atol=3e-7,rtol=2e-6)
    assert core.ABLATION.restore_policy=='positive-only'


@pytest.mark.parametrize('positions',[(1,),(1,2)])
def test_zero_signed_correction_matches_blocked_bitwise(positions):
    module,q,k,v,mask=fixture()
    target=core._final_entity_attention_by_token(module,q,k,mask,3**-.5,5,positions)
    with core.disable_entity_attention(range(0),1,5,6,True,entity_positions=positions):
        blocked,_=core.entity_zero_attention_forward(module,q,k,v,mask,3**-.5)
    with core.disable_entity_attention(range(0),1,5,6,True,entity_positions=positions,
        restore_layers=range(1),ordinary_entity_attention={0:target},restore_policy='signed'):
        noop,_=core.entity_zero_attention_forward(module,q,k,v,mask,3**-.5)
    assert torch.equal(noop,blocked)


def test_policy_resets_after_exception():
    with pytest.raises(RuntimeError):
        with core.disable_entity_attention(range(0),1,5,6,True,restore_policy='signed'):
            raise RuntimeError('intentional')
    assert core.ABLATION.restore_policy=='positive-only'
    assert core.ABLATION.entity_position is None
    with pytest.raises(ValueError,match='unknown'):
        with core.disable_entity_attention(range(0),1,5,6,True,restore_policy='bad'):pass


def test_cli_policy_default_and_signed(monkeypatch):
    monkeypatch.setattr(sys,'argv',['main'])
    assert core.parse_args().restore_entity_attention_policy=='positive-only'
    monkeypatch.setattr(sys,'argv',['main','--restore-entity-attention-start-layer','0','--restore-entity-attention-policy','signed'])
    args=core.parse_args()
    assert args.restore_entity_attention_policy=='signed' and args.restore_entity_attention_start_layer==0
