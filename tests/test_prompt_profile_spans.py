"""Check profile intervention parity and four-slot display alignment."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/inspect'))
import qwen_prompt_attention_profile as profile
import qwen_entity_attention_ablation as core


def encoded(ids, positions):
    return core.EncodedPrompt(0,'test',0,'name','test','test',ids,[str(i) for i in ids],positions[0],entity_positions=positions)


@pytest.mark.parametrize('decode', [False, True])
def test_profile_matches_main_for_full_span_including_hidden_fifth_token(decode):
    torch.manual_seed(42)
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,is_causal=True,
                           o_proj=torch.nn.Linear(12,12,bias=False))
    q=torch.randn(1,4,9,3);k=torch.randn(1,2,9,3);v=torch.randn(1,2,9,3)
    e=encoded(list(range(9)),(1,2,3,4,5))
    if decode:
        q=q[:,:,-1:,:];k=torch.cat([k,k[:,:,:1]],dim=-2);v=torch.cat([v,v[:,:,:1]],dim=-2)
        mask=torch.zeros(1,1,1,10)
    else:
        mask=torch.full((1,1,9,9),float('-inf')).triu(1)
    with core.disable_entity_attention(range(0),1,8,9,True,entity_positions=e.entity_positions):
        expected,_=core.entity_zero_attention_forward(module,q,k,v,mask,3**-0.5)
    with profile.profile_condition('bottleneck_control',e) as state:
        actual,_=profile.explicit_profile_attention_forward(module,q,k,v,mask,3**-0.5)
        assert state.generated_applications==int(decode)
        assert state.intermediate_applications==(0 if decode else 2)
        if not decode:
            assert state.attention_by_layer[0].shape==(4,9)
            torch.testing.assert_close(state.attention_by_layer[0].sum(-1),torch.ones(4))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    # Independent weighted-sum oracle also covers the fifth entity value.
    weights=(q @ k.repeat_interleave(2,1).transpose(-2,-1)*3**-0.5+mask).softmax(-1)
    values=v.repeat_interleave(2,1)
    native=(weights@values).transpose(1,2)
    removal=(weights[...,1:6]@values[...,1:6,:]).transpose(1,2)
    affected=[0] if decode else [6,7]
    native[:,affected]-=removal[:,affected]
    torch.testing.assert_close(actual,native,atol=3e-7,rtol=3e-6)


class ToyTokenizer:
    def __init__(self):self.records={}
    def add(self,prompt,ids,offsets):self.records[prompt]={'input_ids':ids,'offset_mapping':offsets}
    def __call__(self,prompt,**kwargs):return self.records[prompt]
    def decode(self,ids,**kwargs):return f'token_{ids[0]}'


def test_four_slots_missing_values_and_punctuation_boundary():
    tokenizer=ToyTokenizer();template='X {entity}. Q:'
    # A five-token name, a one-token name, and a name whose final token absorbs '. '.
    specs=[('ABCDE',[10,1,2,3,4,5,20,30],[(0,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,9),(9,11)],(1,2,3,4,5)),
           ('Z',[10,6,20,30],[(0,2),(2,3),(3,5),(5,7)],(1,)),
           ('J.',[10,7,8,30],[(0,2),(2,3),(3,6),(6,8)],(1,2))]
    cases=[]
    for i,(name,ids,offsets,positions) in enumerate(specs):
        prompt=template.format(entity=name);tokenizer.add(prompt,ids,offsets)
        cases.append(core.EncodedPrompt(0,'test',i,name,'test',prompt,ids,[str(x) for x in ids],positions[0],entity_positions=positions))
    alignment=profile.build_alignment(tokenizer,cases,template)
    keys=alignment['column_keys'];name_cols=[i for i,k in enumerate(keys) if k[0]=='name']
    assert len(name_cols)==4
    assert [alignment['contributing_name_counts'][i] for i in name_cols]==[3,2,1,1]
    assert alignment['omitted_name_tokens_per_case']==[1,0,0]
    maps=alignment['source_token_indices']
    assert 5 not in maps[0]  # Fifth name token hidden only in the plot.
    assert keys[-2:]==[['suffix',0,2,20],['suffix',2,4,30]]
    assert maps[2][-2:]==[-1,3]  # Missing punctuation does not shift Q:.
    raw=[torch.arange(len(c.ids)).float().reshape(1,-1) for c in cases]
    aligned=torch.stack([profile.align_tokens(v,m) for v,m in zip(raw,maps)])
    assert torch.isnan(aligned[1,0,name_cols[1]])
    mean=profile.masked_mean(aligned)
    assert mean[0,name_cols[3]]==4  # One contributor, not divided by all three.
    assert mean[0,-2]==(6+2)/2  # Excludes missing fused punctuation.
    assert len(raw[0][0])==8 and raw[0][0,5]==5  # Complete raw data retained.


def test_paired_difference_uses_same_mask_and_json_null():
    ordinary=torch.tensor([[[.1,float('nan')]],[[.3,.8]]])
    blocked=torch.tensor([[[.2,float('nan')]],[[.1,.4]]])
    delta=profile.masked_mean(blocked-ordinary)
    torch.testing.assert_close(delta,profile.masked_mean(blocked)-profile.masked_mean(ordinary))
    assert profile.json_matrix(torch.tensor([[float('nan'),.5]]))==[[None,.5]]
