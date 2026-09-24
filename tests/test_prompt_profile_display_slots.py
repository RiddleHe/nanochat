from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/inspect'))
import qwen_prompt_attention_profile as profile
from qwen_entity_attention_ablation import EncodedPrompt
class Tokenizer:
 def __call__(self,prompt,**kwargs):
  return {'input_ids':[10,1,2,20,30] if 'AB' in prompt else [10,3,4,20,30], 'offset_mapping':[(0,2),(2,3),(3,4),(4,6),(6,8)]}
 def decode(self,ids,**kwargs):return str(ids[0])
def test_exact_two_slots_preserve_every_prompt_token():
 tokenizer=Tokenizer();template='X {entity}. Q:';cases=[]
 for i,name in enumerate(('AB','CD')):
  prompt=template.format(entity=name);ids=tokenizer(prompt)['input_ids']
  cases.append(EncodedPrompt(0,'test',i,name,'test',prompt,ids,list(map(str,ids)),1,entity_positions=(1,2)))
 result=profile.build_alignment(tokenizer,cases,template,display_name_slots=2)
 assert result['display_name_slots']==2
 assert result['column_keys'][1:3]==[['name',0],['name',1]]
 assert result['source_token_indices']==[list(range(5)),list(range(5))]
 assert result['contributing_name_counts']==[2]*5
 assert result['omitted_name_tokens_per_case']==[0,0]
 assert len(profile.build_alignment(tokenizer,cases,template)['token_labels'])==7
 with pytest.raises(ValueError):profile.build_alignment(tokenizer,cases,template,0)
def test_cli_slot_count(monkeypatch):
 monkeypatch.setattr(sys,'argv',['profile','--display-name-slots','2'])
 assert profile.parse_args().display_name_slots==2
