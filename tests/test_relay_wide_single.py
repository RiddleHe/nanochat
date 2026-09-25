import json
from pathlib import Path
import tempfile
import unittest
from scripts.inspect import qwen_relay_wide_single as m


class WideTests(unittest.TestCase):
    def test_window_counts_and_boundaries(self):
        self.assertEqual([len(m.windows(k)) for k in m.WIDTHS],[30,28,26,24])
        for width in m.WIDTHS:
            self.assertEqual(m.windows(width)[0],(0,width))
            self.assertEqual(m.windows(width)[-1],(35-width,35))
            for s,t in m.windows(width):self.assertEqual(len(range(s+1,t+1)),width)

    def test_invalid_width(self):
        with self.assertRaises(ValueError):m.windows(4)
        with self.assertRaises(ValueError):m.windows(36)

    def test_window_protocol_roundtrip(self):
        spec={str(k):[[s,t] for s,t in m.windows(k)] for k in m.WIDTHS}
        self.assertEqual(spec,json.loads(json.dumps(spec)))

    def setUp(self):
        self.labels={'by_signature':{},'by_output':{},'whole_response_aliases':{
            'Newton':['Newton'],'Isaac Newton':['Newton'],'Einstein':['Einstein'],
            'ambiguous':['Newton','Einstein']}}
        self.row={'entity':'Newton','clean_output':'Newton','stop_reason':'eos'}

    def test_explicit_full_identity(self):
        self.assertTrue(m.control_decision(self.row,self.labels)['eligible'])
        self.row['clean_output']='Isaac Newton'
        self.assertTrue(m.control_decision(self.row,self.labels)['eligible'])

    def test_no_substring_or_truncated_success(self):
        for text,stop in [('Newton was a scientist','eos'),('Newton','max_new_tokens'),('ambiguous','eos')]:
            self.row.update(clean_output=text,stop_reason=stop)
            self.assertIsNone(m.control_decision(self.row,self.labels)['eligible'])

    def test_other_reviewed_identity_fails_control(self):
        self.row['clean_output']='Einstein'
        self.assertFalse(m.control_decision(self.row,self.labels)['eligible'])

    def test_extra_explicit_review_precedes_alias(self):
        extra={'by_signature':{m.signature('Newton','Newton','eos'):{'category':'partial_name'}}}
        self.assertFalse(m.control_decision(self.row,self.labels,extra)['eligible'])

    def test_unknown_stays_pending(self):
        self.row['clean_output']='some unreviewed output'
        self.assertIsNone(m.control_decision(self.row,self.labels)['eligible'])

    def test_single_token_fixed_selection(self):
        names=[f'name{i}' for i in range(100)]
        cases=[{'template':0,'entity':names[i],'eligible':True} for i in range(100)]
        pair={'id':'p','template':0,'donor':names[0],'recipient':names[1],
              'donor_id':0,'recipient_id':1,'span_length':1}
        selection={'cases':cases,'pairs':[pair,dict(pair,id='double',span_length=2)]}
        self.assertEqual(len(m.select_pairs(selection,names)),1)
        pair['recipient_id']=2
        with self.assertRaises(AssertionError):m.select_pairs(selection,names)

    def test_resume_duplicate_rows_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rows.jsonl'
            self.assertEqual(m.read_rows(path),{})
            path.write_text('{"id":"a"}\n')
            self.assertEqual(set(m.read_rows(path)),{'a'})
            path.write_text('{"id":"a"}\n{"id":"a"}\n')
            with self.assertRaises(RuntimeError):m.read_rows(path)

    def test_tiny_model_wide_identity(self):
        import torch
        from types import SimpleNamespace
        from transformers import AttentionInterface, Qwen3Config, Qwen3ForCausalLM
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from scripts.inspect import qwen_relay_supplied_common as c
        from scripts.inspect import qwen_relay_paper_runtime as w
        torch.manual_seed(73);torch.set_num_threads(1)
        cfg=Qwen3Config(vocab_size=100,hidden_size=32,intermediate_size=48,
                       num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,
                       head_dim=8,max_position_embeddings=128,bos_token_id=0,
                       eos_token_id=1,pad_token_id=0,attention_dropout=0.)
        cfg._attn_implementation='sdpa'
        model=Qwen3ForCausalLM(cfg).eval();layers=list(model.model.layers)
        enc=c.ab.EncodedPrompt(0,'test',0,'entity','test','',list(range(2,9)),
                                list('abcdefg'),1,entity_positions=(1,))
        tok=SimpleNamespace(pad_token_id=0,eos_token_id=1,decode=lambda ids,**kw:str(ids))
        hook=c.configure_model(model);device=torch.device('cpu')
        AttentionInterface.register('sdpa',c.ab.entity_zero_attention_forward)
        try:
            own,_=c.capture(model,layers,enc,device)
            for width in m.WIDTHS:
                for s,t in (m.windows(width)[0],m.windows(width)[-1]):
                    disabled=w.outside_layers(36,s,t)
                    self.assertEqual(set(range(36))-set(disabled),set(range(s+1,t+1)))
                    for policy in ((),disabled):
                        base=c.generate(model,tok,layers,enc,device,policy)
                        patched=c.run_relay(model,tok,layers,enc,device,own[s],s,t,policy)
                        self.assertEqual(base['generated_token_ids'],patched['generated_token_ids'])
        finally:
            hook.remove();AttentionInterface.register('sdpa',sdpa_attention_forward)


if __name__=='__main__':unittest.main()
